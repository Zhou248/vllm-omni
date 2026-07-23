# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Stage input processors for MiniCPM-o 4.5: Thinker (LLM) -> Talker (TTS).

This is the original vLLM-Omni bridge: it converts the thinker stage's
hidden states + token ids into the talker stage's prompt payload. The
talker model itself is adapted from openbmb/MiniCPM-o-4_5 (see the headers
on vllm_omni/model_executor/models/minicpmo_4_5/*.py).

Two transfer modes are supported:

* ``llm2tts`` keeps the original turn-level path and sends the complete
  Thinker result after text generation finishes.
* ``llm2tts_async_chunk`` follows MiniCPM-o 4.5's native streaming cadence:
  it sends aligned text-token/hidden-state chunks while the Thinker is still
  decoding, allowing the Talker and Token2Wav stages to overlap with it.
"""

import logging
from collections.abc import Mapping
from typing import Any

import torch
from vllm.inputs import TextPrompt

from vllm_omni.data_entry_keys import (
    HiddenStatesStruct,
    IdsStruct,
    MetaStruct,
    OmniPayload,
    OmniPayloadStruct,
)
from vllm_omni.engine import OmniEngineCoreRequest
from vllm_omni.inputs.data import OmniTokensPrompt

logger = logging.getLogger(__name__)

_TTS_SPECIAL_TOKEN_PAIRS = {
    151703: 151704,  # MiniCPM-o 4.5
    151691: 151692,  # MiniCPM-o 2.6-compatible checkpoints
}
_DEFAULT_THINKER_CHUNK_SIZE = 10
_ASYNC_STATE_KEY = "_minicpmo45_async_bridge"


def _as_list(value: Any) -> list[Any]:
    """Convert vLLM's ConstantList/tensor-like token history to a list."""
    if hasattr(value, "_x"):
        return list(value._x)
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return value.reshape(-1).tolist()
    return list(value)


def _connector_extra(transfer_manager: Any) -> Mapping[str, Any]:
    connector = getattr(transfer_manager, "connector", None)
    config = getattr(connector, "config", {})
    if not isinstance(config, Mapping):
        config = {
            "extra": getattr(config, "extra", {}),
        }
    extra = config.get("extra", {})
    return extra if isinstance(extra, Mapping) else {}


def _thinker_chunk_size(transfer_manager: Any) -> int:
    raw = _connector_extra(transfer_manager).get(
        "thinker_text_chunk_size",
        _DEFAULT_THINKER_CHUNK_SIZE,
    )
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        logger.warning(
            "Invalid thinker_text_chunk_size=%r; using %d",
            raw,
            _DEFAULT_THINKER_CHUNK_SIZE,
        )
        return _DEFAULT_THINKER_CHUNK_SIZE


def _extract_thinker_hidden_chunk(
    multimodal_output: OmniPayload | Mapping[str, Any] | None,
) -> torch.Tensor | None:
    """Return this scheduler step's hidden rows from a runner payload."""
    if not isinstance(multimodal_output, Mapping):
        return None
    for key in ("hidden", "latent"):
        value = multimodal_output.get(key)
        if isinstance(value, torch.Tensor):
            if value.ndim == 3 and value.shape[0] == 1:
                value = value.squeeze(0)
            return value if value.ndim == 2 else None
    return None


def llm2tts_async_chunk(
    transfer_manager: Any,
    multimodal_output: OmniPayload | dict[str, Any] | None,
    request: OmniEngineCoreRequest,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Send aligned Thinker text/hidden chunks to the Talker.

    The hidden rows produced by a vLLM decode step cover already-computed
    sequence positions. The newest sampled token in ``request.output_token_ids``
    does not have a hidden row until the following decode step. We therefore
    map rows through the confirmed computed-token watermark rather than slicing
    the end of the token list. This also makes replay after preemption
    idempotent.
    """

    request_id = str(getattr(request, "external_req_id", None) or getattr(request, "request_id", ""))
    request_payload = getattr(transfer_manager, "request_payload", None)
    if request_payload is None:
        request_payload = {}
        transfer_manager.request_payload = request_payload

    state_wrapper = request_payload.get(request_id)
    if not isinstance(state_wrapper, dict) or _ASYNC_STATE_KEY not in state_wrapper:
        state_wrapper = {
            _ASYNC_STATE_KEY: {
                "tts_active": False,
                "tts_requested": False,
                "tts_eos_id": None,
                "prompt_scanned": False,
                "last_output_index": 0,
                "emitted_text_tokens": 0,
                "token_buffer": [],
                "hidden_buffer": [],
                "terminal_sent": False,
            }
        }
        request_payload[request_id] = state_wrapper
    state = state_wrapper[_ASYNC_STATE_KEY]

    if bool(state.get("terminal_sent", False)):
        return None

    hidden_chunk = _extract_thinker_hidden_chunk(multimodal_output)
    prompt_token_ids = [int(token_id) for token_id in _as_list(getattr(request, "prompt_token_ids", []))]
    output_token_ids = [int(token_id) for token_id in _as_list(getattr(request, "output_token_ids", []))]
    prompt_len = len(prompt_token_ids)
    if not bool(state.get("prompt_scanned", False)):
        # ``use_tts_template=true`` appends <|tts_bos|> to the assistant
        # prefix, so the normal online-serving path activates TTS from the
        # prompt rather than from the generated output.
        for token_id in prompt_token_ids:
            eos_id = _TTS_SPECIAL_TOKEN_PAIRS.get(token_id)
            if eos_id is not None:
                state["tts_active"] = True
                state["tts_requested"] = True
                state["tts_eos_id"] = eos_id
            elif bool(state.get("tts_active", False)) and token_id == state.get("tts_eos_id"):
                state["tts_active"] = False
        state["prompt_scanned"] = True
    num_computed = int(getattr(request, "num_computed_tokens", 0) or 0)
    placeholders = int(getattr(request, "num_output_placeholders", 0) or 0)
    confirmed = max(0, num_computed - placeholders)

    if hidden_chunk is not None and hidden_chunk.shape[0] > 0:
        num_rows = int(hidden_chunk.shape[0])
        absolute_start = max(0, confirmed - num_rows)
        output_start = max(
            int(state.get("last_output_index", 0)),
            absolute_start - prompt_len,
            0,
        )
        output_end = min(
            max(0, confirmed - prompt_len),
            len(output_token_ids),
        )

        for output_index in range(output_start, output_end):
            absolute_position = prompt_len + output_index
            hidden_row = absolute_position - absolute_start
            if not 0 <= hidden_row < num_rows:
                continue

            token_id = output_token_ids[output_index]
            if token_id < 0:
                continue
            if not bool(state.get("tts_active", False)):
                eos_id = _TTS_SPECIAL_TOKEN_PAIRS.get(token_id)
                if eos_id is None:
                    continue
                state["tts_active"] = True
                state["tts_requested"] = True
                state["tts_eos_id"] = eos_id
                continue

            if token_id == state.get("tts_eos_id"):
                state["tts_active"] = False
                continue

            state["token_buffer"].append(token_id)
            state["hidden_buffer"].append(hidden_chunk[hidden_row].detach().cpu())

        state["last_output_index"] = max(
            int(state.get("last_output_index", 0)),
            output_end,
        )

    chunk_size = _thinker_chunk_size(transfer_manager)
    buffered = len(state["token_buffer"])
    should_emit = buffered >= chunk_size or (is_finished and buffered > 0)

    if should_emit:
        emit_count = chunk_size if buffered >= chunk_size and not is_finished else buffered
        token_chunk = list(state["token_buffer"][:emit_count])
        hidden_rows = state["hidden_buffer"][:emit_count]
        del state["token_buffer"][:emit_count]
        del state["hidden_buffer"][:emit_count]
        terminal = bool(is_finished and not state["token_buffer"])
        text_start = int(state.get("emitted_text_tokens", 0))
        text_end = text_start + emit_count
        state["emitted_text_tokens"] = text_end
        if terminal:
            state["terminal_sent"] = True
        return OmniPayloadStruct(
            request_id=request_id,
            ids=IdsStruct(output=token_chunk),
            hidden_states=HiddenStatesStruct(
                output=torch.stack(hidden_rows, dim=0).contiguous(),
            ),
            meta=MetaStruct(
                finished=torch.tensor(terminal, dtype=torch.bool),
                decode_flag=bool(state.get("tts_requested", False)),
            ),
            # Legacy OmniChunkTransferAdapter gives the Talker this delta,
            # while the unified AR connector may give it the accumulated
            # payload. The absolute text span lets the consumer handle both
            # without synthesizing a previously consumed chunk twice.
            kv_metadata={
                "minicpmo45_text_start": text_start,
                "minicpmo45_text_end": text_end,
            },
        )

    if is_finished:
        state["terminal_sent"] = True
        text_offset = int(state.get("emitted_text_tokens", 0))
        return OmniPayloadStruct(
            request_id=request_id,
            ids=IdsStruct(output=[]),
            hidden_states=HiddenStatesStruct(output=None),
            meta=MetaStruct(
                finished=torch.tensor(True, dtype=torch.bool),
                decode_flag=bool(state.get("tts_requested", False)),
            ),
            kv_metadata={
                "minicpmo45_text_start": text_offset,
                "minicpmo45_text_end": text_offset,
            },
        )

    return None


def llm2tts(
    source_outputs: list[Any],
    prompt: OmniTokensPrompt | TextPrompt | dict | list | None = None,
    requires_multimodal_data: bool = False,
    streaming_context: Any | None = None,
):
    """Convert thinker stage output to talker stage input for MiniCPMO Omni.

    The signature matches the framework's ``custom_process_input_func`` call
    convention used by ``StageEngineCoreClientBase.process_engine_inputs``:

        (source_outputs, prompt, requires_multimodal_data, streaming_context)

    ``source_outputs`` is the already-resolved list of upstream engine
    outputs (one entry per request), so we do not need to look anything up
    via ``stage_list[source_stage_id].engine_outputs``.

    Extracts from thinker output:
      - Full hidden states (prompt + generated) for speaker embedding extraction
      - Prompt token IDs (for finding spk_bos/spk_eos positions)
      - Generated token IDs (for decoding TTS text)

    The talker model will:
      1. Find <|spk_bos|>/<|spk_eos|> positions in prompt_token_ids
      2. Extract speaker embedding from hidden states at those positions
      3. Decode generated text and extract TTS content
      4. Run ConditionalChatTTS pipeline
    """
    del streaming_context  # not used by MiniCPM-o 4.5 turn-taking pipeline

    if not source_outputs:
        raise ValueError("source_outputs cannot be empty")

    llm_outputs = source_outputs
    tts_inputs = []

    if not isinstance(prompt, list):
        prompt = [prompt]

    multi_modal_data = {
        llm_output.request_id: p.get("multi_modal_data", None) if isinstance(p, dict) else None
        for llm_output, p in zip(llm_outputs, prompt)
    }

    for i, llm_output in enumerate(llm_outputs):
        output = llm_output.outputs[0]
        prompt_token_ids = llm_output.prompt_token_ids
        llm_output_ids = output.token_ids
        prompt_token_ids_len = len(prompt_token_ids)

        latent = output.multimodal_output.get("latent", None)
        if latent is None:
            latent = output.hidden_states if hasattr(output, "hidden_states") else None
            if latent is None:
                raise ValueError("No latent or hidden_states found in thinker output")

        thinker_hidden_states = latent.clone().detach()

        # Split hidden states: prompt portion has speaker embedding,
        # generated portion has the text content
        prompt_hidden = thinker_hidden_states[:prompt_token_ids_len].to(torch.float32)

        # Extract decoded text from thinker output for TTS text extraction
        thinker_text = getattr(output, "text", "") or ""

        # Build full token sequence and extract TTS region
        full_token_ids = list(prompt_token_ids) + (
            list(llm_output_ids) if not isinstance(llm_output_ids, list) else llm_output_ids
        )
        full_hidden = thinker_hidden_states.to(torch.float32)

        # Detect TTS token IDs (4.5: 151703/151704, 2.6: 151691/151692)
        tts_bos_id, tts_eos_id = 151691, 151692
        for _id in [151703, 151704]:
            if _id in full_token_ids:
                tts_bos_id, tts_eos_id = 151703, 151704
                break

        tts_bos_idx = tts_eos_idx = None
        for idx_t, tid in enumerate(full_token_ids):
            if tid == tts_bos_id:
                tts_bos_idx = idx_t + 1
            elif tid == tts_eos_id:
                tts_eos_idx = idx_t

        tts_token_ids_slice = tts_hidden_slice = None
        if tts_bos_idx is not None and full_hidden.shape[0] > tts_bos_idx:
            end_idx = tts_eos_idx if tts_eos_idx is not None else full_hidden.shape[0]
            tts_token_ids_slice = torch.tensor(full_token_ids[tts_bos_idx:end_idx], dtype=torch.long)
            tts_hidden_slice = full_hidden[tts_bos_idx:end_idx]

        additional_information = {
            "prompt_embeds": prompt_hidden,
            "prompt_token_ids": list(prompt_token_ids),
            "llm_output_token_ids": list(llm_output_ids) if not isinstance(llm_output_ids, list) else llm_output_ids,
            "llm_output_text": [thinker_text],
        }
        if tts_token_ids_slice is not None:
            additional_information["tts_token_ids"] = tts_token_ids_slice
        if tts_hidden_slice is not None:
            additional_information["tts_hidden_states"] = tts_hidden_slice

        # Minimal prompt token IDs: the talker's AR framework needs *some* tokens
        # to do a single prefill step. We use [BOS, PAD, EOS] as a dummy.
        tts_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[1, 0, 2],
                additional_information=additional_information,
                multi_modal_data=(
                    multi_modal_data[llm_output.request_id]
                    if requires_multimodal_data and multi_modal_data.get(llm_output.request_id) is not None
                    else None
                ),
                mm_processor_kwargs=None,
            )
        )

    return tts_inputs
