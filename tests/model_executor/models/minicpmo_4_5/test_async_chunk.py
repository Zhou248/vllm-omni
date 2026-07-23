# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L1 tests for MiniCPM-o 4.5 Thinker -> Talker chunk transfer."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.stage_input_processors.minicpmo_4_5_omni import (
    llm2tts_async_chunk,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_TTS_BOS = 151703
_TTS_EOS = 151704


def _transfer_manager(chunk_size: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        request_payload={},
        connector=SimpleNamespace(
            config={
                "extra": {
                    "thinker_text_chunk_size": chunk_size,
                }
            }
        ),
    )


def _request(
    output_ids: list[int],
    *,
    computed: int,
    prompt_ids: list[int] | None = None,
    placeholders: int = 0,
) -> SimpleNamespace:
    if prompt_ids is None:
        prompt_ids = [100, 100]
    return SimpleNamespace(
        request_id="internal-r",
        external_req_id="r",
        prompt_token_ids=prompt_ids,
        output_token_ids=output_ids,
        num_computed_tokens=computed,
        num_output_placeholders=placeholders,
    )


def _call(
    transfer_manager: SimpleNamespace,
    output_ids: list[int],
    *,
    computed: int,
    hidden_values: list[float],
    finished: bool = False,
    placeholders: int = 0,
    prompt_ids: list[int] | None = None,
):
    hidden = torch.tensor(hidden_values, dtype=torch.float32).unsqueeze(-1)
    return llm2tts_async_chunk(
        transfer_manager=transfer_manager,
        multimodal_output={"hidden": hidden},
        request=_request(
            output_ids,
            computed=computed,
            placeholders=placeholders,
            prompt_ids=prompt_ids,
        ),
        is_finished=finished,
    )


def test_emits_aligned_text_hidden_chunks() -> None:
    tm = _transfer_manager(chunk_size=2)

    # Prefill computes the two prompt positions. The freshly sampled TTS BOS
    # has no hidden state yet and must not be sent.
    assert (
        _call(
            tm,
            [_TTS_BOS],
            computed=2,
            hidden_values=[100.0, 101.0],
        )
        is None
    )

    # Processing BOS activates the TTS region; token 10 is still freshly
    # sampled and therefore has no aligned hidden row yet.
    assert (
        _call(
            tm,
            [_TTS_BOS, 10],
            computed=3,
            hidden_values=[200.0],
        )
        is None
    )

    # Process token 10, then token 11. The chunk is emitted only after both
    # have real hidden rows.
    assert (
        _call(
            tm,
            [_TTS_BOS, 10, 11],
            computed=4,
            hidden_values=[10.0],
        )
        is None
    )
    payload = _call(
        tm,
        [_TTS_BOS, 10, 11, 12],
        computed=5,
        hidden_values=[11.0],
    )

    assert payload is not None
    assert payload.ids.output == [10, 11]
    assert payload.hidden_states.output.tolist() == [[10.0], [11.0]]
    assert payload.meta.finished.item() is False
    assert payload.request_id == "r"
    assert payload.kv_metadata == {
        "minicpmo45_text_start": 0,
        "minicpmo45_text_end": 2,
    }


def test_terminal_flushes_tail_and_excludes_tts_markers() -> None:
    tm = _transfer_manager(chunk_size=10)
    assert (
        _call(
            tm,
            [_TTS_BOS, 20],
            computed=3,
            hidden_values=[1.0],
        )
        is None
    )

    payload = _call(
        tm,
        [_TTS_BOS, 20, _TTS_EOS],
        computed=4,
        hidden_values=[20.0],
        finished=True,
    )

    assert payload is not None
    assert payload.ids.output == [20]
    assert payload.hidden_states.output.tolist() == [[20.0]]
    assert payload.meta.finished.item() is True
    assert payload.kv_metadata == {
        "minicpmo45_text_start": 0,
        "minicpmo45_text_end": 1,
    }


def test_tts_bos_in_serving_prompt_activates_first_generated_token() -> None:
    tm = _transfer_manager(chunk_size=1)
    prompt_ids = [100, _TTS_BOS]

    # Prefill samples token 50 but only prompt positions have hidden rows.
    assert (
        _call(
            tm,
            [50],
            computed=2,
            hidden_values=[1.0, 2.0],
            prompt_ids=prompt_ids,
        )
        is None
    )

    payload = _call(
        tm,
        [50, 51],
        computed=3,
        hidden_values=[50.0],
        prompt_ids=prompt_ids,
    )
    assert payload.ids.output == [50]
    assert payload.hidden_states.output.tolist() == [[50.0]]


def test_finished_without_tts_region_sends_empty_terminal_marker() -> None:
    tm = _transfer_manager()
    payload = _call(
        tm,
        [42],
        computed=2,
        hidden_values=[1.0, 2.0],
        finished=True,
    )

    assert payload is not None
    assert payload.ids.output == []
    assert payload.hidden_states.output is None
    assert payload.meta.finished.item() is True
    assert payload.meta.decode_flag is False
    assert payload.kv_metadata == {
        "minicpmo45_text_start": 0,
        "minicpmo45_text_end": 0,
    }


def test_async_placeholders_are_not_mapped_to_hidden_rows() -> None:
    tm = _transfer_manager(chunk_size=1)

    # confirmed=3-1=2: only prompt positions are computed, so the BOS token
    # must not consume the final prompt hidden row.
    payload = _call(
        tm,
        [_TTS_BOS, 10],
        computed=3,
        placeholders=1,
        hidden_values=[100.0, 101.0],
    )
    assert payload is None

    # Once BOS is genuinely computed, it activates the region but token 10
    # remains pending.
    payload = _call(
        tm,
        [_TTS_BOS, 10],
        computed=3,
        hidden_values=[200.0],
    )
    assert payload is None


def test_preemption_replay_does_not_duplicate_emitted_tokens() -> None:
    tm = _transfer_manager(chunk_size=1)
    assert (
        _call(
            tm,
            [_TTS_BOS, 30],
            computed=3,
            hidden_values=[1.0],
        )
        is None
    )
    first = _call(
        tm,
        [_TTS_BOS, 30, 31],
        computed=4,
        hidden_values=[30.0],
    )
    assert first.ids.output == [30]

    # Recomputed history at the same confirmed watermark is ignored.
    replay = _call(
        tm,
        [_TTS_BOS, 30, 31],
        computed=4,
        hidden_values=[30.0],
    )
    assert replay is None

    second = _call(
        tm,
        [_TTS_BOS, 30, 31, 32],
        computed=5,
        hidden_values=[31.0],
    )
    assert second.ids.output == [31]
    assert second.kv_metadata == {
        "minicpmo45_text_start": 1,
        "minicpmo45_text_end": 2,
    }
