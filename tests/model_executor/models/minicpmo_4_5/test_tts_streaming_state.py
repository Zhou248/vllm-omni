# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""L1 tests for MiniCPM-o 4.5 Talker request-local streaming state."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    MiniCPMO45OmniTTSForConditionalGeneration,
    _MiniCPMO45StreamingState,
    _select_async_text_span,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _bare_talker(monkeypatch) -> MiniCPMO45OmniTTSForConditionalGeneration:
    talker = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    nn.Module.__init__(talker)
    talker._stream_states = {}
    talker._tts_streaming_generator_cls = None
    talker._tts_gen_logits = None
    talker.audio_tokenizer = None
    talker._outer_step_finished = False
    monkeypatch.setattr(talker, "_lazy_init_tts", lambda: None)
    return talker


def test_old_checkpoint_fallback_accumulates_until_terminal(
    monkeypatch,
) -> None:
    talker = _bare_talker(monkeypatch)
    captured = {}

    def _generate(tokens, hidden):
        captured["tokens"] = tokens.clone()
        captured["hidden"] = hidden.clone()
        return np.array([0.25, -0.25], dtype=np.float32)

    monkeypatch.setattr(talker, "generate_speech", _generate)

    first = talker._generate_speech_async_chunk(
        "r",
        torch.tensor([10, 11]),
        torch.tensor([[1.0], [2.0]]),
        text_finished=False,
    )
    assert first is None
    assert "r" in talker._stream_states

    final = talker._generate_speech_async_chunk(
        "r",
        torch.tensor([12]),
        torch.tensor([[3.0]]),
        text_finished=True,
    )
    assert final.tolist() == [0.25, -0.25]
    assert captured["tokens"].tolist() == [10, 11, 12]
    assert captured["hidden"].tolist() == [[1.0], [2.0], [3.0]]
    assert "r" not in talker._stream_states


def test_native_generator_state_is_reused_across_text_chunks(
    monkeypatch,
) -> None:
    talker = _bare_talker(monkeypatch)

    class _Generator:
        def __init__(self):
            self.calls = []

        def generate_with_buffer(self, *, condition, text_finished):
            self.calls.append((condition.clone(), text_finished))
            return iter([(torch.tensor([[1, 2]]), text_finished)])

    generator = _Generator()
    state = _MiniCPMO45StreamingState(tts_generator=generator)
    monkeypatch.setattr(talker, "_new_streaming_state", lambda: state)
    monkeypatch.setattr(
        talker,
        "_build_streaming_condition",
        lambda tokens, hidden: hidden.unsqueeze(0),
    )
    monkeypatch.setattr(
        talker,
        "_stream_audio_tokens",
        lambda state, chunks, *, text_finished: np.array(
            [len(chunks), float(text_finished)],
            dtype=np.float32,
        ),
    )

    first = talker._generate_speech_async_chunk(
        "r",
        torch.tensor([10]),
        torch.tensor([[1.0]]),
        text_finished=False,
    )
    final = talker._generate_speech_async_chunk(
        "r",
        torch.tensor([11]),
        torch.tensor([[2.0]]),
        text_finished=True,
    )

    assert first.tolist() == [1.0, 0.0]
    assert final.tolist() == [1.0, 1.0]
    assert [finished for _, finished in generator.calls] == [False, True]
    assert "r" not in talker._stream_states


def test_token2wav_uses_lookahead_and_flushes_terminal_tail(
    monkeypatch,
) -> None:
    talker = _bare_talker(monkeypatch)

    class _AudioTokenizer:
        def __init__(self):
            self.flow = SimpleNamespace(pre_lookahead_len=3)
            self.stream_cache = None
            self.hift_cache_dict = None
            self.calls = []

        def stream(
            self,
            tokens,
            prompt_wav,
            *,
            last_chunk,
            return_waveform,
        ):
            self.calls.append((list(tokens), prompt_wav, last_chunk, return_waveform))
            return np.array([len(tokens)], dtype=np.float32)

    tokenizer = _AudioTokenizer()
    talker.audio_tokenizer = tokenizer
    state = _MiniCPMO45StreamingState(
        stream_cache={"flow": 1},
        hift_cache_dict={"hift": 2},
    )

    first = talker._stream_audio_tokens(
        state,
        [(torch.arange(25).reshape(1, -1), False)],
        text_finished=False,
    )
    final = talker._stream_audio_tokens(
        state,
        [(torch.arange(5).reshape(1, -1), True)],
        text_finished=True,
    )

    assert first.tolist() == [28.0]
    assert final.tolist() == [8.0]
    assert len(tokenizer.calls[0][0]) == 28
    assert tokenizer.calls[0][2] is False
    assert len(tokenizer.calls[1][0]) == 8
    assert tokenizer.calls[1][2] is True
    assert state.vocoder_buffer == []


def test_finish_hook_cleans_single_state_with_internal_id(
    monkeypatch,
) -> None:
    talker = _bare_talker(monkeypatch)
    talker._stream_states["external-id"] = _MiniCPMO45StreamingState()

    talker.on_requests_finished(["different-internal-id"])

    assert talker._stream_states == {}


def test_outer_logits_keep_alive_then_stop(monkeypatch) -> None:
    talker = _bare_talker(monkeypatch)
    hidden = torch.zeros((1, 4))

    talker._outer_step_finished = False
    keep_alive_logits = talker.compute_logits(hidden)
    assert keep_alive_logits.argmax(dim=-1).tolist() == [0]

    talker._outer_step_finished = True
    terminal_logits = talker.compute_logits(hidden)
    assert terminal_logits.argmax(dim=-1).tolist() == [1]


def test_async_text_span_accepts_delta_and_cumulative_payloads() -> None:
    metadata = {
        "minicpmo45_text_start": 2,
        "minicpmo45_text_end": 4,
    }
    delta_tokens, delta_hidden = _select_async_text_span(
        torch.tensor([12, 13]),
        torch.tensor([[12.0], [13.0]]),
        metadata,
    )
    assert delta_tokens.tolist() == [12, 13]
    assert delta_hidden.tolist() == [[12.0], [13.0]]

    cumulative_tokens, cumulative_hidden = _select_async_text_span(
        torch.tensor([10, 11, 12, 13]),
        torch.tensor([[10.0], [11.0], [12.0], [13.0]]),
        metadata,
    )
    assert cumulative_tokens.tolist() == [12, 13]
    assert cumulative_hidden.tolist() == [[12.0], [13.0]]


def test_async_empty_terminal_span_drops_accumulated_payload() -> None:
    tokens, hidden = _select_async_text_span(
        torch.tensor([10, 11]),
        torch.tensor([[10.0], [11.0]]),
        {
            "minicpmo45_text_start": 2,
            "minicpmo45_text_end": 2,
        },
    )
    assert tokens is None
    assert hidden is None
