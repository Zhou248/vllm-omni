# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict, state-explicit batching for MiniCPM-o 4.5 Token2wav."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

_SILENCE_TOKEN = 4218
logger = init_logger(__name__)


def _autocast_disabled(device: torch.device):
    """Disable any enclosing autocast region on ``device``.

    ``torch.amp.autocast`` resolves the autocast dtype for ``device_type``
    while constructing the context, which raises on accelerators (e.g. Ascend
    NPU) that never registered autocast support. Degrade to a no-op there: an
    enclosing region can only exist on a device type torch already knows.
    """
    try:
        return torch.amp.autocast(device.type, enabled=False)
    except (RuntimeError, TypeError, ValueError):
        return nullcontext()


def tensor_signature(value: torch.Tensor) -> tuple[tuple[int, ...], str, str]:
    return tuple(value.shape), str(value.dtype), value.device.type


def state_shape_signature(state: BatchedToken2WavState) -> tuple[Any, ...]:
    flow = tuple((name, tensor_signature(state.flow_cache[name])) for name in sorted(state.flow_cache))
    hift = tuple((name, tensor_signature(state.hift_cache[name])) for name in sorted(state.hift_cache))
    return flow, hift


@dataclass(frozen=True)
class PromptFeatures:
    speech_tokens: torch.Tensor
    speaker_embedding: torch.Tensor
    mels: torch.Tensor


@dataclass(frozen=True)
class BatchedToken2WavState:
    flow_cache: dict[str, torch.Tensor]
    hift_cache: dict[str, torch.Tensor]


@dataclass
class _CapturedDeviceGraph:
    graph: Any
    static_inputs: tuple[torch.Tensor, ...]
    static_outputs: tuple[torch.Tensor, ...]

    def replay(self, inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
        for static, current in zip(self.static_inputs, inputs, strict=True):
            static.copy_(current)
        self.graph.replay()
        # Graph outputs are persistent buffers and are overwritten by the next
        # replay. Request-owned streaming state must never alias them.
        return tuple(output.detach().clone() for output in self.static_outputs)


class BatchedToken2Wav(nn.Module):
    """Drive Token2wav's modules with dynamically-sized, request-owned caches.

    This class intentionally never calls ``Token2wav.stream`` or
    ``Token2wav.__call__``. The upstream object is used only as a one-time
    asset loader and prompt feature extractor.
    """

    def __init__(
        self,
        token2wav: Any,
        *,
        enable_npu_graphs: bool = False,
        max_npu_graphs: int = 32,
    ):
        super().__init__()
        self._token2wav = token2wav
        self.flow = token2wav.flow
        self.hift = token2wav.hift
        self.float16 = bool(token2wav.float16)
        self.n_timesteps = int(token2wav.n_timesteps)
        self.mel_cache_len = int(token2wav.mel_cache_len)
        self.source_cache_len = int(token2wav.source_cache_len)
        self.register_buffer(
            "speech_window",
            token2wav.speech_window.detach().clone(),
            persistent=False,
        )
        self._prompt_features: dict[tuple[str, str], PromptFeatures] = {}
        self._enable_npu_graphs = bool(enable_npu_graphs)
        self._max_npu_graphs = max(0, int(max_npu_graphs))
        self._npu_graphs: dict[tuple[Any, ...], _CapturedDeviceGraph] = {}
        self._failed_npu_graphs: set[tuple[Any, ...]] = set()
        self._npu_graph_hits = 0
        self._npu_graph_pool: Any = None

    @staticmethod
    def _npu_graph_supported() -> bool:
        npu = getattr(torch, "npu", None)
        return npu is not None and all(
            hasattr(npu, name)
            for name in (
                "NPUGraph",
                "graph",
                "graph_pool_handle",
                "synchronize",
            )
        )

    @staticmethod
    def _npu_stream_is_capturing() -> bool:
        npu = getattr(torch, "npu", None)
        is_capturing = getattr(npu, "is_current_stream_capturing", None)
        if not callable(is_capturing):
            return False
        try:
            return bool(is_capturing())
        except (RuntimeError, TypeError):
            return False

    def _npu_graph_eligible(self, inputs: tuple[torch.Tensor, ...]) -> bool:
        return (
            bool(inputs)
            and inputs[0].device.type == "npu"
            and self._npu_graph_supported()
            and not self._npu_stream_is_capturing()
        )

    @property
    def npu_graph_stats(self) -> dict[str, int]:
        return {
            "captures": len(self._npu_graphs),
            "failed": len(self._failed_npu_graphs),
            "hits": self._npu_graph_hits,
        }

    def _capture_npu_graph(
        self,
        inputs: tuple[torch.Tensor, ...],
        compute: Callable[..., tuple[torch.Tensor, ...]],
    ) -> _CapturedDeviceGraph:
        npu = torch.npu
        static_inputs = tuple(value.detach().clone() for value in inputs)
        npu.synchronize()
        graph = npu.NPUGraph()
        if self._npu_graph_pool is None:
            self._npu_graph_pool = npu.graph_pool_handle()
        with torch.inference_mode(), npu.graph(graph, pool=self._npu_graph_pool):
            static_outputs = compute(*static_inputs)
        npu.synchronize()
        return _CapturedDeviceGraph(
            graph=graph,
            static_inputs=static_inputs,
            static_outputs=static_outputs,
        )

    def _run_graphable(
        self,
        operation: str,
        inputs: tuple[torch.Tensor, ...],
        constants: tuple[Any, ...],
        compute: Callable[..., tuple[torch.Tensor, ...]],
    ) -> tuple[torch.Tensor, ...]:
        graph_enabled = self._enable_npu_graphs and self._max_npu_graphs > 0 and self._npu_graph_eligible(inputs)
        if not graph_enabled:
            return compute(*inputs)

        key = (
            operation,
            constants,
            tuple(tensor_signature(value) for value in inputs),
        )
        graph = self._npu_graphs.get(key)
        if graph is not None:
            self._npu_graph_hits += 1
            if self._npu_graph_hits == 1:
                logger.info("MiniCPM-o Code2Wav started NPU graph replay")
            return graph.replay(inputs)
        if key in self._failed_npu_graphs:
            return compute(*inputs)

        # The first eager execution initializes lazy kernels and allocator
        # state. Capture after it, then use replay from the next matching call.
        eager_outputs = compute(*inputs)
        if len(self._npu_graphs) >= self._max_npu_graphs:
            logger.warning_once(
                "MiniCPM-o Code2Wav reached the %d-entry NPU graph limit; new tensor shapes will use eager execution.",
                self._max_npu_graphs,
            )
            return eager_outputs
        try:
            self._npu_graphs[key] = self._capture_npu_graph(inputs, compute)
        except Exception:
            self._failed_npu_graphs.add(key)
            logger.warning(
                "MiniCPM-o Code2Wav failed to capture NPU graph for %s; this tensor shape will stay eager.",
                operation,
                exc_info=True,
            )
        else:
            logger.info(
                "MiniCPM-o Code2Wav captured NPU graph %d/%d for %s",
                len(self._npu_graphs),
                self._max_npu_graphs,
                operation,
            )
        return eager_outputs

    def prepare_prompt(self, prompt_cache_id: str, prompt_wav: str) -> PromptFeatures:
        cache_key = (prompt_cache_id, prompt_wav)
        cached = self._prompt_features.get(cache_key)
        if cached is None:
            # The generation runner may wrap model.forward in bf16 autocast,
            # and vLLM constructs the model under a bf16 default dtype, while
            # S3Tokenizer prompt extraction uses fp32 convolution weights.
            previous_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.float32)
                with _autocast_disabled(self.speech_window.device):
                    values = self._token2wav._prepare_prompt(prompt_wav)
            finally:
                torch.set_default_dtype(previous_dtype)
            cached = PromptFeatures(
                speech_tokens=values[0],
                speaker_embedding=values[2],
                mels=values[3],
            )
            self._prompt_features[cache_key] = cached
        return cached

    def evict_prompt(self, prompt_cache_id: str, prompt_wav: str) -> None:
        """Release request-owned prompt features after stream completion."""
        self._prompt_features.pop((prompt_cache_id, prompt_wav), None)

    @staticmethod
    def _repeat_prompt(features: PromptFeatures, batch_size: int) -> tuple[torch.Tensor, ...]:
        return (
            features.speech_tokens.expand(batch_size, -1),
            features.speaker_embedding.expand(batch_size, -1),
            features.mels.expand(batch_size, -1, -1),
        )

    def _autocast(self, device: torch.device):
        if device.type != "cuda":
            return nullcontext()
        if not self.float16:
            return torch.amp.autocast("cuda", enabled=False)
        return torch.amp.autocast(
            "cuda",
            dtype=torch.float16,
        )

    def _pre_lookahead_len(self) -> int | None:
        """Right-context width of the encoder's pre-lookahead convolution.

        ``None`` when the encoder does not expose one, so callers keep working
        against encoder implementations without that layer.
        """
        layer = getattr(self.flow.encoder, "pre_lookahead_layer", None)
        width = getattr(layer, "pre_lookahead_len", None)
        return int(width) if width is not None else None

    def _encode_chunk(
        self,
        tokens: torch.Tensor,
        *,
        last_chunk: bool,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.flow.input_embedding(tokens)
        hidden, new_cnn, new_att = self.flow.encoder.forward_chunk(
            xs=embedded,
            last_chunk=last_chunk,
            cnn_cache=cnn_cache,
            att_cache=att_cache,
        )
        return self.flow.encoder_proj(hidden), new_cnn, new_att

    @staticmethod
    def _estimator_buffers(
        estimator: nn.Module,
        x: torch.Tensor,
        old_att: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        blocks = estimator.blocks
        depth = len(blocks)
        batch_size = int(x.shape[0])
        chunk_size = int(x.shape[2])
        old_att_len = int(old_att.shape[3]) if old_att is not None else 0
        block0 = blocks[0]
        cnn_channels = int(block0.conv.in_channels + block0.conv.out_channels)
        cnn_width = int(block0.conv.block[1].causal_padding[0])
        heads = int(block0.attn.num_heads)
        att_width = int(block0.attn.head_dim * 2)
        cnn = x.new_empty((depth, batch_size, cnn_channels, cnn_width))
        att = x.new_empty((depth, batch_size, heads, old_att_len + chunk_size, att_width))
        return cnn, att

    def _estimator_step(
        self,
        estimator: nn.Module,
        *,
        x: torch.Tensor,
        mu: torch.Tensor,
        time: torch.Tensor,
        speakers: torch.Tensor,
        cond: torch.Tensor,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        time_embedding = estimator.t_embedder(time).unsqueeze(1)
        width = int(x.shape[-1])
        speaker_features = speakers.unsqueeze(-1).expand(-1, -1, width)
        estimator_input = torch.cat((x, mu, speaker_features, cond), dim=1)
        cnn_out, att_out = self._estimator_buffers(estimator, estimator_input, att_cache)
        old_cnn: Any = cnn_cache if cnn_cache is not None else [None] * len(estimator.blocks)
        old_att: Any = att_cache if att_cache is not None else [None] * len(estimator.blocks)
        result = estimator.blocks_forward_chunk(
            estimator_input,
            time_embedding,
            None,
            old_cnn,
            old_att,
            cnn_out,
            att_out,
        )
        return result, cnn_out, att_out

    def _decode_cfm(
        self,
        mu: torch.Tensor,
        speakers: torch.Tensor,
        cond: torch.Tensor,
        *,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decoder = self.flow.decoder
        estimator = decoder.estimator
        batch_size = int(mu.shape[0])
        offset = int(att_cache.shape[4]) if att_cache is not None else 0
        end = offset + int(mu.shape[2])
        if end > int(decoder.rand_noise.shape[2]):
            raise RuntimeError(
                "MiniCPMO45Code2WavBatchError "
                f'{{"reason":"noise_capacity","required":{end},'
                f'"available":{int(decoder.rand_noise.shape[2])}}}'
            )
        x = decoder.rand_noise[:, :, offset:end].expand(batch_size, -1, -1).clone()
        timeline = torch.linspace(
            0,
            1,
            self.n_timesteps + 1,
            device=mu.device,
            dtype=mu.dtype,
        )
        timeline = 1 - torch.cos(timeline * 0.5 * torch.pi)
        time = timeline[0].expand(batch_size)
        mu_cfg = torch.cat((mu, torch.zeros_like(mu)), dim=0)
        speakers_cfg = torch.cat((speakers, torch.zeros_like(speakers)), dim=0)
        cond_cfg = torch.cat((cond, torch.zeros_like(cond)), dim=0)
        next_cnn: list[torch.Tensor] = []
        next_att: list[torch.Tensor] = []
        dt = timeline[1] - timeline[0]
        for step in range(self.n_timesteps):
            old_cnn = cnn_cache[step] if cnn_cache is not None else None
            old_att = att_cache[step] if att_cache is not None else None
            estimate, step_cnn, step_att = self._estimator_step(
                estimator,
                x=torch.cat((x, x), dim=0),
                mu=mu_cfg,
                time=torch.cat((time, time), dim=0),
                speakers=speakers_cfg,
                cond=cond_cfg,
                cnn_cache=old_cnn,
                att_cache=old_att,
            )
            conditional, unconditional = estimate.split(batch_size, dim=0)
            velocity = (1.0 + decoder.inference_cfg_rate) * conditional - decoder.inference_cfg_rate * unconditional
            x = x + dt * velocity
            time = time + dt
            if step + 1 < self.n_timesteps:
                dt = timeline[step + 2] - time[0]
            next_cnn.append(step_cnn)
            next_att.append(step_att)
        return x, torch.stack(next_cnn), torch.stack(next_att)

    @staticmethod
    def _split_flow_cache(cache: dict[str, torch.Tensor], batch_size: int) -> list[dict[str, torch.Tensor]]:
        result: list[dict[str, torch.Tensor]] = []
        for row in range(batch_size):
            result.append(
                {
                    "conformer_cnn_cache": cache["conformer_cnn_cache"][row : row + 1].detach().clone(),
                    "conformer_att_cache": cache["conformer_att_cache"][:, row : row + 1].detach().clone(),
                    "estimator_cnn_cache": torch.cat(
                        (
                            cache["estimator_cnn_cache"][:, :, row : row + 1],
                            cache["estimator_cnn_cache"][:, :, batch_size + row : batch_size + row + 1],
                        ),
                        dim=2,
                    ).detach(),
                    "estimator_att_cache": torch.cat(
                        (
                            cache["estimator_att_cache"][:, :, row : row + 1],
                            cache["estimator_att_cache"][:, :, batch_size + row : batch_size + row + 1],
                        ),
                        dim=2,
                    ).detach(),
                }
            )
        return result

    @staticmethod
    def _stack_flow_cache(states: list[BatchedToken2WavState]) -> dict[str, torch.Tensor]:
        flows = [state.flow_cache for state in states]
        conditional_cnn = [flow["estimator_cnn_cache"][:, :, 0:1] for flow in flows]
        unconditional_cnn = [flow["estimator_cnn_cache"][:, :, 1:2] for flow in flows]
        conditional_att = [flow["estimator_att_cache"][:, :, 0:1] for flow in flows]
        unconditional_att = [flow["estimator_att_cache"][:, :, 1:2] for flow in flows]
        return {
            "conformer_cnn_cache": torch.cat([flow["conformer_cnn_cache"] for flow in flows], dim=0),
            "conformer_att_cache": torch.cat([flow["conformer_att_cache"] for flow in flows], dim=1),
            "estimator_cnn_cache": torch.cat((*conditional_cnn, *unconditional_cnn), dim=2),
            "estimator_att_cache": torch.cat((*conditional_att, *unconditional_att), dim=2),
        }

    def _setup_tensor_batch(
        self,
        prompt_tokens: torch.Tensor,
        speakers: torch.Tensor,
        prompt_mels: torch.Tensor,
        *,
        lookahead_width: int | None,
    ) -> tuple[torch.Tensor, ...]:
        batch_size = int(prompt_tokens.shape[0])
        lookahead = prompt_tokens.new_full(
            (batch_size, 3 if lookahead_width is None else lookahead_width),
            _SILENCE_TOKEN,
        )
        with self._autocast(prompt_tokens.device):
            hidden, conformer_cnn, conformer_att = self._encode_chunk(
                torch.cat((prompt_tokens, lookahead), dim=1),
                last_chunk=False,
                cnn_cache=None,
                att_cache=None,
            )
            projected_speakers = self.flow.spk_embed_affine_layer(F.normalize(speakers, dim=1))
            _, estimator_cnn, estimator_att = self._decode_cfm(
                hidden.transpose(1, 2).contiguous(),
                projected_speakers,
                prompt_mels.transpose(1, 2).contiguous(),
                cnn_cache=None,
                att_cache=None,
            )
        return conformer_cnn, conformer_att, estimator_cnn, estimator_att

    def setup_batch(
        self,
        features: PromptFeatures,
        batch_size: int,
    ) -> list[BatchedToken2WavState]:
        prompt_tensors = self._repeat_prompt(features, batch_size)
        lookahead_width = self._pre_lookahead_len()
        conformer_cnn, conformer_att, estimator_cnn, estimator_att = self._run_graphable(
            "setup",
            prompt_tensors,
            (lookahead_width,),
            lambda *values: self._setup_tensor_batch(
                *values,
                lookahead_width=lookahead_width,
            ),
        )
        flow_cache = {
            "conformer_cnn_cache": conformer_cnn,
            "conformer_att_cache": conformer_att,
            "estimator_cnn_cache": estimator_cnn,
            "estimator_att_cache": estimator_att,
        }
        split = self._split_flow_cache(flow_cache, batch_size)
        prompt_mels = features.mels
        mel_channels = int(prompt_mels.shape[2])
        return [
            BatchedToken2WavState(
                flow_cache=row,
                hift_cache={
                    "mel": prompt_mels.new_zeros((1, mel_channels, 0)),
                    "source": prompt_mels.new_zeros((1, 1, 0)),
                    "speech": prompt_mels.new_zeros((1, 0)),
                },
            )
            for row in split
        ]

    @staticmethod
    def _fade_in_out(
        speech: torch.Tensor,
        previous: torch.Tensor,
        window: torch.Tensor,
    ) -> torch.Tensor:
        overlap = min(
            int(window.shape[0] // 2),
            int(speech.shape[-1]),
            int(previous.shape[-1]),
        )
        result = speech.clone()
        if overlap > 0:
            result[..., :overlap] = (
                result[..., :overlap] * window[:overlap] + previous[..., -overlap:] * window[-overlap:]
            )
        return result

    def _decode_tensor_batch(
        self,
        tokens: torch.Tensor,
        speakers: torch.Tensor,
        conformer_cnn_cache: torch.Tensor,
        conformer_att_cache: torch.Tensor,
        estimator_cnn_cache: torch.Tensor,
        estimator_att_cache: torch.Tensor,
        old_mel: torch.Tensor,
        old_source: torch.Tensor,
        old_speech: torch.Tensor,
        *,
        last_chunk: bool,
        flush_encoder: bool,
        prompt_len: int,
    ) -> tuple[torch.Tensor, ...]:
        with self._autocast(tokens.device):
            hidden, conformer_cnn, conformer_att = self._encode_chunk(
                tokens,
                last_chunk=last_chunk or flush_encoder,
                cnn_cache=conformer_cnn_cache,
                att_cache=conformer_att_cache,
            )
            projected_speakers = self.flow.spk_embed_affine_layer(F.normalize(speakers, dim=1))
            cond = torch.zeros_like(hidden).transpose(1, 2).contiguous()
            chunk_mel, estimator_cnn, estimator_att = self._decode_cfm(
                hidden.transpose(1, 2).contiguous(),
                projected_speakers,
                cond,
                cnn_cache=estimator_cnn_cache,
                att_cache=estimator_att_cache,
            )

        if estimator_att.shape[4] > prompt_len + 100:
            estimator_att = torch.cat(
                (estimator_att[..., :prompt_len, :], estimator_att[..., -100:, :]),
                dim=4,
            )
        if conformer_att.shape[3] > prompt_len + 100:
            conformer_att = torch.cat(
                (conformer_att[..., :prompt_len, :], conformer_att[..., -100:, :]),
                dim=3,
            )
        mel = torch.cat((old_mel, chunk_mel), dim=2)
        speech, source = self.hift(mel, old_source)
        if old_speech.shape[-1] > 0:
            window = self.speech_window.to(device=speech.device, dtype=speech.dtype)
            speech = self._fade_in_out(speech, old_speech, window)
        next_hift = {
            "mel": mel[..., -self.mel_cache_len :].detach(),
            "source": source[..., -self.source_cache_len :].detach(),
            "speech": speech[..., -self.source_cache_len :].detach(),
        }
        emitted = speech if last_chunk else speech[..., : -self.source_cache_len]
        return (
            emitted,
            conformer_cnn,
            conformer_att,
            estimator_cnn,
            estimator_att,
            next_hift["mel"],
            next_hift["source"],
            next_hift["speech"],
        )

    def decode_batch(
        self,
        tokens: torch.Tensor,
        features: PromptFeatures,
        states: list[BatchedToken2WavState],
        *,
        last_chunk: bool,
        flush_encoder: bool = False,
    ) -> tuple[list[torch.Tensor], list[BatchedToken2WavState]]:
        batch_size = int(tokens.shape[0])
        if batch_size != len(states):
            raise ValueError(f"tokens batch {batch_size} != state batch {len(states)}")
        # The encoder's pre-lookahead convolution consumes ``pre_lookahead_len``
        # frames of right context and keeps no left cache, so a non-final chunk
        # must carry at least one full kernel. Only the final chunk is allowed
        # to be shorter: ``forward_chunk`` zero-pads it by the lookahead width.
        lookahead = self._pre_lookahead_len()
        if lookahead is not None and not last_chunk:
            num_frames = int(tokens.shape[1])
            if num_frames <= lookahead:
                raise RuntimeError(
                    "MiniCPMO45Code2WavBatchError "
                    f'{{"reason":"chunk_below_lookahead_window","frames":{num_frames},'
                    f'"minimum":{lookahead + 1}}}'
                )

        flow_cache = self._stack_flow_cache(states)
        speakers = features.speaker_embedding.expand(batch_size, -1)
        old_mel = torch.cat([state.hift_cache["mel"] for state in states], dim=0)
        old_source = torch.cat([state.hift_cache["source"] for state in states], dim=0)
        old_speech = torch.cat([state.hift_cache["speech"] for state in states], dim=0)
        prompt_len = int(features.mels.shape[1])
        tensor_inputs = (
            tokens,
            speakers,
            flow_cache["conformer_cnn_cache"],
            flow_cache["conformer_att_cache"],
            flow_cache["estimator_cnn_cache"],
            flow_cache["estimator_att_cache"],
            old_mel,
            old_source,
            old_speech,
        )
        (
            emitted,
            conformer_cnn,
            conformer_att,
            estimator_cnn,
            estimator_att,
            next_mel,
            next_source,
            next_speech,
        ) = self._run_graphable(
            "decode",
            tensor_inputs,
            (last_chunk, flush_encoder, prompt_len),
            lambda *values: self._decode_tensor_batch(
                *values,
                last_chunk=last_chunk,
                flush_encoder=flush_encoder,
                prompt_len=prompt_len,
            ),
        )
        new_flow = self._split_flow_cache(
            {
                "conformer_cnn_cache": conformer_cnn,
                "conformer_att_cache": conformer_att,
                "estimator_cnn_cache": estimator_cnn,
                "estimator_att_cache": estimator_att,
            },
            batch_size,
        )
        next_hift = {
            "mel": next_mel,
            "source": next_source,
            "speech": next_speech,
        }
        next_states = [
            BatchedToken2WavState(
                flow_cache=new_flow[row],
                hift_cache={name: value[row : row + 1].detach().clone() for name, value in next_hift.items()},
            )
            for row in range(batch_size)
        ]
        audios = [emitted[row].reshape(-1).to(dtype=torch.float32) for row in range(batch_size)]
        return audios, next_states
