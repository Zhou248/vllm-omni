# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Base NPU worker class for vLLM-Omni with OmniProfiler support."""

import time

from vllm_omni.platforms.npu._310p import is_310p

if is_310p():
    from vllm_ascend._310p.worker_310p import NPUWorker310 as NPUWorker
else:
    from vllm_ascend.worker.worker import NPUWorker


class OmniNPUWorkerBase(NPUWorker):
    """Base NPU worker with lazily initialized OmniProfiler."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # NPU profiler must not be constructed before init_device/model init.
        # Only retain its configuration here; create it on /start_profile.
        self.profiler_config = self.vllm_config.profiler_config
        self.profiler = None

    def profile(
        self,
        is_start: bool = True,
        profile_prefix: str | None = None,
    ):
        """Create/start the profiler lazily after NPU initialization."""

        profiler_config = self.profiler_config
        if (
            profiler_config is None
            or profiler_config.profiler != "torch"
        ):
            raise RuntimeError(
                "Profiling is not enabled. Add profiler_config to the "
                "corresponding stage configuration."
            )

        if is_start:
            from vllm_omni.profiler import (
                OmniTorchProfilerWrapper,
                create_omni_profiler,
            )

            # Lazy construction is important on NPU. At this point the service,
            # device context and model runner have already been initialized.
            if self.profiler is None:
                stage_id = getattr(
                    self.vllm_config.model_config,
                    "stage_id",
                    0,
                )
                worker_name = f"stage{stage_id}_rank{self.rank}"
                self.profiler = create_omni_profiler(
                    profiler_config=profiler_config,
                    worker_name=worker_name,
                    local_rank=self.local_rank,
                )

            if isinstance(self.profiler, OmniTorchProfilerWrapper):
                stage_id = getattr(
                    self.vllm_config.model_config,
                    "stage_id",
                    0,
                )
                filename = (
                    profile_prefix
                    or f"stage{stage_id}_rank{self.rank}_{int(time.time())}"
                )
                self.profiler.set_trace_filename(filename)

            self.profiler.start()
            return

        # Calling stop before start should be harmless.
        if self.profiler is None:
            return

        self.profiler.stop()
