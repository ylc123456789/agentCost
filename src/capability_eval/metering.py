from __future__ import annotations

import threading
from contextlib import contextmanager

from .io import ContractError


@contextmanager
def cpu_window(recorder):
    import resource

    initial = resource.getrusage(resource.RUSAGE_SELF)
    try:
        yield
    finally:
        final = resource.getrusage(resource.RUSAGE_SELF)
        for name, value in (("cpu_user_seconds", final.ru_utime - initial.ru_utime),
                            ("cpu_system_seconds", final.ru_stime - initial.ru_stime)):
            recorder.measure(name, max(0, value), "s", data={"scope": "calling_process_window"})


_cuda_outer = threading.Lock()


@contextmanager
def cuda_window(recorder, device, *, nvml_index=None, exclusive=False, interval=0.05):
    """Optional SDK window, not an automatic GPU lease or kernel-busy meter."""
    import torch

    if not exclusive:
        raise ContractError("allocator reset requires an isolated, exclusive CUDA process")
    if interval <= 0 or not _cuda_outer.acquire(blocking=False):
        raise ContractError("CUDA outer window already active or sampler interval invalid")
    sampler = None
    try:
        torch.cuda.synchronize(device)
        for name, getter in (("torch_allocated_baseline_bytes", torch.cuda.memory_allocated),
                             ("torch_reserved_baseline_bytes", torch.cuda.memory_reserved)):
            recorder.measure(name, getter(device), "B", data={"scope": "torch_allocator"})
        torch.cuda.reset_peak_memory_stats(device)
        if nvml_index is not None:
            sampler = NvmlMemorySampler(nvml_index, interval)
            sampler.start()
        try:
            yield
        finally:
            try:
                torch.cuda.synchronize(device)
                for name, getter in (("torch_allocated_peak_bytes", torch.cuda.max_memory_allocated),
                                     ("torch_reserved_peak_bytes", torch.cuda.max_memory_reserved)):
                    recorder.measure(name, getter(device), "B", data={"scope": "torch_allocator"})
            finally:
                if sampler is not None:
                    sampler.finish(recorder)
                else:
                    recorder.measure("nvml_device_sampled_peak_bytes", None, "B",
                                     reason="NVML sampler not configured")
    finally:
        _cuda_outer.release()


class NvmlMemorySampler:
    def __init__(self, index, interval):
        self.index, self.interval = index, interval
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.samples = []
        self.error = None
        self.uuid = None

    def _sample(self):
        try:
            import pynvml as nvml

            nvml.nvmlInit()
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(self.index)
                self.uuid = nvml.nvmlDeviceGetUUID(handle)
                while True:
                    self.samples.append(nvml.nvmlDeviceGetMemoryInfo(handle).used)
                    if self.stop.wait(self.interval):
                        break
            finally:
                nvml.nvmlShutdown()
        except Exception as exc:
            self.error = type(exc).__name__

    def start(self):
        self.thread.start()

    def finish(self, recorder):
        self.stop.set()
        self.thread.join(timeout=2)
        samples = list(self.samples)
        identity = self.uuid.decode() if isinstance(self.uuid, bytes) else self.uuid
        recorder.measure("nvml_device_sampled_peak_bytes", max(samples) if samples else None, "B",
                         reason=self.error or (None if samples else "no NVML sample available"),
                         data={"scope": "whole_device_not_method_allocator", "interval_seconds": self.interval,
                               "sample_count": len(samples), "partial": bool(self.error or self.thread.is_alive())},
                         device_id=identity or f"nvml:{self.index}")
