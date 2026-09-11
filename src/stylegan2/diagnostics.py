"""Optional asynchronous phase timers and host heartbeat; no per-step GPU sync."""

import sys
import threading
import time

import numpy as np
import torch


class Progress:
    def __init__(self, enabled=True, interval=15, label="benchmark"):
        self.enabled, self.interval = enabled, interval
        self.label = label
        self.state = "initialising"
        self.started = time.perf_counter()
        self.stop = threading.Event()
        self.thread = None

    def __enter__(self):
        if self.enabled:
            self.report()
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
        return self

    def report(self):
        print(
            f"[{self.label} {time.perf_counter()-self.started:.0f}s] {self.state}",
            file=sys.stderr,
            flush=True,
        )

    def run(self):
        while not self.stop.wait(self.interval):
            self.report()

    def __exit__(self, *exc):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1)


class PhaseTimer:
    def __init__(self, device, enabled=True, annotate=False, on_phase=None):
        self.device, self.enabled, self.annotate = device, enabled, annotate
        self.on_phase = on_phase
        self.records = []
        self.current = None
        self.region = None

    def stamp(self):
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record(torch.cuda.current_stream(self.device))
            return event
        return time.perf_counter()

    def __call__(self, name):
        stamp = self.stamp() if self.enabled else None
        if self.current is not None:
            if self.enabled:
                self.records.append((self.current[0], self.current[1], stamp))
            if self.region is not None:
                self.region.__exit__(None, None, None)
                self.region = None
        self.current = (name, stamp) if name is not None else None
        if name is not None:
            if self.on_phase is not None:
                self.on_phase(name)
            if self.annotate:
                self.region = torch.profiler.record_function("sg2/" + name)
                self.region.__enter__()

    def summary(self):
        # Caller synchronises once after the entire measured window.
        groups = {}
        for name, start, end in self.records:
            ms = (
                start.elapsed_time(end)
                if self.device.type == "cuda"
                else (end - start) * 1000
            )
            groups.setdefault(name, []).append(ms)
        return {
            name: {
                "calls": len(values),
                "total_ms": float(sum(values)),
                "median_ms": float(np.median(values)),
                "mean_ms": float(np.mean(values)),
            }
            for name, values in groups.items()
        }
