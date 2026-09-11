import json

import pytest

from stylegan2.benchmark import benchmark
from stylegan2.models import ModelConfig
from stylegan2.training import TrainConfig


def configs():
    return (
        ModelConfig(
            resolution=8,
            z_dim=8,
            w_dim=8,
            mapping_layers=2,
            channel_base=64,
            channel_max=8,
        ),
        TrainConfig(
            r1_interval=2, pl_interval=2, microbatch=2, regularizer_conv="analytic"
        ),
    )


def test_phase_counts_and_provenance():
    model, train = configs()
    result = benchmark(model, train, 4, "cpu", 2, 4, progress=False)
    assert result["schema"] == 2 and result["steady_state_valid"]
    assert result["measured_steps"] == 4
    assert len(result["source_sha256"]) == 64
    assert result["phase_timings"]["r1"]["calls"] == 2
    assert result["phase_timings"]["pl"]["calls"] == 2
    for name in ("input", "d_main", "g_main", "ema"):
        assert result["phase_timings"][name]["calls"] == 4
    assert result["images_per_second"] == pytest.approx(16 / result["measured_seconds"])
    assert (
        0
        < sum(v["total_ms"] for v in result["phase_timings"].values())
        <= result["measured_seconds"] * 1000
    )
    assert result["comparison_to_lucidrains"] == "not measured"


def test_no_phase_timers_and_bad_cycles():
    model, train = configs()
    result = benchmark(model, train, 4, "cpu", 2, 2, phase_timing=False, progress=False)
    assert result["phase_timings"] == {}
    with pytest.raises(ValueError, match="multiples"):
        benchmark(model, train, 4, "cpu", 1, 2, progress=False)


def test_precision_environment_conflicts_fail(monkeypatch):
    model, train = configs()
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")
    with pytest.raises(ValueError, match="override"):
        benchmark(model, train, 4, "cpu", 2, 2, progress=False)


def test_bounded_trace(tmp_path):
    model, train = configs()
    path = tmp_path / "trace.json"
    result = benchmark(
        model, train, 4, "cpu", 2, 4, trace=path, trace_steps=1, progress=False
    )
    assert result["profiled"] and not result["steady_state_valid"]
    trace = json.loads(path.read_text())
    names = [event.get("name") for event in trace["traceEvents"]]
    assert "sg2/r1" in names and "sg2/pl" in names
    assert names.count("sg2/d_main") == 1
