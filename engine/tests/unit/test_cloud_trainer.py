"""Tests for cloud-GPU trainer (Tier 5.3)."""
from __future__ import annotations

from dataclasses import dataclass, field

from engine.learning.cloud_trainer import (
    CloudTrainingResult,
    RunPodProvider,
    VastAiProvider,
    run_cloud_training,
)


@dataclass
class _FakeProvider:
    name: str = "fake"
    status_sequence: list[str] = field(default_factory=lambda: ["RUNNING", "RUNNING", "EXITED"])
    launched: bool = False
    terminated: bool = False
    _i: int = 0
    _output: bytes = b"checkpoint"

    def launch_pod(self, *, image, command, env, instance_type):
        self.launched = True
        return "pod-123"

    def pod_status(self, pod_id):
        if self._i >= len(self.status_sequence):
            return self.status_sequence[-1]
        s = self.status_sequence[self._i]
        self._i += 1
        return s

    def fetch_output(self, pod_id):
        return self._output

    def terminate_pod(self, pod_id):
        self.terminated = True


def test_dry_run_skips_provider():
    r = run_cloud_training(
        provider=_FakeProvider(),
        dataset_url="s3://x/y.tar.gz",
        dry_run=True,
    )
    assert r.ok
    assert r.pod_id == "DRYRUN"


def test_successful_run_terminates_pod():
    p = _FakeProvider()
    r = run_cloud_training(
        provider=p,
        dataset_url="s3://x/y.tar.gz",
        poll_sleep=lambda s: None,
    )
    assert r.ok
    assert p.launched and p.terminated


def test_failed_pod_returns_error():
    p = _FakeProvider(status_sequence=["RUNNING", "FAILED"])
    p._output = None  # type: ignore[assignment]
    r = run_cloud_training(
        provider=p,
        dataset_url="s3://x/y.tar.gz",
        poll_sleep=lambda s: None,
    )
    assert not r.ok
    assert r.error and "FAILED" in r.error
    assert p.terminated


def test_max_runtime_enforced():
    @dataclass
    class _Stuck:
        name: str = "stuck"
        launched: bool = False
        terminated: bool = False
        def launch_pod(self, **kw): self.launched = True; return "p"
        def pod_status(self, p): return "RUNNING"
        def fetch_output(self, p): return None
        def terminate_pod(self, p): self.terminated = True

    p = _Stuck()
    r = run_cloud_training(
        provider=p, dataset_url="s3://x/y.tar.gz",
        max_runtime_s=0, poll_sleep=lambda s: None,
    )
    assert not r.ok
    assert "max_runtime" in (r.error or "")
    assert p.terminated


def test_provider_returns_empty_pod_id():
    @dataclass
    class _NoId:
        name: str = "no_id"
        def launch_pod(self, **kw): return ""
        def pod_status(self, p): return "EXITED"
        def fetch_output(self, p): return None
        def terminate_pod(self, p): return

    r = run_cloud_training(
        provider=_NoId(), dataset_url="s3://x/y.tar.gz",
        poll_sleep=lambda s: None,
    )
    assert not r.ok


def test_runpod_without_api_key_raises_on_live_call():
    p = RunPodProvider(api_key="")
    import pytest as _pytest
    with _pytest.raises(RuntimeError):
        p._safe_request("query {}", {})


def test_vast_ai_stub_raises():
    p = VastAiProvider()
    import pytest as _pytest
    with _pytest.raises(NotImplementedError):
        p.launch_pod(image="x", command="y", env={}, instance_type="z")
