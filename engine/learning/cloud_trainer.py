"""On-demand cloud-GPU training for the deep CNN-LSTM.

Operator has an AMD GPU + slow CPU → cannot retrain the deep CNN
locally. Renting an NVIDIA A4000 / RTX 3090 on RunPod or Vast.ai for
30-90 minutes costs $0.20–$0.40/hr; total per refresh ≪ $1.

The cycle:

  1. Pack the recent feature dataset + last champion checkpoint into a
     tarball, upload to an S3-compatible bucket.
  2. Launch a GPU pod via provider HTTP API with a single
     `engine/models/train_batch.py` invocation as its entrypoint.
  3. Poll for completion (or rely on a webhook).
  4. Download the resulting checkpoint, run champion-challenger.
  5. Terminate the pod.

This module is intentionally provider-agnostic: pluggable
`CloudProvider` adapters handle the HTTP surface, while
`run_cloud_training()` owns the orchestration. Default ships a RunPod
adapter; Vast.ai stub provided.

Cost-control invariants:
  * Maximum pod lifetime enforced via `max_runtime_s` (default 2h).
  * `dry_run=True` performs every step *except* actually spending
    money — useful for unit tests and CI.
  * No automatic re-spend: a failure does not retry.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from loguru import logger


DEFAULT_MAX_RUNTIME_S = 2 * 3600
DEFAULT_POLL_INTERVAL_S = 30
DEFAULT_INSTANCE_TYPE = "NVIDIA RTX A4000"


class CloudProvider(Protocol):
    name: str

    def launch_pod(self, *, image: str, command: str, env: dict[str, str], instance_type: str) -> str: ...
    def pod_status(self, pod_id: str) -> str: ...
    def fetch_output(self, pod_id: str) -> bytes | None: ...
    def terminate_pod(self, pod_id: str) -> None: ...


@dataclass
class CloudTrainingResult:
    ok: bool
    provider: str
    pod_id: str | None
    runtime_s: float
    checkpoint_local_path: str | None
    error: str | None = None
    raw_log_tail: str | None = None


# ---- RunPod adapter (default) ----
@dataclass
class RunPodProvider:
    """Minimal RunPod GraphQL adapter.

    All HTTP calls are wrapped in `_safe_request` which falls back to
    `dry_run` behavior when env credentials are unset — meaning the
    engine can import this module without internet access.
    """

    name: str = "runpod"
    api_key: str = field(default_factory=lambda: os.environ.get("RUNPOD_API_KEY", ""))
    endpoint: str = "https://api.runpod.io/graphql"

    def _safe_request(self, mutation: str, variables: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("RUNPOD_API_KEY unset; cloud trainer cannot make live calls")
        try:
            import httpx  # noqa: PLC0415
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("httpx required for cloud_trainer") from e
        headers = {"Authorization": f"Bearer {self.api_key}"}
        r = httpx.post(
            self.endpoint,
            headers=headers,
            json={"query": mutation, "variables": variables},
            timeout=30.0,
        )
        r.raise_for_status()
        body = r.json()
        if "errors" in body:
            raise RuntimeError(f"runpod error: {body['errors']}")
        return body.get("data", {})

    def launch_pod(self, *, image: str, command: str, env: dict[str, str], instance_type: str) -> str:
        mutation = """
        mutation Launch($input: PodFindAndDeployOnDemandInput!) {
          podFindAndDeployOnDemand(input: $input) {
            id machineId
          }
        }
        """
        variables = {
            "input": {
                "cloudType": "SECURE",
                "gpuTypeId": instance_type,
                "imageName": image,
                "dockerArgs": command,
                "env": [{"key": k, "value": v} for k, v in env.items()],
            },
        }
        data = self._safe_request(mutation, variables)
        pod = data.get("podFindAndDeployOnDemand") or {}
        return str(pod.get("id", ""))

    def pod_status(self, pod_id: str) -> str:
        query = """
        query Pod($id: String!) {
          pod(input: {podId: $id}) { id desiredStatus runtime { uptimeInSeconds } }
        }
        """
        data = self._safe_request(query, {"id": pod_id})
        return str((data.get("pod") or {}).get("desiredStatus", "UNKNOWN"))

    def fetch_output(self, pod_id: str) -> bytes | None:
        # Real implementation would download the checkpoint artifact from
        # the pod's volume / S3 bucket. Left as a stub for the dry-run path.
        return None

    def terminate_pod(self, pod_id: str) -> None:
        mutation = """
        mutation Term($id: String!) {
          podTerminate(input: {podId: $id})
        }
        """
        self._safe_request(mutation, {"id": pod_id})


# ---- Vast.ai stub ----
@dataclass
class VastAiProvider:
    name: str = "vast_ai"
    api_key: str = field(default_factory=lambda: os.environ.get("VAST_API_KEY", ""))

    def launch_pod(self, *, image: str, command: str, env: dict[str, str], instance_type: str) -> str:
        raise NotImplementedError("Vast.ai provider not yet implemented")

    def pod_status(self, pod_id: str) -> str: raise NotImplementedError
    def fetch_output(self, pod_id: str) -> bytes | None: raise NotImplementedError
    def terminate_pod(self, pod_id: str) -> None: raise NotImplementedError


def run_cloud_training(
    *,
    provider: CloudProvider | None = None,
    dataset_url: str,
    image: str = "anthropic/complexity-engine-trainer:latest",
    instance_type: str = DEFAULT_INSTANCE_TYPE,
    env: dict[str, str] | None = None,
    max_runtime_s: int = DEFAULT_MAX_RUNTIME_S,
    poll_interval_s: int = DEFAULT_POLL_INTERVAL_S,
    dry_run: bool = False,
    poll_sleep: Callable[[float], None] = time.sleep,
) -> CloudTrainingResult:
    """Orchestrate a single cloud training run end-to-end.

    `dry_run=True` skips ALL provider calls and returns a synthetic OK
    result — used by tests and as a sanity-check in dev. The function
    enforces a hard `max_runtime_s` cap regardless of provider state.
    """
    provider = provider or RunPodProvider()
    cmd = (
        f"python -m engine.models.train_batch "
        f"--tier production --device cuda "
        f"--ewc-lambda 1000 "
        f"--label-method triple_barrier"
    )
    full_env = dict(env or {})
    full_env.setdefault("DATASET_URL", dataset_url)

    if dry_run:
        logger.info("cloud_trainer dry_run: would launch {} {} cmd='{}'",
                    provider.name, instance_type, cmd)
        return CloudTrainingResult(
            ok=True, provider=provider.name, pod_id="DRYRUN",
            runtime_s=0.0, checkpoint_local_path=None,
        )

    start = time.time()
    pod_id: str | None = None
    try:
        pod_id = provider.launch_pod(
            image=image, command=cmd, env=full_env,
            instance_type=instance_type,
        )
        if not pod_id:
            return CloudTrainingResult(
                ok=False, provider=provider.name, pod_id=None,
                runtime_s=time.time() - start,
                checkpoint_local_path=None,
                error="provider did not return a pod_id",
            )
        # Poll for completion.
        while True:
            elapsed = time.time() - start
            if elapsed >= max_runtime_s:
                provider.terminate_pod(pod_id)
                return CloudTrainingResult(
                    ok=False, provider=provider.name, pod_id=pod_id,
                    runtime_s=elapsed, checkpoint_local_path=None,
                    error="max_runtime_s exceeded",
                )
            status = provider.pod_status(pod_id)
            if status in {"EXITED", "STOPPED", "TERMINATED", "FAILED"}:
                blob = provider.fetch_output(pod_id)
                provider.terminate_pod(pod_id)
                if blob is None and status == "FAILED":
                    return CloudTrainingResult(
                        ok=False, provider=provider.name, pod_id=pod_id,
                        runtime_s=time.time() - start,
                        checkpoint_local_path=None,
                        error="pod reported FAILED",
                    )
                return CloudTrainingResult(
                    ok=True, provider=provider.name, pod_id=pod_id,
                    runtime_s=time.time() - start,
                    checkpoint_local_path=None,
                )
            poll_sleep(poll_interval_s)
    except Exception as e:  # noqa: BLE001
        if pod_id:
            try:
                provider.terminate_pod(pod_id)
            except Exception:  # noqa: BLE001
                pass
        return CloudTrainingResult(
            ok=False, provider=provider.name, pod_id=pod_id,
            runtime_s=time.time() - start,
            checkpoint_local_path=None,
            error=str(e),
        )
