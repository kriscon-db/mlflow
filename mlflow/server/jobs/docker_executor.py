"""[POC] Docker implementation of ``AbstractJobExecutor``.

Reuses the hardened container mechanics validated for the scorer sandbox, but implements
the real RFC #2 job-executor contract so it drops into the framework once that lands.

POC deviations from the RFC #3 Docker spec, called out where they occur:
  - Result transport is a bind-mounted ``result.json`` file rather than an HTTP callback to
    the job store, because the job-store/token machinery lives on the unmerged
    ``jobs-execution-rfc2`` branch.
  - The container runs with ``--network none`` since we pre-inject inputs and read a file.
    The RFC callback model instead allows egress to ``tracking_uri`` + ``gateway_uri`` under
    an operator-provided egress policy.
"""

import json
import logging
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities._job_status import JobStatus
from mlflow.environment_variables import MLFLOW_SCORER_SANDBOX_DOCKER_IMAGE
from mlflow.exceptions import MlflowException
from mlflow.server.jobs.executor import (
    AbstractJobExecutor,
    JobExecutionContext,
    JobRecoveryResult,
    JobResult,
)
from mlflow.utils.environment import _PythonEnv

_logger = logging.getLogger(__name__)

_ENTRY_MODULE = "mlflow.server.jobs._docker_job_entry"
_LABEL_JOB_ID = "mlflow.job_id"
_LABEL_JOB_NAME = "mlflow.job_name"
_MEMORY = "1g"
_NANO_CPUS = 1_000_000_000
_PIDS_LIMIT = 256


def _image() -> str:
    """Sandbox image name (read inside a function so tests/consumers can override the env var)."""
    return MLFLOW_SCORER_SANDBOX_DOCKER_IMAGE.get()


@dataclass
class _RunningJob:
    container_id: str
    workdir: Path
    timeout: float | None = None


class DockerJobExecutor(AbstractJobExecutor):
    """Runs each job in a locked-down Docker container via docker-py."""

    def __init__(self, config, result_callback_url: str | None = None) -> None:
        super().__init__(config)
        self._client = None
        self._jobs: dict[str, _RunningJob] = {}
        # POC: when set, the container reports its JobResult over HTTP to this URL (RFC
        # callback model) and runs with host egress instead of --network none. When None,
        # the container is network-isolated and results come back via the bind-mounted file.
        self._result_callback_url = result_callback_url

    @property
    def remote_execution(self) -> bool:
        # Isolation backend: opts into the framework's scoped-token + Gateway-only model.
        return True

    def _get_client(self):
        import docker

        if self._client is None:
            self._client = docker.from_env()
        return self._client

    def check_requirements(self) -> None:
        try:
            self._get_client().ping()
        except Exception as e:
            raise MlflowException(f"Docker daemon is not reachable: {e}")

    def start_executor(self) -> None:
        self.check_requirements()
        self._ensure_image()

    def stop_executor(self) -> None:
        for job_id in list(self._jobs):
            self._cleanup(job_id, kill=True)

    def _ensure_image(self) -> None:
        import docker.errors

        client = self._get_client()
        try:
            client.images.get(_image())
            return
        except docker.errors.ImageNotFound:
            pass
        buildargs = {
            k: os.environ[k] for k in ("PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL") if os.environ.get(k)
        }
        dockerfile = (
            "FROM python:3.11-slim\n"
            "ARG PIP_INDEX_URL=\n"
            "ARG PIP_EXTRA_INDEX_URL=\n"
            "ENV PIP_INDEX_URL=$PIP_INDEX_URL PIP_EXTRA_INDEX_URL=$PIP_EXTRA_INDEX_URL\n"
            "RUN pip install --no-cache-dir mlflow\n"
        )
        with tempfile.TemporaryDirectory(prefix="mlflow-sandbox-image-") as ctx:
            Path(ctx, "Dockerfile").write_text(dockerfile)
            client.images.build(path=ctx, tag=_image(), buildargs=buildargs, rm=True)

    def _job_env(self, context: JobExecutionContext) -> dict[str, str]:
        env = {
            "PYTHONPATH": "/mlflow-src",
            "HOME": "/sandbox",
            "MLFLOW_JOB_TRACKING_URI": context.tracking_uri,
        }
        if context.gateway_uri:
            env["MLFLOW_JOB_GATEWAY_URI"] = context.gateway_uri
        if context.token:
            env["MLFLOW_JOB_TOKEN"] = context.token
        if context.workspace:
            env["MLFLOW_WORKSPACE"] = context.workspace
        if self._result_callback_url:
            base = self._result_callback_url.rstrip("/")
            env["MLFLOW_JOB_RESULT_CALLBACK_URL"] = f"{base}/{context.job_id}"
        return env

    def submit_job(
        self,
        job_id: str,
        job_name: str,
        fn_fullname: str,
        params: dict[str, Any],
        context: JobExecutionContext,
        python_env: _PythonEnv | None = None,
        timeout: float | None = None,
    ) -> None:
        client = self._get_client()
        self._ensure_image()
        workdir = Path(tempfile.mkdtemp(prefix="mlflow-docker-job-"))
        (workdir / "job.json").write_text(
            json.dumps({"fn_fullname": fn_fullname, "params": params})
        )
        mlflow_pkg = Path(mlflow.__file__).resolve().parent
        # File transport -> full network isolation. Callback transport -> host egress so the
        # container can POST its result back (operator egress policy would allowlist this).
        callback = self._result_callback_url is not None
        container = client.containers.run(
            _image(),
            command=["python", "-m", _ENTRY_MODULE, "/sandbox"],
            detach=True,
            labels={_LABEL_JOB_ID: job_id, _LABEL_JOB_NAME: job_name},
            network_mode="bridge" if callback else "none",
            extra_hosts={"host.docker.internal": "host-gateway"} if callback else None,
            mem_limit=_MEMORY,
            memswap_limit=_MEMORY,
            nano_cpus=_NANO_CPUS,
            pids_limit=_PIDS_LIMIT,
            read_only=True,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            tmpfs={"/tmp": ""},
            user=f"{os.getuid()}:{os.getgid()}",
            working_dir="/sandbox",
            environment=self._job_env(context),
            volumes={
                str(workdir): {"bind": "/sandbox", "mode": "rw"},
                str(mlflow_pkg): {"bind": "/mlflow-src/mlflow", "mode": "ro"},
            },
        )
        self._jobs[job_id] = _RunningJob(
            container_id=container.id, workdir=workdir, timeout=timeout
        )
        _logger.info(
            "docker job %s submitted (%s, container %s)", job_id, fn_fullname, container.short_id
        )

    def wait_for_job(self, job_id: str) -> JobResult:
        rec = self._jobs.get(job_id)
        if rec is None:
            return JobResult(status=JobStatus.FAILED, error_message=f"Unknown job {job_id}")
        container = self._get_client().containers.get(rec.container_id)
        # Always clean up (remove container + workdir, drop the record) even if the result
        # file is missing/corrupt or wait raises — otherwise a partial result.json (OOM /
        # disk-full mid-write) would leak the container and temp dir.
        try:
            try:
                # Honor the per-job timeout passed to submit_job; fall back to the framework
                # default only when the job did not specify one.
                wait_timeout = (
                    rec.timeout if rec.timeout is not None else self._config.default_timeout
                )
                outcome = container.wait(timeout=wait_timeout)
            except Exception as e:
                return JobResult(
                    status=JobStatus.TIMEOUT, error_message=str(e), is_transient_error=True
                )
            result_path = rec.workdir / "result.json"
            data = None
            if result_path.exists():
                try:
                    data = json.loads(result_path.read_text())
                except (ValueError, OSError):
                    data = None
            if data is not None:
                return JobResult(
                    status=JobStatus(data["status"]),
                    result=data.get("result"),
                    error_message=data.get("error_message"),
                    is_transient_error=data.get("is_transient_error", False),
                )
            logs = container.logs().decode(errors="replace")[-2000:]
            code = outcome.get("StatusCode")
            return JobResult(
                status=JobStatus.FAILED,
                error_message=f"Sandboxed job produced no result (exit={code}).\n{logs}",
            )
        finally:
            self._cleanup(job_id)

    def cancel_job(self, job_id: str) -> None:
        self._cleanup(job_id, kill=True)

    def recover_jobs(self, unfinished_job_ids: list[str]) -> list[JobRecoveryResult]:
        client = self._get_client()
        results = []
        for job_id in unfinished_job_ids:
            containers = client.containers.list(
                all=True, filters={"label": f"{_LABEL_JOB_ID}={job_id}"}
            )
            if not containers:
                results.append(JobRecoveryResult(job_id=job_id, action="requeue"))
                continue
            container = containers[0]
            if container.status == "running":
                results.append(JobRecoveryResult(job_id=job_id, action="reattach"))
            else:
                results.append(
                    JobRecoveryResult(
                        job_id=job_id,
                        action="fail",
                        error_message=(
                            f"container {container.short_id} exited without a reported result"
                        ),
                    )
                )
        return results

    def _cleanup(self, job_id: str, kill: bool = False) -> None:
        rec = self._jobs.pop(job_id, None)
        if rec is None:
            return
        try:
            container = self._get_client().containers.get(rec.container_id)
            if kill:
                container.kill()
            container.remove(force=True)
        except Exception:
            pass
        shutil.rmtree(rec.workdir, ignore_errors=True)
