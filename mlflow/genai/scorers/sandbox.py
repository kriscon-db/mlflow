"""[POC] Run a ``@scorer`` code scorer's source in an isolated sandbox.

One entry point, ``run_scorer_in_sandbox``, backed by a provider selected via
``MLFLOW_SCORER_SANDBOX_PROVIDER``:

- ``subprocess`` (default): a scrubbed-env child process with a CPU limit and a wall-clock
  timeout. Portable, no infra, but does NOT confine filesystem reads or network egress.
- ``docker``: runs the scorer AS a job on the RFC #2 job-executor substrate
  (``DockerJobExecutor``, obtained from the vendored executor registry). The scorer source
  executes inside a locked-down container (``--network none``, resource limits, non-root,
  read-only rootfs) — the container IS the sandbox. Requires a running docker daemon.

Either way the scorer's source is NEVER ``exec()``-d in the server/worker process — the
source + inputs are shipped to ``_sandbox_runner``, which runs inside the sandbox.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

from mlflow.entities import AssessmentError, Feedback
from mlflow.environment_variables import MLFLOW_SCORER_SANDBOX_PROVIDER
from mlflow.exceptions import MlflowException
from mlflow.utils.os import is_windows

_logger = logging.getLogger(__name__)

# Only these env vars reach the subprocess sandbox; everything else (including any
# credential-bearing var) is dropped. HOME/PYTHONPATH are set explicitly in _scrubbed_env.
_ENV_ALLOWLIST = frozenset({
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "LC_NUMERIC",
    "TZ",
    "TERM",
    # Windows essentials for launching the Python interpreter.
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "COMSPEC",
    "PATHEXT",
    "NUMBER_OF_PROCESSORS",
    "TEMP",
    "TMP",
})

_CPU_SECONDS = 30
_WALL_TIMEOUT_SECONDS = 120
_RUNNER_MODULE = "mlflow.genai.scorers._sandbox_runner"


# --------------------------------------------------------------------------------------
# Payload / result plumbing (backend-agnostic)
# --------------------------------------------------------------------------------------
def _serialize_trace(trace: Any) -> Any:
    if trace is None:
        return None
    errors = []
    for attr in ("to_json", "to_dict"):
        fn = getattr(trace, attr, None)
        if callable(fn):
            try:
                return fn()
            except Exception as e:
                errors.append(f"{attr}: {e}")
    # Do NOT silently degrade to None: a trace-dependent scorer would then run with
    # trace=None and emit a plausible-but-wrong result. Surface the failure instead.
    detail = "; ".join(errors) or "no to_json/to_dict method"
    raise MlflowException(f"Failed to serialize trace for the sandbox ({detail}).")


def _json_safe_kwargs(call_kwargs: dict[str, Any]) -> dict[str, Any]:
    safe = dict(call_kwargs)
    if safe.get("trace") is not None:
        safe["trace"] = _serialize_trace(safe["trace"])
    if safe.get("session") is not None:
        safe["session"] = [_serialize_trace(t) for t in safe["session"]]
    return {k: v for k, v in safe.items() if v is not None}


def _feedback_from_dict(v: dict[str, Any]) -> Feedback:
    error = v.get("error")
    return Feedback(
        name=v.get("name"),
        value=v.get("value"),
        rationale=v.get("rationale"),
        error=AssessmentError(**error) if error else None,
    )


def _deserialize_result(result: dict[str, Any]) -> Any:
    kind = result["kind"]
    if kind == "feedback":
        return _feedback_from_dict(result["value"])
    if kind == "feedback_list":
        return [_feedback_from_dict(v) for v in result["value"]]
    return result["value"]


# --------------------------------------------------------------------------------------
# Provider: subprocess (default, portable, no infra)
# --------------------------------------------------------------------------------------
def _scrubbed_env(workdir: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k in _ENV_ALLOWLIST}
    env["HOME"] = str(workdir)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    return env


def _apply_cpu_limit() -> None:
    # CPU-time cap only. RLIMIT_AS is skipped (caps virtual address space, breaks
    # numpy/pandas imports); real memory limiting belongs to the container tier.
    import resource  # clint: disable=lazy-import  (unix-only; unavailable on Windows)

    try:
        resource.setrlimit(resource.RLIMIT_CPU, (_CPU_SECONDS, _CPU_SECONDS))
    except (ValueError, OSError):
        pass


def _run_subprocess(workdir: Path, func_name: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-m", _RUNNER_MODULE, str(workdir)],
        cwd=workdir,
        env=_scrubbed_env(workdir),
        preexec_fn=None if is_windows() else _apply_cpu_limit,
        capture_output=True,
        text=True,
        timeout=_WALL_TIMEOUT_SECONDS,
    )
    _require_result(workdir, func_name, proc.returncode, proc.stderr)


# Provider: docker. Runs the scorer as a job on DockerJobExecutor (from the vendored RFC #2
# registry, so a Kubernetes backend can slot in later) — the hardened container is the sandbox.
_SCORER_JOB_FN = "mlflow.genai.scorers._sandbox_runner.run_scorer_job"

_job_executor = None
_job_executor_lock = threading.Lock()


def _get_scorer_job_executor():
    """Return a ``DockerJobExecutor`` registered under the "docker" backend of a
    ``JobExecutorRegistry`` (the RFC #2 selection seam).
    """
    global _job_executor
    from mlflow.server.jobs.docker_executor import DockerJobExecutor
    from mlflow.server.jobs.executor import JobExecutorConfig
    from mlflow.server.jobs.executor_registry import JobExecutorRegistry

    with _job_executor_lock:
        if _job_executor is None:
            registry = JobExecutorRegistry(JobExecutorConfig())
            executor = DockerJobExecutor(registry.config)
            executor.start_executor()
            registry.register("docker", executor)
            _job_executor = registry.get("docker")
        return _job_executor


def _run_docker(workdir: Path, func_name: str) -> None:
    from mlflow.entities._job_status import JobStatus
    from mlflow.server.jobs.executor import JobExecutionContext

    executor = _get_scorer_job_executor()
    payload = json.loads((workdir / "payload.json").read_text())
    job_id = uuid.uuid4().hex
    executor.submit_job(
        job_id=job_id,
        job_name="scorer_sandbox",
        fn_fullname=_SCORER_JOB_FN,
        params={"payload": payload},
        context=JobExecutionContext(job_id=job_id, tracking_uri=""),
        timeout=_WALL_TIMEOUT_SECONDS,
    )
    result = executor.wait_for_job(job_id)
    # JobResult.result is the JSON text of run_scorer_job's envelope ({"ok", "result"}); write
    # it straight to result.json for run_scorer_in_sandbox. A crashed container yields a
    # non-SUCCEEDED status with no result -> _require_result raises with the executor's message.
    if result.status == JobStatus.SUCCEEDED and result.result is not None:
        (workdir / "result.json").write_text(result.result)
    _require_result(
        workdir,
        func_name,
        0 if result.status == JobStatus.SUCCEEDED else 1,
        result.error_message or "",
    )


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
_PROVIDERS = {"subprocess": _run_subprocess, "docker": _run_docker}

_subprocess_isolation_warned = False


def _auth_enabled() -> bool:
    """Whether MLflow's auth app is active (i.e. a multi-user server). Fails safe to False when
    the auth plugin isn't installed (localhost/dev), so single-user dev keeps the subprocess
    provider.
    """
    try:
        from mlflow.server.auth import is_auth_enabled
    except ImportError:
        return False
    return is_auth_enabled()


def _warn_subprocess_isolation_once() -> None:
    global _subprocess_isolation_warned
    if not _subprocess_isolation_warned:
        _subprocess_isolation_warned = True
        _logger.warning(
            "Server-side code scorers are using the 'subprocess' sandbox provider, which "
            "scrubs env vars and caps CPU/wall-time but does NOT confine filesystem reads or "
            "network egress: untrusted @scorer code can read local credential files and reach "
            "the network. Set MLFLOW_SCORER_SANDBOX_PROVIDER=docker for filesystem/network "
            "isolation."
        )


def _require_result(workdir: Path, func_name: str, returncode: int, stderr: str) -> None:
    if not (workdir / "result.json").exists():
        raise MlflowException(
            f"Sandboxed scorer '{func_name}' produced no result "
            f"(exit={returncode}). stderr:\n{stderr[-2000:]}"
        )


def run_scorer_in_sandbox(
    *, source: str, signature: str, func_name: str, call_kwargs: dict[str, Any]
) -> Any:
    """Reconstruct and run a decorator scorer's source in an isolated sandbox.

    Returns the scorer's result (primitive, ``Feedback``, or ``list[Feedback]``).
    Raises ``MlflowException`` if the sandbox cannot run or the scorer raises.
    """
    provider_name = MLFLOW_SCORER_SANDBOX_PROVIDER.get()
    provider = _PROVIDERS.get(provider_name)
    if provider is None:
        raise MlflowException.invalid_parameter_value(
            f"Unknown sandbox provider '{provider_name}'. Valid: {sorted(_PROVIDERS)}."
        )
    if provider_name == "subprocess":
        # The subprocess provider does not confine filesystem or network access, so on an
        # auth-enabled (multi-user) server it would let one user's scorer code read another
        # tenant's on-disk credentials / reach the network. Refuse it there — require docker.
        if _auth_enabled():
            raise MlflowException(
                "The 'subprocess' scorer sandbox provider does not isolate filesystem or "
                "network access and is not permitted on an auth-enabled server. Set "
                "MLFLOW_SCORER_SANDBOX_PROVIDER=docker."
            )
        _warn_subprocess_isolation_once()

    with tempfile.TemporaryDirectory(prefix="mlflow-scorer-sandbox-") as tmp:
        workdir = Path(tmp)
        try:
            payload = json.dumps({
                "source": source,
                "signature": signature,
                "func_name": func_name,
                "kwargs": _json_safe_kwargs(call_kwargs),
            })
        except TypeError as e:
            raise MlflowException.invalid_parameter_value(
                f"Scorer '{func_name}' received inputs that are not JSON-serializable and "
                f"cannot be sent to the sandbox: {e}"
            )
        (workdir / "payload.json").write_text(payload)

        try:
            provider(workdir, func_name)
        except subprocess.TimeoutExpired:
            raise MlflowException(
                f"Sandboxed scorer '{func_name}' timed out after {_WALL_TIMEOUT_SECONDS}s."
            )

        outcome = json.loads((workdir / "result.json").read_text())
        if not outcome.get("ok"):
            raise MlflowException(
                f"Sandboxed scorer '{func_name}' raised:\n"
                f"{outcome.get('traceback') or outcome.get('error')}"
            )
        return _deserialize_result(outcome["result"])
