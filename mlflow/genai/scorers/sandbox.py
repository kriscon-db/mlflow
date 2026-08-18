"""[POC] Run a ``@scorer`` code scorer's source in an isolated sandbox.

This is the Project Delight sandbox seam. It exposes one entry point,
``run_scorer_in_sandbox``, backed by a pluggable provider selected via
``MLFLOW_SCORER_SANDBOX_PROVIDER``:

- ``subprocess`` (default): a scrubbed-env child process with a CPU limit and a wall-clock
  timeout. Portable, no infra, but does NOT confine filesystem reads or network egress.
- ``docker``: a locked-down ``docker run`` (``--network none``, resource limits, non-root,
  read-only rootfs). Real fs + network isolation. Requires a running docker daemon.

Either way the scorer's source is NEVER ``exec()``-d in the server/worker process — the
source + inputs are shipped to ``_sandbox_runner``, which runs inside the sandbox.
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import mlflow
from mlflow.entities import Feedback
from mlflow.exceptions import MlflowException
from mlflow.utils.os import is_windows

_logger = logging.getLogger(__name__)

_SECRET_SUBSTRINGS = ("TOKEN", "SECRET", "PASSWORD", "APIKEY", "API_KEY", "CREDENTIAL")
_SECRET_PREFIXES = (
    "MLFLOW_TRACKING_",
    "DATABRICKS",
    "AWS_",
    "AZURE_",
    "GOOGLE_",
    "GCP_",
    "OPENAI",
    "ANTHROPIC",
)

_CPU_SECONDS = 30
_WALL_TIMEOUT_SECONDS = 120
_MEMORY = "1g"
_PIDS_LIMIT = "256"
_DEFAULT_DOCKER_IMAGE = "mlflow-scorer-sandbox:poc"
_RUNNER_MODULE = "mlflow.genai.scorers._sandbox_runner"


# --------------------------------------------------------------------------------------
# Payload / result plumbing (backend-agnostic)
# --------------------------------------------------------------------------------------
def _is_secret_env(name: str) -> bool:
    upper = name.upper()
    return upper.startswith(_SECRET_PREFIXES) or any(s in upper for s in _SECRET_SUBSTRINGS)


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


def _deserialize_result(result: dict[str, Any]) -> Any:
    kind = result["kind"]
    if kind == "feedback":
        v = result["value"]
        return Feedback(name=v.get("name"), value=v.get("value"), rationale=v.get("rationale"))
    if kind == "feedback_list":
        return [
            Feedback(name=v.get("name"), value=v.get("value"), rationale=v.get("rationale"))
            for v in result["value"]
        ]
    return result["value"]


# --------------------------------------------------------------------------------------
# Provider: subprocess (default, portable, no infra)
# --------------------------------------------------------------------------------------
def _scrubbed_env(workdir: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not _is_secret_env(k)}
    env["HOME"] = str(workdir)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    env.pop("MLFLOW_TRACKING_URI", None)
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


# --------------------------------------------------------------------------------------
# Provider: docker (real fs + network isolation; requires a running daemon)
# --------------------------------------------------------------------------------------
def _docker_image() -> str:
    return os.environ.get("MLFLOW_SCORER_SANDBOX_DOCKER_IMAGE", _DEFAULT_DOCKER_IMAGE)


def _ensure_docker_image(image: str) -> None:
    """Build the sandbox base image on first use if it isn't present.

    The image only needs mlflow's runtime dependencies installed; the actual mlflow
    package (including the new runner) is bind-mounted from the local source at run time,
    so the container always runs this branch's code regardless of the image's mlflow.
    """
    inspect = subprocess.run(["docker", "image", "inspect", image], capture_output=True, text=True)
    if inspect.returncode == 0:
        return
    with tempfile.TemporaryDirectory(prefix="mlflow-sandbox-image-") as ctx:
        dockerfile = (
            "FROM python:3.11-slim\n"
            # Honor a host-provided pip index (e.g. an internal proxy behind a VPN that
            # blackholes pypi.org). Empty by default, so plain PyPI is used elsewhere.
            "ARG PIP_INDEX_URL=\n"
            "ARG PIP_EXTRA_INDEX_URL=\n"
            "ENV PIP_INDEX_URL=$PIP_INDEX_URL PIP_EXTRA_INDEX_URL=$PIP_EXTRA_INDEX_URL\n"
            # Runtime deps only; mlflow source is mounted at run time.
            "RUN pip install --no-cache-dir mlflow\n"
        )
        Path(ctx, "Dockerfile").write_text(dockerfile)
        # --load so the buildx (docker-container driver) result lands in the local image
        # store rather than only the build cache.
        build_cmd = ["docker", "build", "--load", "-t", image]
        for arg in ("PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL"):
            if os.environ.get(arg):
                build_cmd += ["--build-arg", f"{arg}={os.environ[arg]}"]
        build_cmd.append(ctx)
        build = subprocess.run(
            build_cmd,
            capture_output=True,
            text=True,
            timeout=600,
        )
        if build.returncode != 0:
            raise MlflowException(
                f"Failed to build sandbox image '{image}':\n{build.stderr[-2000:]}"
            )


def _run_docker(workdir: Path, func_name: str) -> None:
    image = _docker_image()
    _ensure_docker_image(image)
    mlflow_pkg = Path(mlflow.__file__).resolve().parent  # .../mlflow
    cmd = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--memory",
        _MEMORY,
        "--memory-swap",
        _MEMORY,
        "--pids-limit",
        _PIDS_LIMIT,
        "--cpus",
        "1",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--tmpfs",
        "/tmp",
        # Run as the host user so the bind-mounted workdir stays writable for result.json.
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-v",
        f"{workdir}:/sandbox",
        "-v",
        f"{mlflow_pkg}:/mlflow-src/mlflow:ro",
        "-e",
        "PYTHONPATH=/mlflow-src",
        "-e",
        "HOME=/sandbox",
        "-w",
        "/sandbox",
        image,
        "python",
        "-m",
        _RUNNER_MODULE,
        "/sandbox",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=_WALL_TIMEOUT_SECONDS)
    _require_result(workdir, func_name, proc.returncode, proc.stderr)


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
_PROVIDERS = {"subprocess": _run_subprocess, "docker": _run_docker}

_subprocess_isolation_warned = False


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
    provider_name = os.environ.get("MLFLOW_SCORER_SANDBOX_PROVIDER", "subprocess")
    provider = _PROVIDERS.get(provider_name)
    if provider is None:
        raise MlflowException.invalid_parameter_value(
            f"Unknown sandbox provider '{provider_name}'. Valid: {sorted(_PROVIDERS)}."
        )
    if provider_name == "subprocess":
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
