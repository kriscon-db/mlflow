"""[POC] Docker implementation of ``AbstractSessionExecutor``.

Each Assistant conversation gets its own long-lived, network-isolated container. Compute
tools run via ``docker exec`` against that container, so scratch state (files the agent
writes) persists across turns. The container is destroyed at session end and never reused
for another session.

A background pool maintainer keeps ``min_idle`` clean, pre-warmed containers ready, so
``start_session`` assigns one instantly instead of paying container start on the critical
path. It reconciles to a desired idle count (event-driven on checkout + periodic self-heal),
rather than coupling replenishment to teardown.

POC scope: compute tools only (bash/read/write/edit). Data/privileged tools stay server-side
under the caller's RBAC, so the sandbox needs no network (``--network none``) and no creds.
"""

import base64
import logging
import os
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.server.assistant.sandbox.session_executor import (
    AbstractSessionExecutor,
    SessionContext,
    SessionRecoveryResult,
    ToolCall,
    ToolResult,
)

_logger = logging.getLogger(__name__)

_IMAGE = os.environ.get("MLFLOW_SCORER_SANDBOX_DOCKER_IMAGE", "mlflow-scorer-sandbox:poc")
_LABEL_ROLE = "mlflow.assistant.role"
_LABEL_SESSION = "mlflow.assistant.session_id"
_WORKSPACE = "/workspace"
_MEMORY = "1g"
_NANO_CPUS = 1_000_000_000
_PIDS_LIMIT = 256
_RECONCILE_INTERVAL = 2.0
_EXEC_TIMEOUT = 120  # per compute-tool wall-clock cap, enforced by the in-container `timeout`

_WRITE_SCRIPT = """
import base64, os
p = os.environ['P']
d = base64.b64decode(os.environ['C'])
os.makedirs(os.path.dirname(p) or '.', exist_ok=True)
open(p, 'wb').write(d)
print(f'wrote {len(d)} bytes to {p}')
"""

_EDIT_SCRIPT = """
import base64, os, sys
p = os.environ['P']
old = base64.b64decode(os.environ['O']).decode()
new = base64.b64decode(os.environ['N']).decode()
s = open(p, encoding='utf-8').read()
if old not in s:
    sys.exit(f'old_string not found in {p}')
open(p, 'w', encoding='utf-8').write(s.replace(old, new, 1))
print(f'edited {p}')
"""


class DockerSessionExecutor(AbstractSessionExecutor):
    def __init__(self, min_idle: int = 2, max_total: int = 8, idle_ttl: float = 900.0) -> None:
        self._client = None
        self._min_idle = min_idle
        self._max_total = max_total
        self._idle_ttl = idle_ttl  # seconds of inactivity before a session's sandbox is reaped
        self._idle: deque[str] = deque()  # container ids, clean + unassigned
        self._active: dict[str, str] = {}  # session_id -> container id
        self._last_activity: dict[str, float] = {}  # session_id -> monotonic timestamp
        # session ids whose sandbox was reaped for inactivity while still active; a
        # one-shot resume notice is delivered when such a session next runs a tool.
        self._reaped: set[str] = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._maintainer: threading.Thread | None = None

    @property
    def remote_execution(self) -> bool:
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
        self._reconcile()
        self._maintainer = threading.Thread(
            target=self._maintain, name="mlflow-assistant-sandbox-maintainer", daemon=True
        )
        self._maintainer.start()

    def stop_executor(self) -> None:
        self._stop.set()
        if self._maintainer:
            self._maintainer.join(timeout=5)
        with self._lock:
            ids = list(self._idle) + list(self._active.values())
            self._idle.clear()
            self._active.clear()
        for cid in ids:
            self._destroy(cid)

    # -- pool maintainer (reconciler) --------------------------------------------------
    def _maintain(self) -> None:
        while not self._stop.wait(_RECONCILE_INTERVAL):
            try:
                self._reap_idle_sessions()
                self._reconcile()
            except Exception:
                pass

    def _reap_idle_sessions(self) -> None:
        now = time.monotonic()
        with self._lock:
            stale = [
                sid for sid, last in self._last_activity.items() if now - last > self._idle_ttl
            ]
        for sid in stale:
            _logger.info("assistant sandbox: reaping idle session %s", sid)
            # tombstone=True sets the reap marker atomically with teardown, so a resume that
            # races in right after still receives the "workspace was reset" notice.
            self.stop_session(sid, tombstone=True)

    def consume_reap_notice(self, session_id: str) -> str | None:
        """One-shot notice if this session's sandbox was reaped for inactivity.

        Returns the notice (and clears the tombstone) so the caller can tell the agent its
        workspace was reset; returns None otherwise.
        """
        with self._lock:
            if session_id not in self._reaped:
                return None
            self._reaped.discard(session_id)
        minutes = int(self._idle_ttl // 60)
        span = f"{minutes} minute(s)" if minutes else f"{int(self._idle_ttl)} second(s)"
        return (
            "[sandbox notice] Your working directory was cleared after "
            f"{span} of inactivity; files created earlier in this conversation are no longer "
            "available. Recreate anything you need before using it."
        )

    def _reconcile(self) -> None:
        with self._lock:
            total = len(self._idle) + len(self._active)
            deficit = self._min_idle - len(self._idle)
            headroom = self._max_total - total
            to_spawn = max(0, min(deficit, headroom))
        for _ in range(to_spawn):
            cid = self._spawn_warm()
            with self._lock:
                self._idle.append(cid)

    def _spawn_warm(self) -> str:
        client = self._get_client()
        mlflow_pkg = Path(mlflow.__file__).resolve().parent
        container = client.containers.run(
            _IMAGE,
            command=["sleep", "infinity"],
            detach=True,
            labels={_LABEL_ROLE: "idle"},
            network_mode="none",
            mem_limit=_MEMORY,
            memswap_limit=_MEMORY,
            nano_cpus=_NANO_CPUS,
            pids_limit=_PIDS_LIMIT,
            read_only=True,
            cap_drop=["ALL"],
            security_opt=["no-new-privileges:true"],
            # Writable scratch (persists across turns, dies with the container). mode 1777
            # so the non-root exec user can write.
            tmpfs={_WORKSPACE: "rw,mode=1777", "/tmp": "rw,mode=1777"},
            user=f"{os.getuid()}:{os.getgid()}",
            working_dir=_WORKSPACE,
            environment={"PYTHONPATH": "/mlflow-src", "HOME": _WORKSPACE},
            volumes={str(mlflow_pkg): {"bind": "/mlflow-src/mlflow", "mode": "ro"}},
        )
        _logger.debug("assistant sandbox: spawned warm container %s", container.short_id)
        return container.id

    def _ensure_image(self) -> None:
        import docker.errors

        client = self._get_client()
        try:
            client.images.get(_IMAGE)
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
            client.images.build(path=ctx, tag=_IMAGE, buildargs=buildargs, rm=True)

    # -- session lifecycle -------------------------------------------------------------
    def start_session(self, context: SessionContext) -> None:
        with self._lock:
            cid = self._idle.popleft() if self._idle else None
        if cid is None:
            # Cold path: pool exhausted, spawn synchronously.
            cid = self._spawn_warm()
        container = self._get_client().containers.get(cid)
        container.exec_run(["true"])  # confirm liveness before adopting
        with self._lock:
            self._active[context.session_id] = cid
            self._last_activity[context.session_id] = time.monotonic()
        _logger.info(
            "assistant sandbox: session %s assigned container %s (owner=%r, idle pool: %d)",
            context.session_id,
            container.short_id,
            context.owner,
            len(self._idle),
        )
        # The maintainer backfills the pool asynchronously on its next reconcile tick.
        # NOTE: warm containers are spawned before assignment, so their labels can't carry
        # the session id. Full reattach-after-restart therefore needs a persisted
        # session->container map (the server-side equivalent of the job store); until then
        # recover_sessions() only distinguishes "known running" from "gone".

    def exec_in_session(self, session_id: str, call: ToolCall) -> ToolResult:
        cid = self._active.get(session_id)
        if cid is None:
            return ToolResult(output=f"No active sandbox for session {session_id}", is_error=True)
        with self._lock:
            self._last_activity[session_id] = time.monotonic()
        container = self._get_client().containers.get(cid)
        cmd, env = self._build_exec(call)
        # Prefix the in-container `timeout` so a runaway tool (e.g. `sleep infinity`, an
        # infinite loop) can't wedge the calling worker thread forever. Exit 124 on timeout.
        cmd = ["timeout", str(_EXEC_TIMEOUT), *cmd]
        result = container.exec_run(cmd, workdir=_WORKSPACE, environment=env, demux=False)
        _logger.debug(
            "assistant sandbox: exec %s in session %s -> exit %s",
            call.tool,
            session_id,
            result.exit_code,
        )
        return ToolResult(
            output=result.output.decode(errors="replace"),
            is_error=result.exit_code != 0,
        )

    def stop_session(self, session_id: str, tombstone: bool = False) -> None:
        with self._lock:
            cid = self._active.pop(session_id, None)
            self._last_activity.pop(session_id, None)
            # Under the same lock that removes the session: set the tombstone when reaping
            # (so a racing resume still gets the notice), or clear it on an explicit end (so
            # `_reaped` can't accumulate ids that will never resume — finding #10).
            if tombstone:
                self._reaped.add(session_id)
            else:
                self._reaped.discard(session_id)
        if cid:
            self._destroy(cid)
            _logger.info("assistant sandbox: session %s sandbox destroyed", session_id)

    def recover_sessions(self, unfinished_session_ids: list[str]) -> list[SessionRecoveryResult]:
        client = self._get_client()
        results = []
        for session_id in unfinished_session_ids:
            containers = client.containers.list(
                all=True, filters={"label": f"{_LABEL_SESSION}={session_id}"}
            )
            if not containers:
                results.append(SessionRecoveryResult(session_id=session_id, action="requeue"))
                continue
            container = containers[0]
            action = "reattach" if container.status == "running" else "fail"
            results.append(SessionRecoveryResult(session_id=session_id, action=action))
        return results

    # -- helpers -----------------------------------------------------------------------
    def _build_exec(self, call: ToolCall) -> tuple[list[str], dict[str, str]]:
        # Content/strings are passed via env as base64 to avoid shell/argv escaping issues.
        if call.tool == "bash":
            return ["bash", "-lc", call.args["command"]], {}
        if call.tool == "read":
            return ["cat", call.args["path"]], {}
        if call.tool == "write":
            env = {
                "P": call.args["path"],
                "C": base64.b64encode(call.args["content"].encode()).decode(),
            }
            return ["python3", "-c", _WRITE_SCRIPT], env
        if call.tool == "edit":
            env = {
                "P": call.args["path"],
                "O": base64.b64encode(call.args["old"].encode()).decode(),
                "N": base64.b64encode(call.args["new"].encode()).decode(),
            }
            return ["python3", "-c", _EDIT_SCRIPT], env
        raise MlflowException.invalid_parameter_value(f"Unknown compute tool: {call.tool}")

    def is_active(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._active

    def _destroy(self, container_id: str) -> None:
        try:
            container = self._get_client().containers.get(container_id)
            container.remove(force=True)
        except Exception:
            pass

    def idle_count(self) -> int:
        with self._lock:
            return len(self._idle)
