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
import shlex
import socket
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import mlflow
from mlflow.environment_variables import (
    MLFLOW_ASSISTANT_NODE_ID,
    MLFLOW_SCORER_SANDBOX_DOCKER_IMAGE,
)
from mlflow.exceptions import MlflowException
from mlflow.server.assistant.sandbox.session_executor import (
    AbstractSessionExecutor,
    SessionContext,
    SessionRecoveryResult,
    ToolCall,
    ToolResult,
)

_logger = logging.getLogger(__name__)

_LABEL_ROLE = "mlflow.assistant.role"
# Namespaces this executor's containers (label value = pool name), so orphan reaping only ever
# touches this pool's containers, never anything else on the daemon.
_LABEL_POOL = "mlflow.sandbox.pool"
# Owning node/replica. Reaping is also scoped by this, so a replica never destroys another
# replica's live containers when they share one Docker daemon (e.g. distinct k8s pods on a host).
_LABEL_NODE = "mlflow.sandbox.node"

_WORKSPACE = "/workspace"
_MEMORY = "1g"
_NANO_CPUS = 1_000_000_000
_PIDS_LIMIT = 256
_RECONCILE_INTERVAL = 2.0
_EXEC_TIMEOUT = 120  # per compute-tool wall-clock cap, enforced by the in-container `timeout`
_MAX_OUTPUT_BYTES = 1_000_000  # cap on tool output pulled back into the server (host-OOM guard)

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


def _image() -> str:
    """Sandbox image name (read inside a function so tests/consumers can override the env var)."""
    return MLFLOW_SCORER_SANDBOX_DOCKER_IMAGE.get()


def get_node_id() -> str:
    """This replica's node identity: the configured id, or the hostname. Used to scope
    container reattach and orphan reaping so replicas sharing one Docker daemon don't collide.
    """
    return MLFLOW_ASSISTANT_NODE_ID.get() or socket.gethostname()


class DockerSessionExecutor(AbstractSessionExecutor):
    def __init__(
        self,
        min_idle: int = 2,
        max_total: int = 8,
        idle_ttl: float = 900.0,
        pool: str = "assistant",
    ) -> None:
        self._client = None
        self._pool = pool
        self._min_idle = min_idle
        self._max_total = max_total
        self._idle_ttl = idle_ttl  # seconds of inactivity before a session's sandbox is reaped
        self._idle: deque[str] = deque()  # container ids, clean + unassigned
        self._active: dict[str, str] = {}  # session_id -> container id
        self._last_activity: dict[str, float] = {}  # session_id -> monotonic timestamp
        # session ids whose sandbox was reaped for inactivity while still active; a
        # one-shot resume notice is delivered when such a session next runs a tool.
        self._reaped: set[str] = set()
        # Cold-path spawns in flight (reserved but not yet in _active), counted toward the
        # max_total ceiling so a burst of concurrent new sessions can't collectively exceed it.
        self._pending = 0
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

    def start_executor(self, recover_bindings: dict[str, str] | None = None) -> None:
        """Start the executor and (optionally) recover sandboxes from a previous process.

        ``recover_bindings`` maps ``session_id -> container_id`` for this node's unfinished
        sessions (loaded from the persisted session store). Live containers are reattached;
        sessions whose container is gone get a reap tombstone (so their next turn receives the
        "workspace was reset" notice). Every other container carrying our label is an orphan
        from a dead process and is destroyed — this is what keeps idle containers from
        accumulating across restarts.
        """
        self.check_requirements()
        self._ensure_image()
        if recover_bindings:
            self.recover_sessions(recover_bindings)
        self._reap_orphans()
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
            total = len(self._idle) + len(self._active) + self._pending
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
            _image(),
            command=["sleep", "infinity"],
            detach=True,
            labels={_LABEL_ROLE: "idle", _LABEL_POOL: self._pool, _LABEL_NODE: get_node_id()},
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

    # -- session lifecycle -------------------------------------------------------------
    def start_session(self, context: SessionContext) -> None:
        with self._lock:
            cid = self._idle.popleft() if self._idle else None
            spawn = cid is None
            if spawn:
                # Cold path: reserve a slot under the lock so max_total is a hard ceiling even
                # under a concurrent burst of new sessions (was previously unbounded).
                if len(self._active) + len(self._idle) + self._pending >= self._max_total:
                    raise MlflowException(
                        f"Assistant sandbox capacity reached ({self._max_total} containers); "
                        "retry shortly."
                    )
                self._pending += 1
        try:
            if spawn:
                cid = self._spawn_warm()
            container = self._get_client().containers.get(cid)
            container.exec_run(["true"])  # confirm liveness before adopting
        except Exception:
            if spawn:
                with self._lock:
                    self._pending -= 1
            raise
        with self._lock:
            self._active[context.session_id] = cid
            self._last_activity[context.session_id] = time.monotonic()
            if spawn:
                self._pending -= 1
        _logger.info(
            "assistant sandbox: session %s assigned container %s (owner=%r, idle pool: %d)",
            context.session_id,
            container.short_id,
            context.owner,
            len(self._idle),
        )
        # The maintainer backfills the pool asynchronously on its next reconcile tick.
        # The caller persists this session->container binding (a sidecar file) so
        # recover_sessions() can reattach the live container after a server restart; warm
        # containers are spawned before assignment, so the binding — not a container label —
        # is the authoritative record.

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
        # Cap output IN-CONTAINER: a tool emitting unbounded stdout (e.g. `cat /dev/zero`) would
        # otherwise stream multi-GB back into the server process and OOM the host — the container
        # mem limit doesn't cover output that crosses back to the server. `head -c` truncates at
        # the source; `pipefail` preserves the tool's real exit code through the pipe.
        capped = f"set -o pipefail; {shlex.join(cmd)} 2>&1 | head -c {_MAX_OUTPUT_BYTES}"
        result = container.exec_run(
            ["bash", "-lc", capped], workdir=_WORKSPACE, environment=env, demux=False
        )
        _logger.debug(
            "assistant sandbox: exec %s in session %s -> exit %s",
            call.tool,
            session_id,
            result.exit_code,
        )
        output = result.output.decode(errors="replace")
        if len(result.output) >= _MAX_OUTPUT_BYTES:
            output += f"\n\n[output truncated to {_MAX_OUTPUT_BYTES} bytes]"
        # 141 = SIGPIPE, which our own `head` cap raises on the writer when it truncates; that is
        # not a tool failure.
        return ToolResult(output=output, is_error=result.exit_code not in (0, 141))

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

    def recover_sessions(self, bindings: dict[str, str]) -> list[SessionRecoveryResult]:
        """Reattach sessions to their sandboxes after a restart, from a persisted map.

        Warm containers are spawned before assignment, so their labels can't carry the
        session id; the authoritative session -> container binding lives in a per-session
        sidecar file and is passed in here. For each binding whose container is still alive we
        repopulate ``_active`` (a true reattach); if the container is gone we set a reap
        tombstone so the session's next turn is told its workspace was reset.
        """
        results = []
        for session_id, container_id in bindings.items():
            if not container_id or not self._container_alive(container_id):
                with self._lock:
                    self._reaped.add(session_id)
                results.append(SessionRecoveryResult(session_id=session_id, action="fail"))
                continue
            with self._lock:
                self._active[session_id] = container_id
                self._last_activity[session_id] = time.monotonic()
            _logger.info(
                "assistant sandbox: reattached session %s to container %s",
                session_id,
                container_id[:12],
            )
            results.append(SessionRecoveryResult(session_id=session_id, action="reattach"))
        return results

    def _container_alive(self, container_id: str) -> bool:
        import docker.errors

        try:
            container = self._get_client().containers.get(container_id)
        except docker.errors.NotFound:
            return False
        container.reload()
        if container.status != "running":
            return False
        # A running status alone can be stale; confirm the container actually accepts execs.
        try:
            container.exec_run(["true"])
        except docker.errors.APIError:
            return False
        return True

    def _reap_orphans(self) -> None:
        """Destroy every container carrying our label that this executor isn't tracking.

        Runs at startup after reattach: reattached sessions are in ``_active`` and the pool is
        still empty, so anything else with our label is a leftover from a prior process
        (stale idle warmers or containers whose session is gone). Reaping them here is what
        prevents the idle-container pileup across restarts.
        """
        with self._lock:
            keep = set(self._active.values()) | set(self._idle)
        for container in self._get_client().containers.list(
            all=True,
            filters={"label": [f"{_LABEL_POOL}={self._pool}", f"{_LABEL_NODE}={get_node_id()}"]},
        ):
            if container.id not in keep:
                _logger.info("assistant sandbox: reaping orphan container %s", container.short_id)
                self._destroy(container.id)

    def get_container_id(self, session_id: str) -> str | None:
        with self._lock:
            return self._active.get(session_id)

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
