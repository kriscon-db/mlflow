"""[POC] Glue between the Assistant tool loop and the per-session Docker sandbox.

When ``MLFLOW_ENABLE_ASSISTANT_SANDBOX`` is set, ``tool_executor.execute_tool`` routes the
compute tools (Bash/Read/Write/Edit) here instead of running them in the server process.
A single process-wide ``DockerSessionExecutor`` (with a warm pool) backs all sessions; each
Assistant conversation lazily gets its own sandbox on first compute-tool call.
"""

import logging
import threading
from typing import Any

from mlflow.environment_variables import (
    MLFLOW_ASSISTANT_SANDBOX_IDLE_TTL,
    MLFLOW_ASSISTANT_SANDBOX_MAX_TOTAL,
    MLFLOW_ASSISTANT_SANDBOX_MIN_IDLE,
)
from mlflow.server.assistant.sandbox.docker_session_executor import (
    DockerSessionExecutor,
    get_node_id,
)
from mlflow.server.assistant.sandbox.session_executor import SessionContext, ToolCall
from mlflow.server.assistant.session import (
    clear_container_binding,
    list_container_bindings,
    save_container_binding,
)

_logger = logging.getLogger(__name__)

# Assistant tool-schema names that are isolated compute (routed to the sandbox). Data/
# privileged tools are not in this set: the server executes those under the caller's RBAC.
SANDBOX_TOOLS = {"Bash", "Read", "Write", "Edit"}

_executor: DockerSessionExecutor | None = None
_lock = threading.Lock()


def _recover_bindings() -> dict[str, str]:
    """Session -> container bindings this node should reattach after a restart.

    Reads the persisted session store and keeps only bindings owned by this node (a container
    on another host is unreachable from here). Anything not returned here that is still running
    gets reaped as an orphan by the executor.
    """
    return {
        session_id: container_id
        for session_id, (container_id, node_id) in list_container_bindings().items()
        if node_id == get_node_id() and container_id
    }


def _get_executor() -> DockerSessionExecutor:
    global _executor
    with _lock:
        if _executor is None:
            executor = DockerSessionExecutor(
                min_idle=MLFLOW_ASSISTANT_SANDBOX_MIN_IDLE.get(),
                max_total=MLFLOW_ASSISTANT_SANDBOX_MAX_TOTAL.get(),
                idle_ttl=MLFLOW_ASSISTANT_SANDBOX_IDLE_TTL.get(),
            )
            executor.start_executor(recover_bindings=_recover_bindings())
            _executor = executor
        return _executor


def _to_tool_call(tool_name: str, tool_input: dict[str, Any]) -> ToolCall | None:
    path = tool_input.get("file_path") or tool_input.get("path", "")
    match tool_name:
        case "Bash":
            return ToolCall("bash", {"command": tool_input.get("command", "")})
        case "Read":
            return ToolCall("read", {"path": path})
        case "Write":
            return ToolCall("write", {"path": path, "content": tool_input.get("content", "")})
        case "Edit":
            return ToolCall(
                "edit",
                {
                    "path": path,
                    "old": tool_input.get("old_string", ""),
                    "new": tool_input.get("new_string", ""),
                },
            )
        case _:
            return None


def _ensure_session(executor, session_id, tracking_uri, owner=None):
    """Lazily start the session's sandbox; return a one-shot resume notice if one is pending."""
    notice = None
    if not executor.is_active(session_id):
        # A resume after the sandbox was reaped for inactivity: tell the agent its scratch
        # workspace was reset (returns None for a brand-new session that was never started).
        notice = executor.consume_reap_notice(session_id)
        if notice:
            _logger.info("assistant sandbox: delivering resume notice to session %s", session_id)
        executor.start_session(
            SessionContext(session_id=session_id, tracking_uri=tracking_uri or "", owner=owner)
        )
        _persist_binding(executor, session_id)
    return notice


def _persist_binding(executor, session_id: str) -> None:
    """Record the session's assigned container (in a sidecar file) so the sandbox can be
    reattached after a server restart.
    """
    if container_id := executor.get_container_id(session_id):
        save_container_binding(session_id, container_id, get_node_id())


def run_sandboxed_tool(
    session_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    tracking_uri: str | None = None,
) -> tuple[str, bool]:
    """Run one compute tool in the session's sandbox, matching execute_tool's (output, is_error).

    Synchronous (docker-py is blocking); ``execute_tool`` calls this via ``asyncio.to_thread``.
    """
    call = _to_tool_call(tool_name, tool_input)
    if call is None:
        return f"Unknown sandbox tool: {tool_name}", True
    executor = _get_executor()
    notice = _ensure_session(executor, session_id, tracking_uri)
    result = executor.exec_in_session(session_id, call)
    output = f"{notice}\n{result.output}" if notice else result.output
    return output, result.is_error


def materialize_in_sandbox(
    session_id: str, rel_path: str, content: str, tracking_uri: str | None = None, owner=None
) -> None:
    """Write server-fetched data (e.g. traces) into the session's sandbox workspace so the
    agent's compute tools can analyze it. Reuses the sandbox write path; lazy-starts the
    session. Data flows server -> sandbox only; the sandbox never reaches out for it.
    """
    executor = _get_executor()
    _ensure_session(executor, session_id, tracking_uri, owner=owner)
    result = executor.exec_in_session(
        session_id, ToolCall("write", {"path": rel_path, "content": content})
    )
    if result.is_error:
        raise RuntimeError(f"Failed to materialize {rel_path} into the sandbox: {result.output}")


def stop_sandbox_session(session_id: str) -> None:
    """Tear down a session's sandbox (called when the Assistant session ends/cancels)."""
    if _executor is not None:
        _executor.stop_session(session_id)
    # Clear the persisted binding: the container is gone, so a later restart must not try to
    # reattach it (or tombstone a session the user deliberately ended).
    clear_container_binding(session_id)
