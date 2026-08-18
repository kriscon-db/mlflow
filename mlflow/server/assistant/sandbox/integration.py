"""[POC] Glue between the Assistant tool loop and the per-session Docker sandbox.

When ``MLFLOW_ENABLE_ASSISTANT_SANDBOX`` is set, ``tool_executor.execute_tool`` routes the
compute tools (Bash/Read/Write/Edit) here instead of running them in the server process.
A single process-wide ``DockerSessionExecutor`` (with a warm pool) backs all sessions; each
Assistant conversation lazily gets its own sandbox on first compute-tool call.
"""

import logging
import threading
from typing import Any

from mlflow.server.assistant.sandbox.docker_session_executor import DockerSessionExecutor
from mlflow.server.assistant.sandbox.session_executor import SessionContext, ToolCall

_logger = logging.getLogger(__name__)

# Assistant tool-schema names that are isolated compute (routed to the sandbox). Data/
# privileged tools are not in this set: the server executes those under the caller's RBAC.
SANDBOX_TOOLS = {"Bash", "Read", "Write", "Edit"}

_executor: DockerSessionExecutor | None = None
_lock = threading.Lock()


def _get_executor() -> DockerSessionExecutor:
    global _executor
    with _lock:
        if _executor is None:
            executor = DockerSessionExecutor()
            executor.start_executor()
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
    notice: str | None = None
    if not executor.is_active(session_id):
        # A resume after the sandbox was reaped for inactivity: tell the agent its scratch
        # workspace was reset (returns None for a brand-new session that was never started).
        notice = executor.consume_reap_notice(session_id)
        if notice:
            _logger.info("assistant sandbox: delivering resume notice to session %s", session_id)
        executor.start_session(
            SessionContext(session_id=session_id, tracking_uri=tracking_uri or "")
        )
    result = executor.exec_in_session(session_id, call)
    output = f"{notice}\n{result.output}" if notice else result.output
    return output, result.is_error


def stop_sandbox_session(session_id: str) -> None:
    """Tear down a session's sandbox (called when the Assistant session ends/cancels)."""
    if _executor is not None:
        _executor.stop_session(session_id)
