import asyncio
import logging
import os
import shlex
from pathlib import Path
from typing import Any

from mlflow.assistant.config import PermissionsConfig
from mlflow.assistant.custom_view import RENDER_CUSTOM_VIEW_TOOL_NAME
from mlflow.environment_variables import MLFLOW_ENABLE_ASSISTANT_SANDBOX

_logger = logging.getLogger(__name__)

_FILE_TOOLS = {"Read", "Write", "Edit"}
# Restricted mode only permits MLflow CLI and Python; anything else needs Full Access.
_ALLOWED_BASH_COMMANDS = {"mlflow", "python3", "python"}
# Compute tools that run in the per-session sandbox when MLFLOW_ENABLE_ASSISTANT_SANDBOX is set.
_SANDBOX_TOOLS = {"Bash", "Read", "Write", "Edit"}
# Server-side data tools (the data tier): run in the server under the caller's RBAC and
# materialize results into the sandbox. Names mirror mlflow.server.assistant.sandbox.data_tools.
_SERVER_DATA_TOOLS = {"search_traces", "get_trace"}

# Tools executed on the CLIENT (browser), not the server: the assistant loop pauses the turn and
# waits for a client-submitted result instead of routing the call through execute_tool/the static
# permission gate. See openai_compatible.py's tool loop.
CLIENT_TOOLS = {RENDER_CUSTOM_VIEW_TOOL_NAME}


def _is_path_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _resolve_file_path(raw_path: str, cwd: Path | None) -> Path:
    p = Path(raw_path).expanduser()
    if not p.is_absolute() and cwd:
        p = cwd / p
    return p.resolve()


def static_permission_error(
    tool_name: str,
    tool_input: dict[str, Any],
    perms: PermissionsConfig,
    cwd: Path | None,
) -> str | None:
    """Return a denial message if the call is NOT permitted under static (non-full-access)
    permissions, or None if it is allowed.

    Shared by ``execute_tool`` (to enforce the policy) and the assistant's per-call permission gate
    (to decide whether an interactive prompt is even needed): a call the static policy already
    allows — e.g. an ``mlflow`` CLI command or an in-workspace file op — runs without prompting,
    just as it did before tool-call permissions existed.
    """
    if perms.full_access:
        return None

    if tool_name == "Bash":
        command = tool_input.get("command", "").strip()
        try:
            argv = shlex.split(command)
        except ValueError:
            return "Permission denied: malformed command"
        if not argv or argv[0] not in _ALLOWED_BASH_COMMANDS:
            return (
                f"Permission denied: only {', '.join(sorted(_ALLOWED_BASH_COMMANDS))} "
                "commands are allowed"
            )

    if tool_name in _FILE_TOOLS and not perms.allow_edit_files:
        return f"Permission denied: {tool_name} is not allowed"

    if tool_name in {"Write", "Edit"} and not cwd:
        return f"Permission denied: {tool_name} requires a configured project directory"

    if tool_name in _FILE_TOOLS and cwd:
        if raw_path := tool_input.get("file_path") or tool_input.get("path", ""):
            target = _resolve_file_path(raw_path, cwd)
            if not _is_path_within(target, cwd):
                return f"Permission denied: path {raw_path} is outside the workspace {cwd}"

    return None


async def execute_tool(
    tool_name: str,
    tool_input: dict[str, Any],
    cwd: Path | None = None,
    tracking_uri: str | None = None,
    permissions: PermissionsConfig | None = None,
    session_id: str | None = None,
    caller: str | None = None,
) -> tuple[str, bool]:
    # Data tier: server-side, RBAC-checked read tools that fetch MLflow data as the caller and
    # materialize it into the sandbox. Run in the server process (which holds the store + identity),
    # never in the network-isolated sandbox.
    if session_id and MLFLOW_ENABLE_ASSISTANT_SANDBOX.get() and tool_name in _SERVER_DATA_TOOLS:
        from mlflow.server.assistant.sandbox.data_tools import run_data_tool

        _logger.info("routing tool %s for session %s -> server data tier", tool_name, session_id)
        try:
            return await asyncio.to_thread(
                run_data_tool, caller, session_id, tool_name, tool_input, tracking_uri
            )
        except Exception as e:
            _logger.exception("data tool failed for %s", tool_name)
            return f"Data tool failed: {e}", True

    # When the Assistant sandbox is enabled, compute tools run inside the session's isolated
    # container instead of the server process. Isolation is the safety boundary, so the static
    # host-permission gate (workspace confinement, allow-listed commands) is bypassed here.
    # Only the server-loop providers (openai_compatible / mlflow_gateway) reach this path with a
    # session_id; the CLI providers (claude_code, codex) run their own host process and are
    # localhost-only (allows_remote_access=False), so they never expose host exec to remote users.
    if session_id and MLFLOW_ENABLE_ASSISTANT_SANDBOX.get() and tool_name in _SANDBOX_TOOLS:
        from mlflow.server.assistant.sandbox.integration import run_sandboxed_tool

        _logger.info("routing tool %s for session %s -> sandbox", tool_name, session_id)
        try:
            return await asyncio.to_thread(
                run_sandboxed_tool, session_id, tool_name, tool_input, tracking_uri
            )
        except Exception as e:
            # Docker missing/daemon down, or the sandbox was torn down mid-call: surface a
            # tool error to the model instead of crashing the streaming turn.
            _logger.exception("sandbox execution failed for tool %s", tool_name)
            return f"Sandbox unavailable: {e}", True

    perms = permissions or PermissionsConfig()

    if (denial := static_permission_error(tool_name, tool_input, perms, cwd)) is not None:
        return denial, True

    try:
        match tool_name:
            case "Bash":
                return await _execute_bash(tool_input, cwd=cwd, tracking_uri=tracking_uri)
            case "Read":
                return await asyncio.to_thread(_execute_read, tool_input, cwd=cwd)
            case "Write":
                return await asyncio.to_thread(_execute_write, tool_input, cwd=cwd)
            case "Edit":
                return await asyncio.to_thread(_execute_edit, tool_input, cwd=cwd)
            case _:
                return f"Unknown tool: {tool_name}", True
    except Exception as e:
        _logger.exception("Tool execution error for %s", tool_name)
        return f"Tool execution failed: {e}", True


async def _execute_bash(
    tool_input: dict[str, Any],
    cwd: Path | None,
    tracking_uri: str | None,
) -> tuple[str, bool]:
    command = tool_input.get("command", "")
    if not command:
        return "No command provided", True

    env = os.environ.copy()
    if tracking_uri:
        env["MLFLOW_TRACKING_URI"] = tracking_uri

    try:
        # Shell required: LLM-generated commands may use pipes, redirects, or && chaining.
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        output = stdout.decode("utf-8", errors="replace")
        err_output = stderr.decode("utf-8", errors="replace")

        if proc.returncode != 0:
            result = (
                output + err_output if output or err_output else f"Exit code: {proc.returncode}"
            )
            return result.strip(), True

        return (output + err_output).strip() or "(no output)", False
    except asyncio.TimeoutError:
        return "Command timed out after 120 seconds", True


def _execute_read(tool_input: dict[str, Any], cwd: Path | None = None) -> tuple[str, bool]:
    file_path = tool_input.get("file_path") or tool_input.get("path", "")
    if not file_path:
        return "No file_path provided", True
    try:
        content = _resolve_file_path(file_path, cwd).read_text(encoding="utf-8")
        return content, False
    except Exception as e:
        return str(e), True


def _execute_write(tool_input: dict[str, Any], cwd: Path | None = None) -> tuple[str, bool]:
    file_path = tool_input.get("file_path") or tool_input.get("path", "")
    content = tool_input.get("content", "")
    if not file_path:
        return "No file_path provided", True
    try:
        p = _resolve_file_path(file_path, cwd)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to {file_path}", False
    except Exception as e:
        return str(e), True


def _execute_edit(tool_input: dict[str, Any], cwd: Path | None = None) -> tuple[str, bool]:
    file_path = tool_input.get("file_path") or tool_input.get("path", "")
    old_string = tool_input.get("old_string", "")
    new_string = tool_input.get("new_string", "")
    if not file_path:
        return "No file_path provided", True
    try:
        p = _resolve_file_path(file_path, cwd)
        content = p.read_text(encoding="utf-8")
        if old_string not in content:
            return f"old_string not found in {file_path}", True
        new_content = content.replace(old_string, new_string, 1)
        p.write_text(new_content, encoding="utf-8")
        return f"Edited {file_path}", False
    except Exception as e:
        return str(e), True


def build_tools_schema() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "Bash",
                "description": (
                    "Execute a shell command to query or interact with MLflow. "
                    "Use 'mlflow' CLI commands or Python one-liners with the MLflow SDK."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute.",
                        }
                    },
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read the contents of a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute or relative path to the file.",
                        }
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Write",
                "description": "Write content to a file (creates or overwrites).",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute or relative path to the file.",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write.",
                        },
                    },
                    "required": ["file_path", "content"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Edit",
                "description": (
                    "Replace the first occurrence of old_string with new_string in a file."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute or relative path to the file.",
                        },
                        "old_string": {
                            "type": "string",
                            "description": "Exact string to find.",
                        },
                        "new_string": {
                            "type": "string",
                            "description": "String to replace it with.",
                        },
                    },
                    "required": ["file_path", "old_string", "new_string"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": RENDER_CUSTOM_VIEW_TOOL_NAME,
                "description": (
                    "Render a custom trace view in the UI: a reusable, trace-agnostic layout of "
                    "cards, stat tiles, key-value viewers, and assessment boards, built from the "
                    "current trace's data. Call this once you've designed the layout; the client "
                    "renders it and reports back whether it applied successfully."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {
                            "type": "string",
                            "description": "Short display title for the view.",
                        },
                        "messages": {
                            "type": "array",
                            "description": (
                                "A2UI message list describing the view's component tree."
                            ),
                            "items": {"type": "object"},
                        },
                    },
                    "required": ["title", "messages"],
                },
            },
        },
    ] + (_DATA_TOOL_SCHEMAS if MLFLOW_ENABLE_ASSISTANT_SANDBOX.get() else [])


# Advertised only when the sandbox (data tier) is enabled; execute_tool routes these to the
# server-side, RBAC-checked data tools which materialize results into the sandbox workspace.
_DATA_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_traces",
            "description": (
                "Search MLflow traces in an experiment and materialize the full trace JSON into "
                "your sandbox workspace under traces/<trace_id>.json for analysis with the "
                "compute tools (Bash/Read). Returns a summary of what was written."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "experiment_id": {
                        "type": "string",
                        "description": "The experiment to search traces in.",
                    },
                    "filter": {
                        "type": "string",
                        "description": "Optional MLflow trace filter string.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Max traces to fetch (default 10).",
                    },
                },
                "required": ["experiment_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_trace",
            "description": "Fetch a single MLflow trace by ID and return its full JSON inline.",
            "parameters": {
                "type": "object",
                "properties": {
                    "trace_id": {"type": "string", "description": "The trace ID to fetch."},
                },
                "required": ["trace_id"],
            },
        },
    },
]
