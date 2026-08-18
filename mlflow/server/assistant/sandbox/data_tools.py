"""[POC] Server-side data tools for the MLflow Assistant (the "data tier").

These run in the SERVER process — which holds the tracking store and the caller's identity —
NOT in the sandbox. Each tool re-checks the caller's RBAC per call, then reads MLflow data
through the EXISTING tracking-store APIs (no parallel data API) and, for bulk reads,
materializes the results into the caller's network-isolated sandbox workspace for the
compute tools to analyze.

The set of tools here is the security allow-list: the LLM can only invoke these named,
RBAC-gated read operations — never arbitrary queries.
"""

import json
import logging
from typing import Any

from mlflow.tracking._tracking_service.utils import _get_store

_logger = logging.getLogger(__name__)

# The data-tool allow-list. Read-only for the POC (writes like log_feedback are deferred).
SERVER_DATA_TOOLS = {"search_traces", "get_trace"}

_SEARCH_DEFAULT_MAX = 10


def _can_read_experiment(experiment_id: str, caller: str) -> bool:
    """Non-raising RBAC check (tools return an error string, not an HTTP 403). Fails open
    only when the auth plugin isn't installed (no-auth dev server).
    """
    if experiment_id is None:
        return False
    try:
        from mlflow.server.auth import _get_experiment_permission, is_auth_enabled
    except ImportError:
        return True
    if not is_auth_enabled():
        return True
    return _get_experiment_permission(experiment_id, caller).can_read


def _run_search_traces(
    caller: str, session_id: str, tool_input: dict[str, Any], tracking_uri: str | None
) -> tuple[str, bool]:
    experiment_id = tool_input.get("experiment_id")
    if not experiment_id:
        return "search_traces requires an experiment_id.", True
    if not _can_read_experiment(experiment_id, caller):
        _logger.warning("data tool: %r denied read on experiment %s", caller, experiment_id)
        return f"Permission denied: you cannot read experiment {experiment_id}.", True

    store = _get_store()
    max_results = int(tool_input.get("max_results") or _SEARCH_DEFAULT_MAX)
    trace_infos, _ = store.search_traces(
        experiment_ids=[experiment_id],
        filter_string=tool_input.get("filter"),
        max_results=max_results,
    )
    # Materialize full traces into the sandbox workspace for the compute tools to analyze.
    from mlflow.server.assistant.sandbox.integration import materialize_in_sandbox

    written = []
    for info in trace_infos:
        trace = store.get_trace(info.trace_id)
        materialize_in_sandbox(
            session_id, f"traces/{info.trace_id}.json", trace.to_json(), tracking_uri, owner=caller
        )
        written.append(info.trace_id)
    _logger.info(
        "data tool: search_traces by %r on exp %s -> materialized %d traces",
        caller,
        experiment_id,
        len(written),
    )
    summary = {
        "materialized": len(written),
        "directory": "traces/",
        "trace_ids": written,
        "note": "Full trace JSON written to traces/<trace_id>.json in your workspace. "
        "Use the compute tools (bash/read) to analyze them.",
    }
    return json.dumps(summary, indent=2), False


def _run_get_trace(caller: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
    trace_id = tool_input.get("trace_id")
    if not trace_id:
        return "get_trace requires a trace_id.", True
    store = _get_store()
    info = store.get_trace_info(trace_id)
    experiment_id = info.experiment_id
    if experiment_id and not _can_read_experiment(experiment_id, caller):
        _logger.warning("data tool: %r denied get_trace on %s", caller, trace_id)
        return f"Permission denied: you cannot read trace {trace_id}.", True
    trace = store.get_trace(trace_id)
    _logger.info("data tool: get_trace %s by %r", trace_id, caller)
    return trace.to_json(), False


def run_data_tool(
    caller: str,
    session_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    tracking_uri: str | None = None,
) -> tuple[str, bool]:
    """Execute a server-side data tool, matching execute_tool's (output, is_error) contract."""
    if tool_name == "search_traces":
        return _run_search_traces(caller, session_id, tool_input, tracking_uri)
    if tool_name == "get_trace":
        return _run_get_trace(caller, tool_input)
    return f"Unknown data tool: {tool_name}", True
