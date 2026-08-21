"""[POC] Server-side data tools for the MLflow Assistant (the "data tier").

These run in the SERVER process — which holds the tracking store and the caller's identity —
NOT in the sandbox. Each tool re-checks the caller's RBAC per call, then reads MLflow data
through the EXISTING tracking-store APIs (no parallel data API) and, for bulk reads,
materializes the results into the caller's network-isolated sandbox workspace for the
compute tools to analyze.
"""

import json
import logging
from typing import Any, Literal

from mlflow.entities import AssessmentSource, AssessmentSourceType, Feedback
from mlflow.tracking._tracking_service.utils import _get_store

_logger = logging.getLogger(__name__)

# The data-tool allow-list. Read tools (search_traces/get_trace) plus one write (log_feedback),
# each RBAC-gated per call. This named set IS the security surface: the LLM can only invoke
# these operations, never arbitrary store calls.
SERVER_DATA_TOOLS = {"search_traces", "get_trace", "log_feedback"}

_SEARCH_DEFAULT_MAX = 10


def _has_experiment_permission(
    experiment_id: str | None, caller: str, capability: Literal["can_read", "can_update"]
) -> bool:
    """Non-raising RBAC check (tools return an error string, not an HTTP 403). Fails open
    only when the auth plugin isn't installed (no-auth dev server). ``capability`` selects
    read (search/get) vs write (log_feedback) enforcement.
    """
    if experiment_id is None:
        return False
    try:
        from mlflow.server.auth import _get_experiment_permission, is_auth_enabled
    except ImportError:
        return True
    if not is_auth_enabled():
        return True
    permission = _get_experiment_permission(experiment_id, caller)
    return permission.can_update if capability == "can_update" else permission.can_read


def _can_read_experiment(experiment_id: str, caller: str) -> bool:
    return _has_experiment_permission(experiment_id, caller, "can_read")


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
    # Fail CLOSED: _can_read_experiment returns False for a falsy experiment_id, so a trace with
    # no experiment is denied rather than returned unchecked (matches the log_feedback path).
    if not _can_read_experiment(info.experiment_id, caller):
        _logger.warning("data tool: %r denied get_trace on %s", caller, trace_id)
        return f"Permission denied: you cannot read trace {trace_id}.", True
    trace = store.get_trace(trace_id)
    _logger.info("data tool: get_trace %s by %r", trace_id, caller)
    return trace.to_json(), False


def _run_log_feedback(caller: str, tool_input: dict[str, Any]) -> tuple[str, bool]:
    """Attach a feedback assessment to a trace (the one write tool). Gated on the caller's
    WRITE (can_update) permission on the trace's experiment, and attributed to the caller.
    """
    trace_id = tool_input.get("trace_id")
    if not trace_id:
        return "log_feedback requires a trace_id.", True
    if (value := tool_input.get("value")) is None:
        return "log_feedback requires a value.", True

    store = _get_store()
    experiment_id = store.get_trace_info(trace_id).experiment_id
    if not _has_experiment_permission(experiment_id, caller, "can_update"):
        _logger.warning("data tool: %r denied log_feedback on %s", caller, trace_id)
        return f"Permission denied: you cannot write to trace {trace_id}.", True

    feedback = Feedback(
        name=tool_input.get("name") or "feedback",
        value=value,
        rationale=tool_input.get("rationale"),
        trace_id=trace_id,
        # Attribute to the Assistant, and record which user drove it so the write is auditable.
        source=AssessmentSource(
            source_type=AssessmentSourceType.LLM_JUDGE, source_id="mlflow-assistant"
        ),
        metadata={"logged_by": caller},
    )
    created = store.create_assessment(feedback)
    _logger.info("data tool: log_feedback %r on %s by %r", feedback.name, trace_id, caller)
    return json.dumps({"logged": True, "assessment_id": created.assessment_id}), False


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
    if tool_name == "log_feedback":
        return _run_log_feedback(caller, tool_input)
    return f"Unknown data tool: {tool_name}", True
