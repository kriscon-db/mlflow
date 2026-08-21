"""[POC] Reconstruct and run a ``@scorer`` code scorer inside an isolated sandbox.

Two entry points share one core (``run_scorer_payload``):

- ``python -m mlflow.genai.scorers._sandbox_runner <workdir>`` — the subprocess provider:
  reads ``<workdir>/payload.json`` and writes ``<workdir>/result.json``.
- ``run_scorer_job(payload)`` — imported by fully-qualified name and run as a *job* inside a
  job-executor's sandbox container (``DockerJobExecutor``); its return value becomes the
  ``JobResult``. This is how server-side code scorers ride the RFC job-executor substrate.

Either way the ``exec()`` of the scorer source happens HERE, in the isolated sandbox — never
in the server/worker process.
"""

import inspect
import json
import sys
import traceback
from pathlib import Path
from typing import Any


def _reconstruct_trace(data):
    # Let errors propagate to run_scorer_payload (recorded as a FAILED envelope). Returning None
    # instead would run a trace-dependent scorer with trace=None and yield a wrong result.
    if data is None:
        return None
    from mlflow.entities import Trace

    if isinstance(data, str):
        return Trace.from_json(data)
    return Trace.from_dict(data)


def _serialize_result(result):
    from mlflow.entities import Feedback

    if isinstance(result, Feedback):
        return {"kind": "feedback", "value": _feedback_dict(result)}
    if isinstance(result, list) and result and all(isinstance(r, Feedback) for r in result):
        return {"kind": "feedback_list", "value": [_feedback_dict(r) for r in result]}
    return {"kind": "primitive", "value": result}


def _feedback_dict(fb):
    error = fb.error
    return {
        "name": fb.name,
        "value": fb.value,
        "rationale": fb.rationale,
        # Carry the error so a scorer that reports a computation failure (e.g. an LLM-judge
        # timeout) round-trips as an errored Feedback, not a silently value-less one.
        "error": (
            {
                "error_code": error.error_code,
                "error_message": error.error_message,
                "stack_trace": error.stack_trace,
            }
            if error is not None
            else None
        ),
    }


def run_scorer_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the scorer from its source and run it, returning a result envelope
    (``{"ok": True, "result": {...}}`` or ``{"ok": False, "error", "traceback"}``). A scorer
    that raises is captured in the envelope (not propagated) so both entry points report it
    uniformly rather than crashing the sandbox.
    """
    try:
        from mlflow.genai.scorers.scorer_utils import recreate_function

        fn = recreate_function(payload["source"], payload["signature"], payload["func_name"])
        kwargs = dict(payload.get("kwargs") or {})
        if "trace" in kwargs:
            kwargs["trace"] = _reconstruct_trace(kwargs["trace"])
        if kwargs.get("session"):
            kwargs["session"] = [_reconstruct_trace(t) for t in kwargs["session"]]

        params = set(inspect.signature(fn).parameters)
        filtered = {k: v for k, v in kwargs.items() if k in params}
        return {"ok": True, "result": _serialize_result(fn(**filtered))}
    except Exception as e:
        return {"ok": False, "error": str(e), "traceback": traceback.format_exc()}


def run_scorer_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Job-executor entry point (loaded by fully-qualified name inside the sandbox container).
    Its return value becomes the ``JobResult`` the executor hands back to the server.
    """
    return run_scorer_payload(payload)


def main() -> None:
    workdir = Path(sys.argv[1])
    payload = json.loads((workdir / "payload.json").read_text())
    (workdir / "result.json").write_text(json.dumps(run_scorer_payload(payload)))


if __name__ == "__main__":
    main()
