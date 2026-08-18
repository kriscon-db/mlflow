"""[POC] Subprocess entry point that reconstructs and runs a ``@scorer`` code scorer.

Invoked by ``mlflow.genai.scorers.sandbox.run_scorer_in_sandbox`` as::

    python -m mlflow.genai.scorers._sandbox_runner <workdir>

Reads ``<workdir>/payload.json`` (source + signature + func_name + call kwargs) and
writes ``<workdir>/result.json``. The ``exec()`` of the scorer source happens HERE, in
the isolated child — never in the server/worker process.
"""

import inspect
import json
import sys
import traceback
from pathlib import Path


def _reconstruct_trace(data):
    # Let reconstruction errors propagate to main(), which records a FAILED result. Silently
    # returning None would run a trace-dependent scorer with trace=None and yield a wrong result.
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
    return {"name": fb.name, "value": fb.value, "rationale": getattr(fb, "rationale", None)}


def main() -> None:
    workdir = Path(sys.argv[1])
    out = workdir / "result.json"
    try:
        payload = json.loads((workdir / "payload.json").read_text())
        from mlflow.genai.scorers.scorer_utils import recreate_function

        fn = recreate_function(
            payload["source"], payload["signature"], payload["func_name"]
        )
        kwargs = dict(payload.get("kwargs") or {})
        if "trace" in kwargs:
            kwargs["trace"] = _reconstruct_trace(kwargs["trace"])
        if kwargs.get("session"):
            kwargs["session"] = [_reconstruct_trace(t) for t in kwargs["session"]]

        params = set(inspect.signature(fn).parameters)
        filtered = {k: v for k, v in kwargs.items() if k in params}
        result = fn(**filtered)
        out.write_text(json.dumps({"ok": True, "result": _serialize_result(result)}))
    except Exception as e:
        out.write_text(
            json.dumps({"ok": False, "error": str(e), "traceback": traceback.format_exc()})
        )


if __name__ == "__main__":
    main()
