"""[POC] In-container entry point for ``DockerJobExecutor``.

Invoked inside the sandbox container as::

    python -m mlflow.server.jobs._docker_job_entry <workdir>

Reads ``<workdir>/job.json`` (``fn_fullname`` + ``params``), loads the job function by
its fully-qualified name, runs it, and writes ``<workdir>/result.json`` in ``JobResult``
shape. The job function is imported and executed HERE, inside the isolated container,
never in the server process.

POC note: results are returned through a bind-mounted file. The RFC #3 remote model
instead reports results back to the tracking server over HTTP using the scoped token in
``JobExecutionContext``; that transport depends on the job-store machinery that is still
on the unmerged ``jobs-execution-rfc2`` branch.
"""

import json
import os
import sys
import traceback
import urllib.request
from pathlib import Path
from typing import Any


def _report(workdir: Path, result: dict[str, Any]) -> None:
    """Return the result via bind-mounted file and, if configured, an HTTP callback.

    The callback mirrors the RFC #3 remote model: POST the ``JobResult`` back to the
    tracking server with the scoped ``JobExecutionContext`` token as a bearer credential.
    """
    (workdir / "result.json").write_text(json.dumps(result))

    callback_url = os.environ.get("MLFLOW_JOB_RESULT_CALLBACK_URL")
    if not callback_url:
        return
    headers = {"Content-Type": "application/json"}
    if token := os.environ.get("MLFLOW_JOB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        callback_url, data=json.dumps(result).encode(), headers=headers, method="POST"
    )
    try:
        urllib.request.urlopen(request, timeout=30).close()
    except Exception as e:
        # The file result already stands; a rejected/unreachable callback must not crash
        # the job. The real framework would retry here.
        print(f"result callback to {callback_url} failed: {e}", file=sys.stderr)  # noqa: T201


def main() -> None:
    workdir = Path(sys.argv[1])
    try:
        payload = json.loads((workdir / "job.json").read_text())
        from mlflow.server.jobs.utils import _load_function

        fn = _load_function(payload["fn_fullname"])
        value = fn(**(payload.get("params") or {}))
        result = {
            "status": "SUCCEEDED",
            "result": json.dumps(value),
            "error_message": None,
            "is_transient_error": False,
        }
    except Exception as e:
        result = {
            "status": "FAILED",
            "result": None,
            "error_message": f"{e}\n{traceback.format_exc()}",
            "is_transient_error": False,
        }
    _report(workdir, result)


if __name__ == "__main__":
    main()
