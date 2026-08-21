"""[POC] Demo job functions exercised by ``DockerJobExecutor`` through the
``AbstractJobExecutor`` contract.

These are importable by fully-qualified name inside the sandbox container (the local
mlflow source is bind-mounted), so ``submit_job(fn_fullname=...)`` can load and run them.
"""

import inspect
import os
import time
from typing import Any

from mlflow.entities import Feedback
from mlflow.genai.scorers.scorer_utils import recreate_function


def add(a: int, b: int) -> int:
    return a + b


def sleep_and_return(seconds: float, value: Any) -> Any:
    time.sleep(seconds)
    return value


def hard_exit() -> None:
    # Bypass the entry's try/except so the container exits WITHOUT writing result.json,
    # exercising recover_jobs()'s "exited without a reported result" -> fail path.
    os._exit(1)


def score_from_source(source: str, signature: str, func_name: str, kwargs: dict[str, Any]) -> Any:
    """Track 2 bridge: reconstruct a ``@scorer``'s source and run it inside the container.

    This is the stable framework entry point a custom-scorer job targets. The scorer
    source travels in ``params`` and is ``exec()``-d HERE in the isolated container, never
    in the server process, which is what makes server-side custom scorers safe in OSS.
    """
    fn = recreate_function(source, signature, func_name)
    allowed = set(inspect.signature(fn).parameters)
    filtered = {k: v for k, v in (kwargs or {}).items() if k in allowed}
    result = fn(**filtered)
    if isinstance(result, Feedback):
        return {"name": result.name, "value": result.value, "rationale": result.rationale}
    return result
