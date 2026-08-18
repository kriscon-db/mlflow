"""[POC] Session executor contract for the sandboxed, multi-user MLflow Assistant.

This is the Track 1 sibling of the job-executor framework (``AbstractJobExecutor``): it
runs the Assistant's untrusted *compute* tools (bash/read/write/edit) inside a per-session
sandbox instead of the server process. It rhymes with the job executor (same container
primitive, same Topology B security: scoped token, Gateway-only LLM, no secrets in the
sandbox) but has a long-running, interactive lifecycle rather than run-to-completion:

  - job executor:     submit_job -> wait_for_job          (one shot)
  - session executor: start_session -> exec_in_session*   (persistent, many turns) -> stop

Data/privileged tools (fetch a trace, log feedback, call the model) are NOT part of this
contract: the server executes those under the calling user's RBAC. Only isolated compute
crosses into the sandbox, which is why compute tools can be auto-approved.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class SessionContext:
    """Per-session execution metadata handed to the backend at ``start_session``."""

    session_id: str
    tracking_uri: str
    gateway_uri: str | None = None
    token: str | None = None
    workspace: str | None = None
    # Calling user's identity, resolved by the server's entry-gate authz. Carried so the
    # sandbox and its resources can be attributed/scoped to one user.
    owner: str | None = None


@dataclass
class ToolCall:
    """A compute-tool invocation routed into the sandbox."""

    tool: Literal["bash", "read", "write", "edit"]
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolResult:
    output: str
    is_error: bool = False


@dataclass
class SessionRecoveryResult:
    session_id: str
    action: Literal["reattach", "requeue", "fail"]
    error_message: str | None = None


class AbstractSessionExecutor(ABC):
    """Backend contract for per-session Assistant compute sandboxes."""

    def start_executor(self) -> None:
        """Called on server startup: acquire resources, warm the pool."""

    def stop_executor(self) -> None:
        """Called on server shutdown: tear down all sandboxes and the pool."""

    @abstractmethod
    def start_session(self, context: SessionContext) -> None:
        """Assign an isolated sandbox to the session (ideally from a warm pool)."""

    @abstractmethod
    def exec_in_session(self, session_id: str, call: ToolCall) -> ToolResult:
        """Run one compute tool inside the session's persistent sandbox."""

    @abstractmethod
    def stop_session(self, session_id: str) -> None:
        """Destroy the session's sandbox. Sandboxes are never recycled across sessions."""

    @abstractmethod
    def recover_sessions(self, unfinished_session_ids: list[str]) -> list[SessionRecoveryResult]:
        """Determine recovery action for sessions whose server process disappeared."""

    @property
    def remote_execution(self) -> bool:
        return False

    def check_requirements(self) -> None:
        """Optional fail-fast validation run during server startup."""
