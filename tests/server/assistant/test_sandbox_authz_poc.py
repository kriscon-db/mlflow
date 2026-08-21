# [POC] Tests for the Assistant sandbox entry gate and tool routing (Docker-free): caller
# identity resolution (including the fix that a remote client cannot spoof identity when auth is
# off), session-ownership enforcement, experiment RBAC, the compute-tool -> sandbox routing
# decision, and tool mapping.

import asyncio
import base64
import sys
import types
from unittest import mock

import pytest
from fastapi import HTTPException

from mlflow.assistant.config import PermissionsConfig
from mlflow.assistant.providers import (
    ClaudeCodeProvider,
    CodexProvider,
    MlflowGatewayProvider,
    OllamaProvider,
    OpenAICompatibleProvider,
)
from mlflow.assistant.providers.tool_executor import execute_tool
from mlflow.server.assistant.api import (
    _authorize_experiment_access,
    _authorize_session_access,
    _enforce_remote_access,
    _resolve_caller_identity,
)
from mlflow.server.assistant.sandbox.integration import _to_tool_call
from mlflow.server.assistant.session import Session


def _request(host=None, headers=None, username=None):
    request = types.SimpleNamespace()
    request.client = types.SimpleNamespace(host=host) if host else None
    request.headers = headers or {}
    request.state = types.SimpleNamespace()
    if username is not None:
        request.state.username = username
    return request


def test_authenticated_principal_wins_over_headers():
    request = _request("10.0.0.9", {"x-mlflow-user": "hdr"}, username="dave")
    assert _resolve_caller_identity(request) == "dave"


def test_localhost_no_auth_is_local():
    assert _resolve_caller_identity(_request("127.0.0.1")) == "local"


def test_localhost_honors_forwarded_user_header():
    assert _resolve_caller_identity(_request("127.0.0.1", {"x-mlflow-user": "bob"})) == "bob"


def test_localhost_honors_basic_auth():
    basic = "Basic " + base64.b64encode(b"alice:pw").decode()
    assert _resolve_caller_identity(_request("127.0.0.1", {"authorization": basic})) == "alice"


def test_remote_ignores_spoofable_identity_headers():
    # Security fix: a remote client must not be able to impersonate a user via
    # x-mlflow-user / Basic when MLflow auth is not active.
    basic = "Basic " + base64.b64encode(b"victim:pw").decode()
    request = _request("10.0.0.9", {"x-mlflow-user": "victim", "authorization": basic})
    assert _resolve_caller_identity(request) == "remote-anonymous"


def test_session_owner_can_access():
    _authorize_session_access(Session(owner="alice"), "alice")


def test_session_non_owner_denied():
    with pytest.raises(HTTPException, match="do not have access") as exc:
        _authorize_session_access(Session(owner="alice"), "bob")
    assert exc.value.status_code == 403


def test_legacy_ownerless_session_not_enforced():
    _authorize_session_access(Session(owner=None), "anyone")


def test_experiment_rbac_noop_without_experiment():
    assert _authorize_experiment_access("alice", None) is None


def test_experiment_rbac_failsafe_when_auth_plugin_absent():
    # mlflow[auth] isn't installed in the base test env -> ImportError -> no enforcement.
    assert _authorize_experiment_access("alice", "1") is None


def test_experiment_rbac_allow_and_deny():
    fake_auth = types.ModuleType("mlflow.server.auth")
    fake_auth.is_auth_enabled = lambda: True
    fake_auth._get_experiment_permission = lambda eid, user: types.SimpleNamespace(
        can_read=(user == "alice")
    )
    with mock.patch.dict(sys.modules, {"mlflow.server.auth": fake_auth}):
        _authorize_experiment_access("alice", "123")  # has read -> no raise
        with pytest.raises(HTTPException, match="do not have read access") as exc:
            _authorize_experiment_access("mallory", "123")
    assert exc.value.status_code == 403


@pytest.mark.parametrize(
    ("tool_name", "tool_input", "expected_tool"),
    [
        ("Bash", {"command": "ls"}, "bash"),
        ("Read", {"file_path": "a.txt"}, "read"),
        ("Write", {"file_path": "a.txt", "content": "c"}, "write"),
        ("Edit", {"file_path": "a.txt", "old_string": "o", "new_string": "n"}, "edit"),
    ],
)
def test_to_tool_call_maps_known_tools(tool_name, tool_input, expected_tool):
    assert _to_tool_call(tool_name, tool_input).tool == expected_tool


def test_to_tool_call_returns_none_for_unknown():
    assert _to_tool_call("RenderCustomView", {}) is None


def test_execute_tool_routes_to_sandbox_with_flag_and_session(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_ASSISTANT_SANDBOX", "true")
    with mock.patch(
        "mlflow.server.assistant.sandbox.integration.run_sandboxed_tool",
        return_value=("sandbox-output", False),
    ) as mock_run:
        result = asyncio.run(execute_tool("Bash", {"command": "ls"}, session_id="s1"))
    assert result == ("sandbox-output", False)
    mock_run.assert_called_once()


def test_execute_tool_stays_in_process_without_session(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_ASSISTANT_SANDBOX", "true")
    with mock.patch("mlflow.server.assistant.sandbox.integration.run_sandboxed_tool") as mock_run:
        # No session_id -> must not route to the sandbox even with the flag on.
        asyncio.run(
            execute_tool(
                "Bash", {"command": "echo hi"}, permissions=PermissionsConfig(full_access=True)
            )
        )
    mock_run.assert_not_called()


def test_gateway_provider_authenticates_to_in_server_gateway_with_internal_token(monkeypatch):
    # On an auth-enabled server the Assistant's server-side call to the in-server AI Gateway
    # must send Basic (caller, internal_token) so it authenticates AND attributes to the user.
    from mlflow.assistant.providers.mlflow_gateway import MlflowGatewayProvider

    monkeypatch.setenv("_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN", "sekret-token")
    headers = MlflowGatewayProvider()._auth_headers(api_key=None, caller="alice")
    assert headers["Authorization"] == "Basic " + base64.b64encode(b"alice:sekret-token").decode()


def test_gateway_provider_no_auth_header_without_token(monkeypatch):
    # No internal token (e.g. a no-auth server) -> fall back to the default (no header when
    # there's no api_key), rather than sending a broken credential.
    from mlflow.assistant.providers.mlflow_gateway import MlflowGatewayProvider

    monkeypatch.delenv("_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN", raising=False)
    assert MlflowGatewayProvider()._auth_headers(api_key=None, caller="alice") == {}


def test_execute_tool_routes_data_tools_to_server_tier(monkeypatch):
    # Data tools (search_traces/get_trace) run server-side under the caller's RBAC, NOT in the
    # sandbox. execute_tool must dispatch them to run_data_tool with the caller identity.
    monkeypatch.setenv("MLFLOW_ENABLE_ASSISTANT_SANDBOX", "true")
    with mock.patch(
        "mlflow.server.assistant.sandbox.data_tools.run_data_tool",
        return_value=("{}", False),
    ) as mock_data:
        asyncio.run(
            execute_tool("search_traces", {"experiment_id": "1"}, session_id="s1", caller="alice")
        )
    mock_data.assert_called_once()
    # caller + session_id + tool name are forwarded so RBAC + materialization can happen.
    args = mock_data.call_args.args
    assert args[0] == "alice"
    assert args[1] == "s1"
    assert args[2] == "search_traces"


def test_data_tools_advertised_only_when_sandbox_enabled(monkeypatch):
    from mlflow.assistant.providers.tool_executor import build_tools_schema

    data_tools = {"search_traces", "get_trace", "log_feedback"}
    monkeypatch.setenv("MLFLOW_ENABLE_ASSISTANT_SANDBOX", "true")
    names_on = {t["function"]["name"] for t in build_tools_schema()}
    assert data_tools <= names_on

    monkeypatch.setenv("MLFLOW_ENABLE_ASSISTANT_SANDBOX", "false")
    names_off = {t["function"]["name"] for t in build_tools_schema()}
    assert not (data_tools & names_off)


def test_gateway_family_is_the_sandbox_capable_family():
    # Only OpenAICompatibleProvider calls execute_tool (the sandbox routing point). The
    # gateway/server-loop providers subclass it, so every gateway-backed model — OpenAI,
    # Anthropic, and Gemini alike — routes its compute tools through the sandbox. There is no
    # per-model sandbox work: the coverage is structural, via this inheritance.
    assert issubclass(MlflowGatewayProvider, OpenAICompatibleProvider)
    assert issubclass(OllamaProvider, OpenAICompatibleProvider)


@pytest.mark.parametrize(
    ("provider_cls", "expected_remote"),
    [
        (ClaudeCodeProvider, False),
        (CodexProvider, False),
        (MlflowGatewayProvider, True),
    ],
)
def test_cli_providers_are_localhost_only(provider_cls, expected_remote):
    # Security boundary as a regression test: the CLI providers execute tools in a host process
    # (not the sandbox), so they must stay localhost-only. If someone flips one to remote, this
    # fails loudly — because a remote-reachable CLI provider would expose host exec to other users.
    assert provider_cls().allows_remote_access is expected_remote


def test_remote_request_with_localhost_only_provider_is_blocked():
    remote_request = _request("10.0.0.9")
    with pytest.raises(HTTPException, match="only accessible") as exc:
        _enforce_remote_access(remote_request, ClaudeCodeProvider())
    assert exc.value.status_code == 403
    # The same provider from localhost is allowed (the gate keys on client IP, not the provider).
    _enforce_remote_access(_request("127.0.0.1"), ClaudeCodeProvider())


def test_remote_access_requires_auth_enabled(monkeypatch):
    # A remote-capable provider still must not serve remote clients when auth is off: without a
    # verified identity, RBAC would fail open. Localhost stays exempt.
    from mlflow.server.assistant import api

    monkeypatch.setenv("MLFLOW_ENABLE_REMOTE_ASSISTANT", "true")
    remote = _request("10.0.0.9")
    gateway = MlflowGatewayProvider()

    # auth disabled (base test env has no auth plugin -> _auth_enabled() is False) -> refused
    with pytest.raises(HTTPException, match="requires authentication") as exc:
        _enforce_remote_access(remote, gateway)
    assert exc.value.status_code == 403

    # auth enabled -> the same remote request is allowed
    with mock.patch.object(api, "_auth_enabled", return_value=True):
        _enforce_remote_access(remote, gateway)

    # localhost is unaffected even with auth off
    _enforce_remote_access(_request("127.0.0.1"), gateway)


def test_data_tool_rbac_denies_without_read(monkeypatch):
    # The data tier re-checks the caller's experiment permission; a denied caller gets an
    # error result (not the data), even though the tool routing succeeded.
    from mlflow.server.assistant.sandbox import data_tools

    fake_auth = types.ModuleType("mlflow.server.auth")
    fake_auth.is_auth_enabled = lambda: True
    fake_auth._get_experiment_permission = lambda eid, user: types.SimpleNamespace(can_read=False)
    with mock.patch.dict(sys.modules, {"mlflow.server.auth": fake_auth}):
        out, is_error = data_tools.run_data_tool(
            "mallory", "s1", "search_traces", {"experiment_id": "7"}
        )
    assert is_error
    assert "Permission denied" in out


def test_log_feedback_writes_when_caller_has_write_access():
    from mlflow.server.assistant.sandbox import data_tools

    fake_store = mock.Mock()
    fake_store.get_trace_info.return_value = types.SimpleNamespace(experiment_id="7")
    fake_store.create_assessment.return_value = types.SimpleNamespace(assessment_id="a-1")
    with (
        mock.patch.object(data_tools, "_get_store", return_value=fake_store),
        mock.patch.object(data_tools, "_has_experiment_permission", return_value=True) as perm,
    ):
        out, is_error = data_tools.run_data_tool(
            "alice", "s1", "log_feedback", {"trace_id": "tr-1", "value": True, "name": "relevance"}
        )
    assert not is_error
    assert '"logged": true' in out
    # Write tools must gate on write (can_update), not read.
    assert perm.call_args.args == ("7", "alice", "can_update")
    feedback = fake_store.create_assessment.call_args.args[0]
    assert feedback.trace_id == "tr-1"
    assert feedback.metadata == {"logged_by": "alice"}


def test_log_feedback_denied_without_write_access():
    from mlflow.server.assistant.sandbox import data_tools

    fake_store = mock.Mock()
    fake_store.get_trace_info.return_value = types.SimpleNamespace(experiment_id="7")
    with (
        mock.patch.object(data_tools, "_get_store", return_value=fake_store),
        mock.patch.object(data_tools, "_has_experiment_permission", return_value=False),
    ):
        out, is_error = data_tools.run_data_tool(
            "mallory", "s1", "log_feedback", {"trace_id": "tr-1", "value": True}
        )
    assert is_error
    assert "Permission denied" in out
    fake_store.create_assessment.assert_not_called()


def test_get_trace_fails_closed_when_trace_has_no_experiment():
    # A trace with no experiment_id must be DENIED, not returned unchecked: _can_read_experiment
    # returns False for a falsy experiment_id, so get_trace never runs. (Regression for the
    # earlier fail-open bug where `if experiment_id and not _can_read_experiment(...)` skipped
    # the check entirely.)
    from mlflow.server.assistant.sandbox import data_tools

    fake_store = mock.Mock()
    fake_store.get_trace_info.return_value = types.SimpleNamespace(experiment_id=None)
    with mock.patch.object(data_tools, "_get_store", return_value=fake_store):
        out, is_error = data_tools.run_data_tool("alice", "s1", "get_trace", {"trace_id": "tr-1"})
    assert is_error
    assert "Permission denied" in out
    fake_store.get_trace.assert_not_called()


def test_log_feedback_requires_value():
    from mlflow.server.assistant.sandbox import data_tools

    out, is_error = data_tools.run_data_tool("alice", "s1", "log_feedback", {"trace_id": "tr-1"})
    assert is_error
    assert "requires a value" in out
