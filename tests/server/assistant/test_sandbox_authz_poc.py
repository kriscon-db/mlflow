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
from mlflow.assistant.providers.tool_executor import execute_tool
from mlflow.server.assistant.api import (
    _authorize_experiment_access,
    _authorize_session_access,
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

    monkeypatch.setenv("MLFLOW_ENABLE_ASSISTANT_SANDBOX", "true")
    names_on = {t["function"]["name"] for t in build_tools_schema()}
    assert {"search_traces", "get_trace"} <= names_on

    monkeypatch.setenv("MLFLOW_ENABLE_ASSISTANT_SANDBOX", "false")
    names_off = {t["function"]["name"] for t in build_tools_schema()}
    assert not ({"search_traces", "get_trace"} & names_off)


def test_data_tool_rbac_denies_without_read(monkeypatch):
    # The data tier re-checks the caller's experiment permission; a denied caller gets an
    # error result (not the data), even though the tool routing succeeded.
    import sys
    import types

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
