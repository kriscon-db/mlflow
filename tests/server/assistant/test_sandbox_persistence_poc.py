# [POC] Docker-free tests for the session->container persistence + restart reconciliation:
# the sandbox-binding sidecar round-trips (separate from the Session record so a mid-turn write
# isn't clobbered by the end-of-turn session save), the binding map skips session/PID files,
# _recover_bindings scopes to this node, and the executor's recover/reap logic reattaches live
# containers, tombstones gone ones, and reaps orphans.

import json
import types
from unittest import mock

import pytest

from mlflow.exceptions import MlflowException
from mlflow.server.assistant.sandbox import integration
from mlflow.server.assistant.sandbox.docker_session_executor import (
    _LABEL_NODE,
    _LABEL_POOL,
    DockerSessionExecutor,
    get_node_id,
)
from mlflow.server.assistant.sandbox.session_executor import SessionContext
from mlflow.server.assistant.session import (
    clear_container_binding,
    list_container_bindings,
    save_container_binding,
)


def test_container_binding_sidecar_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr("mlflow.server.assistant.session.SESSION_DIR", tmp_path)
    session_id = "11111111-1111-1111-1111-111111111111"

    save_container_binding(session_id, "c0ffee", "host-1")
    assert list_container_bindings() == {session_id: ("c0ffee", "host-1")}

    clear_container_binding(session_id)
    assert list_container_bindings() == {}


def test_list_container_bindings_reads_only_container_sidecars(tmp_path, monkeypatch):
    monkeypatch.setattr("mlflow.server.assistant.session.SESSION_DIR", tmp_path)
    bound = "11111111-1111-1111-1111-111111111111"
    save_container_binding(bound, "abc123", "host-1")
    # A session record and a PID sidecar in the same directory must be ignored.
    (tmp_path / f"{bound}.json").write_text(json.dumps({"owner": "alice"}))
    (tmp_path / f"{bound}.process.json").write_text(json.dumps({"pid": 999}))

    assert list_container_bindings() == {bound: ("abc123", "host-1")}


def test_recover_bindings_scopes_to_this_node():
    with mock.patch.object(
        integration,
        "list_container_bindings",
        return_value={
            "here": ("cid-here", get_node_id()),
            "elsewhere": ("cid-elsewhere", "some-other-host"),
        },
    ):
        assert integration._recover_bindings() == {"here": "cid-here"}


def test_recover_sessions_reattaches_live_and_tombstones_gone():
    executor = DockerSessionExecutor()
    alive = {"cid-live"}
    with mock.patch.object(
        DockerSessionExecutor, "_container_alive", side_effect=lambda cid: cid in alive
    ):
        results = executor.recover_sessions({"s-live": "cid-live", "s-gone": "cid-gone"})

    actions = {r.session_id: r.action for r in results}
    assert actions == {"s-live": "reattach", "s-gone": "fail"}
    # Live session is truly reattached (usable by exec_in_session); gone session is tombstoned
    # so its next turn receives the reap/resume notice.
    assert executor.is_active("s-live")
    assert executor.get_container_id("s-live") == "cid-live"
    assert not executor.is_active("s-gone")
    assert executor.consume_reap_notice("s-gone") is not None


def test_reap_orphans_destroys_only_untracked_labeled_containers():
    # A scorer-pool executor must reap only its OWN pool's containers, never the assistant
    # pool's — the two share a process and image, so reaping is scoped by the pool label.
    executor = DockerSessionExecutor(pool="scorer")
    executor._active = {"s1": "keep-active"}
    executor._idle.append("keep-idle")

    containers = [
        types.SimpleNamespace(id="keep-active", short_id="keep-active"[:12]),
        types.SimpleNamespace(id="keep-idle", short_id="keep-idle"[:12]),
        types.SimpleNamespace(id="orphan-1", short_id="orphan-1"),
        types.SimpleNamespace(id="orphan-2", short_id="orphan-2"),
    ]
    fake_client = mock.Mock()
    fake_client.containers.list.return_value = containers

    with (
        mock.patch.object(DockerSessionExecutor, "_get_client", return_value=fake_client),
        mock.patch.object(DockerSessionExecutor, "_destroy") as mock_destroy,
    ):
        executor._reap_orphans()

    # Reap is scoped to BOTH this pool AND this node, so a replica never reaps another
    # replica's containers on a shared Docker daemon.
    fake_client.containers.list.assert_called_once_with(
        all=True, filters={"label": [f"{_LABEL_POOL}=scorer", f"{_LABEL_NODE}={get_node_id()}"]}
    )
    destroyed = {c.args[0] for c in mock_destroy.call_args_list}
    assert destroyed == {"orphan-1", "orphan-2"}


def test_start_session_enforces_max_total():
    # The cold path must not spawn past max_total — the cap is a hard ceiling, not advisory.
    executor = DockerSessionExecutor(max_total=1)
    executor._active = {"s0": "cid0"}  # at capacity, idle pool empty
    with pytest.raises(MlflowException, match="capacity reached"):
        executor.start_session(SessionContext(session_id="s1", tracking_uri=""))
