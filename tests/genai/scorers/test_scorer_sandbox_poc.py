# [POC] Tests for server-side (sandboxed) custom code scorers, covering the Docker-free,
# security-critical logic: the OSS registration gate, that a reloaded decorator scorer becomes
# a sandboxed scorer that never exec()s in-process, and a round-trip through the subprocess
# provider.

from unittest import mock

import pytest

from mlflow.entities import Feedback
from mlflow.exceptions import MlflowException
from mlflow.genai.scorers import scorer
from mlflow.genai.scorers.base import Scorer, _SandboxedDecoratorScorer


@pytest.fixture
def sandbox_enabled(monkeypatch):
    monkeypatch.setenv("MLFLOW_ENABLE_SERVER_SIDE_CODE_SCORERS", "true")
    monkeypatch.setenv("MLFLOW_SCORER_SANDBOX_PROVIDER", "subprocess")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)


def test_decorator_scorer_blocked_in_oss_without_flag(monkeypatch):
    monkeypatch.delenv("MLFLOW_ENABLE_SERVER_SIDE_CODE_SCORERS", raising=False)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

    @scorer
    def s(outputs) -> bool:
        return True

    with pytest.raises(MlflowException, match="not supported outside of Databricks"):
        s._check_can_be_registered()


def test_flag_lifts_gate_and_reload_is_sandboxed(sandbox_enabled):
    @scorer
    def has_citation(outputs) -> bool:
        return "[source:" in outputs

    has_citation._check_can_be_registered()  # must not raise with the flag on
    loaded = Scorer.model_validate(has_citation.model_dump())
    assert isinstance(loaded, _SandboxedDecoratorScorer)


def test_sandboxed_scorer_dispatches_to_sandbox_not_in_process(sandbox_enabled):
    @scorer
    def s(outputs) -> bool:
        return True

    loaded = Scorer.model_validate(s.model_dump())
    # The reconstructed scorer must run its source in the sandbox, never exec it in-process.
    with mock.patch(
        "mlflow.genai.scorers.sandbox.run_scorer_in_sandbox", return_value=True
    ) as mock_run:
        loaded.run(outputs="x")
    mock_run.assert_called_once()


def test_subprocess_provider_roundtrip_bool(sandbox_enabled):
    @scorer
    def has_citation(outputs) -> bool:
        return "[source:" in outputs

    loaded = Scorer.model_validate(has_citation.model_dump())
    assert loaded.run(outputs="answer [source: kb#1]") is True
    assert loaded.run(outputs="no citation") is False


def test_subprocess_provider_roundtrip_feedback(sandbox_enabled):
    @scorer
    def citation_feedback(outputs) -> Feedback:
        ok = "[source:" in outputs
        return Feedback(value=ok, rationale="cited" if ok else "no citation")

    loaded = Scorer.model_validate(citation_feedback.model_dump())
    result = loaded.run(outputs="answer [source: kb#1]")
    assert isinstance(result, Feedback)
    assert result.value is True
    assert result.rationale == "cited"
