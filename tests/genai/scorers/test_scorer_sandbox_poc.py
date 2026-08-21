# [POC] Tests for server-side (sandboxed) custom code scorers, covering the Docker-free,
# security-critical logic: the OSS registration gate, that a reloaded decorator scorer becomes
# a sandboxed scorer that never exec()s in-process, and a round-trip through the subprocess
# provider.

import json
import types
import uuid
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
    # Pin what actually crosses the sandbox boundary (not just that it was called): the scorer's
    # name and source must be forwarded, else a wrong-payload regression would pass silently.
    assert mock_run.call_args.kwargs["func_name"] == "s"
    assert "return True" in mock_run.call_args.kwargs["source"]


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


def test_subprocess_env_allowlist_hides_secrets(sandbox_enabled, monkeypatch):
    # A credential-bearing var in the parent env must NOT reach the sandboxed scorer: the
    # subprocess env is built from an allow-list, so arbitrary/secret names are dropped.
    monkeypatch.setenv("SNEAKY_SECRET", "leak-me")

    @scorer
    def reads_env(outputs) -> str:
        import os  # clint: disable=lazy-import  (scorer body; runs in the sandbox, not here)

        return os.environ.get("SNEAKY_SECRET", "<absent>")

    loaded = Scorer.model_validate(reads_env.model_dump())
    assert loaded.run(outputs="x") == "<absent>"


def test_subprocess_provider_roundtrips_feedback_error(sandbox_enabled):
    from mlflow.entities import AssessmentError

    @scorer
    def erroring(outputs) -> Feedback:
        from mlflow.entities import AssessmentError, Feedback

        return Feedback(error=AssessmentError(error_code="JUDGE_TIMEOUT", error_message="slow"))

    loaded = Scorer.model_validate(erroring.model_dump())
    result = loaded.run(outputs="x")
    assert isinstance(result, Feedback)
    assert isinstance(result.error, AssessmentError)
    assert result.error.error_code == "JUDGE_TIMEOUT"
    assert result.error.error_message == "slow"


def test_subprocess_provider_refused_on_auth_server(sandbox_enabled):
    # subprocess doesn't confine fs/network, so it must be refused on an auth-enabled
    # (multi-user) server — the operator has to use the docker provider there.
    from mlflow.genai.scorers import sandbox as sandbox_mod

    @scorer
    def s(outputs) -> bool:
        return True

    loaded = Scorer.model_validate(s.model_dump())
    with mock.patch.object(sandbox_mod, "_auth_enabled", return_value=True):
        with pytest.raises(MlflowException, match="not permitted on an auth-enabled server"):
            loaded.run(outputs="x")


def test_docker_provider_runs_scorer_as_a_job(sandbox_enabled, monkeypatch):
    # The `docker` provider runs the scorer AS a job on DockerJobExecutor (container = sandbox).
    # This mocks the executor (simulating the in-container run_scorer_job by running the payload
    # locally) and asserts the scorer is dispatched to the job entry point and its result
    # round-trips through the JobResult envelope.
    monkeypatch.setenv("MLFLOW_SCORER_SANDBOX_PROVIDER", "docker")

    from mlflow.entities._job_status import JobStatus
    from mlflow.genai.scorers import sandbox as sandbox_mod
    from mlflow.genai.scorers._sandbox_runner import run_scorer_payload
    from mlflow.server.jobs.executor import JobResult

    @scorer
    def has_citation(outputs) -> bool:
        return "[source:" in outputs

    captured = {}

    class FakeJobExecutor:
        def submit_job(self, *, job_id, job_name, fn_fullname, params, context, timeout=None):
            captured["fn_fullname"] = fn_fullname
            captured["payload"] = params["payload"]

        def wait_for_job(self, job_id):
            envelope = run_scorer_payload(captured["payload"])
            return JobResult(status=JobStatus.SUCCEEDED, result=json.dumps(envelope))

    monkeypatch.setattr(sandbox_mod, "_get_scorer_job_executor", lambda: FakeJobExecutor())

    loaded = Scorer.model_validate(has_citation.model_dump())
    assert loaded.run(outputs="answer [source: kb#1]") is True
    assert loaded.run(outputs="no citation") is False
    assert captured["fn_fullname"] == "mlflow.genai.scorers._sandbox_runner.run_scorer_job"


def test_online_scheduler_path_sandboxes_decorator_scorer(sandbox_enabled):
    # The online-scoring worker deserializes scorers in OnlineScorerSampler and invokes them
    # via the eval harness (TraceProcessor._execute_scoring -> _compute_eval_scores). This
    # drives that actual scheduler path (not Scorer.run() directly) and asserts the stored
    # source is executed in the sandbox, never exec()'d in the worker process.
    from mlflow.genai.evaluation.harness import _compute_eval_scores
    from mlflow.genai.scorers.online.entities import OnlineScorer, OnlineScoringConfig
    from mlflow.genai.scorers.online.sampler import OnlineScorerSampler

    @scorer
    def has_citation(outputs) -> bool:
        return "[source:" in outputs

    online = OnlineScorer(
        name=has_citation.name,
        serialized_scorer=json.dumps(has_citation.model_dump()),
        online_config=OnlineScoringConfig(
            online_scoring_config_id=uuid.uuid4().hex,
            scorer_id=uuid.uuid4().hex,
            sample_rate=1.0,
            experiment_id="exp1",
        ),
    )
    sampler = OnlineScorerSampler([online])
    sampled = [
        s for group in sampler.group_scorers_by_filter(session_level=False).values() for s in group
    ]
    assert sampled
    assert all(isinstance(s, _SandboxedDecoratorScorer) for s in sampled)

    eval_item = types.SimpleNamespace(
        inputs=None, outputs="answer [source: kb#1]", expectations=None, trace=None
    )
    with mock.patch(
        "mlflow.genai.scorers.sandbox.run_scorer_in_sandbox", return_value=True
    ) as mock_run:
        _compute_eval_scores(eval_item=eval_item, scorers=sampled)
    mock_run.assert_called_once()
    assert mock_run.call_args.kwargs["func_name"] == "has_citation"
