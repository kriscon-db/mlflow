"""MLflow AI Gateway preset of the OpenAI-compatible assistant provider."""

import base64
import logging
from typing import ClassVar

from mlflow.assistant.providers.openai_compatible import OpenAICompatibleProvider
from mlflow.environment_variables import _MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN

_logger = logging.getLogger(__name__)


class MlflowGatewayProvider(OpenAICompatibleProvider):
    """OpenAI-compatible provider backed by an in-server MLflow AI Gateway."""

    # Provider name for the in-server MLflow AI Gateway backend. The frontend
    # mirrors this literal in `server/js/src/assistant/constants.ts`
    # (GATEWAY_PROVIDER_ID); keep the two in sync.
    GATEWAY_PROVIDER_NAME: ClassVar[str] = "mlflow_gateway"

    @staticmethod
    def _build_chat_url(_base_url: str | None, tracking_uri: str) -> str | None:
        """The in-server MLflow Gateway is reachable through the same MLflow
        server, so the chat URL is derived from the tracking URI instead of
        a separate base_url stored in config.
        """
        if not tracking_uri:
            return None
        return f"{tracking_uri.rstrip('/')}/gateway/mlflow/v1/chat/completions"

    def __init__(self) -> None:
        super().__init__(
            name=self.GATEWAY_PROVIDER_NAME,
            display_name="MLflow AI Gateway",
            description=(
                "AI-powered assistant backed by an MLflow AI Gateway endpoint "
                "configured on this server."
            ),
            connection_hint=(
                "Configure an LLM chat endpoint on the MLflow AI Gateway and select it."
            ),
            chat_url_builder=self._build_chat_url,
            allows_remote_access=True,
        )

    def _auth_headers(self, api_key: str | None, caller: str | None = None) -> dict[str, str]:
        """Authenticate the server-side call to the in-server AI Gateway.

        When MLflow auth is enabled, the server auto-generates an internal token (shared with
        this process via ``_MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN``) that the auth middleware
        accepts as a Basic-auth password on ``/gateway/`` routes for a given user. We send
        ``Basic base64(caller:token)`` so the request is authorized and attributed to the
        user driving the turn — mirroring how job workers authenticate to the Gateway. Falls
        back to the default (bearer/none) when the token isn't set (e.g. no-auth servers).
        """
        token = _MLFLOW_INTERNAL_GATEWAY_AUTH_TOKEN.get()
        if token and caller:
            creds = base64.b64encode(f"{caller}:{token}".encode()).decode()
            return {"Authorization": f"Basic {creds}"}
        return super()._auth_headers(api_key, caller)

    @staticmethod
    def _list_endpoints():
        from mlflow.tracking._tracking_service.utils import _get_store

        store = _get_store()
        try:
            return store.list_gateway_endpoints()
        except (AttributeError, NotImplementedError):
            return []

    def list_models(self, base_url: str | None = None, api_key: str | None = None) -> list[str]:
        return sorted(endpoint.name for endpoint in self._list_endpoints() if endpoint.name)

    def is_available(self) -> bool:
        try:
            return bool(self.list_models())
        except Exception:
            _logger.debug("Failed to list gateway endpoints", exc_info=True)
            return False
