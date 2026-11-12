"""Provider gọi mô hình chạy local qua Ollama (`POST /api/chat`).

Lưu ý quyền riêng tư: module này cố ý KHÔNG dùng `logging` và không đưa nội dung
prompt/response vào thông điệp lỗi, vì prompt có thể chứa dữ liệu sức khỏe.
"""

from __future__ import annotations

from types import TracebackType
from typing import Any
from ai_healthcare.providers.base import GenerationConfig
import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "qwen3:8b"
NO_THINK_SUFFIX = "/no_think"


class OllamaProviderError(Exception):
    """Lớp cơ sở cho mọi lỗi của OllamaProvider."""


class OllamaConnectionError(OllamaProviderError):
    """Không kết nối được tới Ollama (chưa chạy, sai endpoint, ...)."""


class OllamaTimeoutError(OllamaProviderError):
    """Ollama không phản hồi trong thời gian cho phép."""


class OllamaHTTPError(OllamaProviderError):
    """Ollama trả về HTTP status lỗi (>= 400)."""

    def __init__(self, status_code: int, detail: str = "") -> None:
        self.status_code = status_code
        message = f"Ollama returned HTTP {status_code}"
        if detail:
            message = f"{message}: {detail}"
        super().__init__(message)


class OllamaResponseError(OllamaProviderError):
    """Response không đúng cấu trúc mong đợi (không phải JSON, thiếu message.content, ...)."""


class OllamaProvider:
    """Client đồng bộ cho Ollama với Structured Outputs.

    Client httpx được tạo một lần và tái sử dụng; gọi `close()` hoặc dùng
    như context manager để giải phóng. Nếu truyền `client` từ ngoài (ví dụ
    trong test với `httpx.MockTransport`), provider KHÔNG đóng client đó.
    """

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        temperature: float = 0,
        num_ctx: int = 16384,
        num_predict: int = 512,
        timeout: float = 120.0,
        no_think: bool = True,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.timeout = timeout
        self.no_think = no_think
        self._owns_client = client is None
        self._client = client if client is not None else httpx.Client(timeout=timeout)

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/api/chat"

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        json_schema: dict[str, Any],
        *,
        config: GenerationConfig | None = None,
    ) -> str:
        """Gửi prompt, ràng buộc output theo `json_schema`, trả về chuỗi do model sinh ra.

        Hàm này KHÔNG parse hay validate JSON; việc đó thuộc về tầng gọi.

        Raises:
            OllamaTimeoutError, OllamaConnectionError, OllamaHTTPError, OllamaResponseError
        """
        system_content = (
            f"{system_prompt}\n\n{NO_THINK_SUFFIX}" if self.no_think else system_prompt
        )
        if config is not None and not isinstance(config, GenerationConfig):
            raise TypeError("config must be a GenerationConfig or None.")

        options = {
            "temperature": (
                self.temperature
                if config is None or config.temperature is None
                else config.temperature
            ),
            "num_ctx": (
                self.num_ctx
                if config is None or config.num_ctx is None
                else config.num_ctx
            ),
            "num_predict": (
                self.num_predict
                if config is None or config.num_predict is None
                else config.num_predict
            ),
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "format": json_schema,
            "think": False,
            "options": options,
        }

        try:
            response = self._client.post(self.chat_url, json=payload)
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError(
                f"Ollama did not respond within {self.timeout}s"
            ) from exc
        except httpx.TransportError as exc:
            raise OllamaConnectionError(
                f"Cannot reach Ollama at {self.base_url}: {type(exc).__name__}"
            ) from exc

        if response.status_code >= 400:
            raise OllamaHTTPError(response.status_code, self._error_detail(response))

        return self._extract_content(response)

    def close(self) -> None:
        """Đóng HTTP client nếu provider sở hữu nó."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> OllamaProvider:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        """Lấy trường `error` của Ollama (nếu có); không dump toàn bộ body."""
        try:
            body = response.json()
        except ValueError:
            return ""
        if isinstance(body, dict) and isinstance(body.get("error"), str):
            return body["error"][:200]
        return ""

    @staticmethod
    def _extract_content(response: httpx.Response) -> str:
        try:
            data = response.json()
        except ValueError as exc:
            raise OllamaResponseError("Ollama response is not valid JSON") from exc
        if not isinstance(data, dict):
            raise OllamaResponseError("Ollama response is not a JSON object")
        message = data.get("message")
        if not isinstance(message, dict):
            raise OllamaResponseError("Ollama response is missing 'message'")
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise OllamaResponseError("Ollama response is missing 'message.content'")
        return content
