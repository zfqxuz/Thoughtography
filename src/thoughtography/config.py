from __future__ import annotations

import os
from dataclasses import dataclass


class ConfigError(RuntimeError):
    pass


@dataclass(slots=True)
class ProviderConfig:
    name: str
    base_url: str
    api_key: str
    model: str
    chat_url: str
    timeout_sec: float = 180.0
    json_mode: bool = True

    def public_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "base_url": self.base_url,
            "model": self.model,
            "chat_url": self.chat_url,
            "timeout_sec": self.timeout_sec,
            "json_mode": self.json_mode,
            "api_key": "***" if self.api_key else "",
        }


def _env_prefix(provider: str) -> str:
    return provider.strip().upper().replace("-", "_")


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value.strip()
    return None


def resolve_chat_url(base_url: str) -> str:
    url = base_url.strip().rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return f"{url}/chat/completions"


def load_provider_config(
    provider: str = "deepseek",
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    timeout_sec: float | None = None,
    json_mode: bool | None = None,
) -> ProviderConfig:
    """Build provider config from CLI overrides and environment variables.

    Supported providers:
    - ``dsh``: DSH_BASE_URL / DSH_API_KEY / DSH_VISION_MODEL
    - ``deepseek``: DEEPSEEK_BASE_URL / DEEPSEEK_API_KEY / DEEPSEEK_VISION_MODEL
    - any other name: <PROVIDER>_BASE_URL / <PROVIDER>_API_KEY / <PROVIDER>_VISION_MODEL

    ``BASE_URL`` should include the API prefix; for example
    ``https://api.deepseek.com`` or ``https://relay.example.com/v1``.
    """
    provider = provider.strip().lower()
    prefix = _env_prefix(provider)

    resolved_base = (
        base_url
        or os.environ.get(f"{prefix}_BASE_URL")
        or (os.environ.get("DEEPSEEK_BASE_URL") if provider == "deepseek" else None)
        or (os.environ.get("DSH_BASE_URL") if provider == "dsh" else None)
    )
    resolved_key = (
        api_key
        or os.environ.get(f"{prefix}_API_KEY")
        or (os.environ.get("DEEPSEEK_API_KEY") if provider == "deepseek" else None)
        or (os.environ.get("DSH_API_KEY") if provider == "dsh" else None)
    )
    resolved_model = (
        model
        or _first_env(
            f"{prefix}_VISION_MODEL",
            f"{prefix}_MODEL",
            "DEEPSEEK_VISION_MODEL" if provider == "deepseek" else "",
            "DEEPSEEK_MODEL" if provider == "deepseek" else "",
            "DSH_VISION_MODEL" if provider == "dsh" else "",
            "DSH_MODEL" if provider == "dsh" else "",
        )
    )

    missing: list[str] = []
    if not resolved_base:
        missing.append(f"{prefix}_BASE_URL")
    if not resolved_key:
        missing.append(f"{prefix}_API_KEY")
    if not resolved_model:
        missing.append(f"{prefix}_VISION_MODEL")
    if missing:
        raise ConfigError(
            "缺少模型配置: " + ", ".join(missing) + "\n"
            "可以在 .env 中设置，或使用 --base-url / --api-key / --model 参数。"
        )

    resolved_timeout = timeout_sec
    if resolved_timeout is None:
        raw_timeout = os.environ.get("VISION_TIMEOUT_SEC")
        resolved_timeout = float(raw_timeout) if raw_timeout else 180.0

    resolved_json_mode = json_mode
    if resolved_json_mode is None:
        raw_json_mode = (os.environ.get("VISION_JSON_MODE") or "true").strip().lower()
        resolved_json_mode = raw_json_mode not in {"0", "false", "no", "off"}

    base = str(resolved_base).strip()
    return ProviderConfig(
        name=provider,
        base_url=base,
        api_key=str(resolved_key),
        model=str(resolved_model),
        chat_url=resolve_chat_url(base),
        timeout_sec=float(resolved_timeout or 180.0),
        json_mode=bool(resolved_json_mode),
    )
