from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class JudgeBenchSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="JUDGEBENCH_", case_sensitive=False)

    api_base_url: str = "https://api.llm7.io/v1"
    api_key: str | None = None
    model: str = "replace-me"

