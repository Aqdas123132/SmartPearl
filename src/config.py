"""Central, validated configuration. Every tunable is read from the environment exactly once."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[1]


class ConfigError(RuntimeError):
    """Raised at start-up when required configuration is missing or inconsistent."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore")

    # Providers
    llm_provider: Literal["openai", "ollama", "extractive"] = "extractive"
    llm_model: str = "gpt-4o-mini"
    embedding_provider: Literal["openai", "huggingface", "hash"] = "hash"
    embedding_model: str = "text-embedding-3-small"
    openai_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    temperature: float = 0.0
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 3

    # Chunking
    chunk_strategy: Literal["recursive", "fixed", "structure"] = "structure"
    chunk_size: int = 800
    chunk_overlap: int = 120

    # Retrieval
    retrieval_strategy: Literal["similarity", "mmr", "threshold", "hybrid", "hybrid_rerank"] = "hybrid_rerank"
    retrieval_k: int = 4
    fetch_k: int = 20
    score_threshold: float = 0.29  # calibrated with evaluation/retrieval_experiments.py for the 'hash' embeddings; RECALIBRATE per embedding model
    high_confidence_margin: float = 0.10
    hybrid_weight_dense: float = 0.5
    reranker: Literal["none", "lexical", "cross_encoder"] = "lexical"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_context_chunks: int = 4
    max_context_chars: int = 6000
    history_window_turns: int = 4

    # Paths
    knowledge_base_dir: Path = ROOT / "data" / "knowledge_base"
    upload_dir: Path = ROOT / "data" / "uploads"
    vector_store_path: Path = ROOT / "data" / "vector_store"
    collection_name: str = "pearlz_kb"
    embedding_cache_path: Path = ROOT / "data" / "embedding_cache.sqlite"
    log_dir: Path = ROOT / "logs"
    log_level: str = "INFO"

    # App
    database_url: str = f"sqlite:///{ROOT / 'data' / 'app.db'}"
    secret_key: str = "change-me"
    token_ttl_minutes: int = 480
    rate_limit_per_minute: int = 30
    request_timeout_s: float = 60.0
    max_upload_mb: int = 10
    admin_email: str = ""
    admin_password: str = ""
    support_email: str = "support@pearlzhome.example"
    langsmith_api_key: str = ""

    @model_validator(mode="after")
    def _check(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ConfigError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        needs_key = "openai" in (self.llm_provider, self.embedding_provider)
        if needs_key and not self.openai_api_key:
            raise ConfigError("OPENAI_API_KEY is required when LLM_PROVIDER or EMBEDDING_PROVIDER is 'openai'. "
                              "Set it in your .env file.")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    try:
        return Settings()
    except ConfigError:
        raise
    except Exception as exc:  # pydantic wraps ConfigError in ValidationError
        msg = str(exc).split("\n")
        raise ConfigError(" ".join(m.strip() for m in msg if "Value error" in m or "OPENAI" in m) or str(exc)) from exc


def masked(settings: Settings) -> dict:
    out = settings.model_dump(mode="json")
    for k in ("openai_api_key", "secret_key", "langsmith_api_key"):
        if out.get(k):
            out[k] = out[k][:3] + "***" if len(out[k]) > 6 else "***"
    return out


if __name__ == "__main__":
    import json
    try:
        print(json.dumps(masked(get_settings()), indent=2))
    except ConfigError as e:
        raise SystemExit(f"CONFIG ERROR: {e}")
