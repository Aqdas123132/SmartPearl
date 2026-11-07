"""Chat-model factory. LLM_PROVIDER=extractive means no LLM: an offline, deterministic baseline used for plumbing tests."""
from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from src.config import get_settings


def get_chat_model() -> BaseChatModel | None:
    s = get_settings()
    if s.llm_provider == "extractive":
        return None
    if s.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=s.llm_model, api_key=s.openai_api_key, temperature=s.temperature, timeout=s.llm_timeout_s, max_retries=0)
    try:
        from langchain_ollama import ChatOllama
    except ImportError as exc:
        raise RuntimeError("LLM_PROVIDER=ollama needs `pip install langchain-ollama`.") from exc
    return ChatOllama(model=s.llm_model, base_url=s.ollama_base_url, temperature=s.temperature)


def transient_errors() -> tuple[type[BaseException], ...]:
    errs: list[type[BaseException]] = [TimeoutError, ConnectionError]
    try:
        import openai
        errs += [openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError, openai.InternalServerError]
    except ImportError:
        pass
    return tuple(errs)


def auth_errors() -> tuple[type[BaseException], ...]:
    try:
        import openai
        return (openai.AuthenticationError, openai.PermissionDeniedError)
    except ImportError:
        return ()
