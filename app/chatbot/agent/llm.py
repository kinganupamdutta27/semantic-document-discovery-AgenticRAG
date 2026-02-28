"""LLM and embedding model initialisation with multi-provider support.

Supports two modes controlled via the admin panel toggle:

- **Local mode** (default): Uses Ollama with ``.env`` settings.
- **Remote mode**: Uses the provider configured in the admin panel
  (OpenAI, Anthropic, Google GenAI, Groq, Mistral, or even remote Ollama).

Models are lazily initialised and cached.  A lightweight config-hash
check on every ``get_model()`` / ``get_embeddings()`` call detects
admin-panel changes and transparently reinitialises the underlying
client — no server restart required.
"""

import hashlib
import logging
import threading
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings

from app.core.config import settings

logger = logging.getLogger(__name__)

_lock = threading.Lock()

_current_model: Optional[BaseChatModel] = None
_current_embeddings: Optional[Embeddings] = None
_model_config_hash: str = ""
_embeddings_config_hash: str = ""

# Providers supported by init_chat_model
LLM_PROVIDERS = {
    "ollama": "Ollama (Local)",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google-genai": "Google GenAI",
    "groq": "Groq",
    "mistralai": "Mistral AI",
}

EMBEDDING_PROVIDERS = {
    "ollama": "Ollama (Local)",
    "openai": "OpenAI",
    "google-genai": "Google GenAI",
    "huggingface": "HuggingFace (Local)",
}

_LLM_DEFAULTS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-sonnet-20241022",
    "google-genai": "gemini-1.5-flash",
    "groq": "llama-3.1-70b-versatile",
    "mistralai": "mistral-large-latest",
    "ollama": "llama3.1:8b",
}

_EMBEDDING_DEFAULTS = {
    "openai": "text-embedding-3-small",
    "ollama": "nomic-embed-text:latest",
    "google-genai": "models/embedding-001",
    "huggingface": "sentence-transformers/all-MiniLM-L6-v2",
}

# ---------------------------------------------------------------------------
# Config hash helpers (detect admin-panel changes without polling DB)
# ---------------------------------------------------------------------------


def _rc():
    """Late import to avoid circular imports at module load time."""
    from app.admin.runtime_config import rc
    return rc


def _llm_config_hash() -> str:
    """Hash only the settings that actually change LLM behaviour."""
    rc = _rc()
    use_remote = rc.get("use_remote_models", "false")
    if use_remote.lower() == "true":
        parts = [
            "remote",
            rc.get("llm_provider", ""),
            rc.get("llm_model_name", ""),
            rc.get("llm_api_key", ""),
            rc.get("llm_temperature", ""),
        ]
    else:
        parts = [
            "local",
            rc.get("llm_model_name", ""),
            rc.get("llm_base_url", ""),
        ]
    return hashlib.md5("|".join(parts).encode()).hexdigest()


def _emb_config_hash() -> str:
    """Hash only the settings that actually change embedding behaviour."""
    rc = _rc()
    use_remote = rc.get("use_remote_models", "false")
    if use_remote.lower() == "true":
        parts = [
            "remote",
            rc.get("embedding_provider", ""),
            rc.get("embedding_model", ""),
            rc.get("embedding_api_key", ""),
            rc.get("embedding_dimensions", ""),
        ]
    else:
        parts = [
            "local",
            rc.get("embedding_model", ""),
            rc.get("llm_base_url", ""),
        ]
    return hashlib.md5("|".join(parts).encode()).hexdigest()


# ---------------------------------------------------------------------------
# Decryption helper
# ---------------------------------------------------------------------------


def _decrypt_key(setting_key: str) -> str:
    """Read an API key from runtime config and decrypt if encrypted."""
    raw = _rc().get(setting_key, "")
    if not raw:
        return ""
    try:
        from app.filesource.crypto import decrypt
        decrypted = decrypt(raw)
        return decrypted if decrypted else raw
    except Exception:
        return raw


# ---------------------------------------------------------------------------
# LLM builders
# ---------------------------------------------------------------------------


def _build_local_llm() -> BaseChatModel:
    model_name = settings.LLM_MODEL_NAME
    base_url = settings.LLM_BASE_URL
    try:
        rc = _rc()
        model_name = rc.get("llm_model_name", "") or model_name
        base_url = rc.get("llm_base_url", "") or base_url
    except Exception:
        pass
    logger.info("Initialising local Ollama LLM: model=%s url=%s", model_name, base_url)
    return ChatOllama(
        model=model_name,
        base_url=base_url,
        timeout=settings.LLM_TIMEOUT_SECONDS,
    )


def _build_remote_llm() -> BaseChatModel:
    from langchain.chat_models import init_chat_model

    rc = _rc()
    provider = rc.get("llm_provider", "openai")
    model_name = rc.get("llm_model_name", "") or _LLM_DEFAULTS.get(provider, "gpt-4o-mini")
    api_key = _decrypt_key("llm_api_key")
    base_url = rc.get("llm_base_url", "")
    temperature_str = rc.get("llm_temperature", "")

    kwargs: dict = {"timeout": settings.LLM_TIMEOUT_SECONDS}

    if api_key:
        kwargs["api_key"] = api_key
    if base_url and provider == "ollama":
        kwargs["base_url"] = base_url
    if temperature_str:
        try:
            kwargs["temperature"] = float(temperature_str)
        except ValueError:
            pass

    logger.info("Initialising remote LLM: provider=%s model=%s", provider, model_name)
    return init_chat_model(model_name, model_provider=provider, **kwargs)


# ---------------------------------------------------------------------------
# Embedding builders
# ---------------------------------------------------------------------------


def _build_local_embeddings() -> Embeddings:
    model_name = settings.EMBEDDING_MODEL
    base_url = settings.LLM_BASE_URL
    try:
        rc = _rc()
        model_name = rc.get("embedding_model", "") or model_name
        base_url = rc.get("llm_base_url", "") or base_url
    except Exception:
        pass
    logger.info("Initialising local Ollama embeddings: model=%s url=%s", model_name, base_url)
    return OllamaEmbeddings(
        model=model_name,
        base_url=base_url,
    )


def _build_remote_embeddings() -> Embeddings:
    rc = _rc()
    provider = rc.get("embedding_provider", "openai")
    model_name = (
        rc.get("embedding_model", "")
        or _EMBEDDING_DEFAULTS.get(provider, "text-embedding-3-small")
    )
    api_key = _decrypt_key("embedding_api_key") or _decrypt_key("llm_api_key")
    base_url = rc.get("llm_base_url", "")
    dims_str = rc.get("embedding_dimensions", "")

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        kw: dict = {"model": model_name}
        if api_key:
            kw["api_key"] = api_key
        if dims_str:
            try:
                kw["dimensions"] = int(dims_str)
            except ValueError:
                pass
        logger.info("Initialising OpenAI embeddings: %s", model_name)
        return OpenAIEmbeddings(**kw)

    if provider == "ollama":
        kw = {"model": model_name}
        if base_url:
            kw["base_url"] = base_url
        logger.info("Initialising remote Ollama embeddings: %s", model_name)
        return OllamaEmbeddings(**kw)

    if provider == "google-genai":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        kw = {"model": model_name}
        if api_key:
            kw["google_api_key"] = api_key
        logger.info("Initialising Google GenAI embeddings: %s", model_name)
        return GoogleGenerativeAIEmbeddings(**kw)

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        logger.info("Initialising HuggingFace embeddings: %s", model_name)
        return HuggingFaceEmbeddings(model_name=model_name)

    logger.warning("Unknown embedding provider '%s' — falling back to local Ollama", provider)
    return _build_local_embeddings()


# ---------------------------------------------------------------------------
# Public getters (thread-safe, cached, auto-reinitialise on config change)
# ---------------------------------------------------------------------------


def get_model() -> BaseChatModel:
    """Return the active LLM instance, reinitialising when config changes."""
    global _current_model, _model_config_hash

    with _lock:
        try:
            new_hash = _llm_config_hash()
        except Exception:
            if _current_model is None:
                _current_model = _build_local_llm()
            return _current_model

        if _current_model is not None and new_hash == _model_config_hash:
            return _current_model

        use_remote = _rc().get("use_remote_models", "false").lower() == "true"

        try:
            _current_model = _build_remote_llm() if use_remote else _build_local_llm()
            _model_config_hash = new_hash
        except Exception as exc:
            logger.error(
                "Failed to initialise %s LLM: %s — falling back to local Ollama",
                "remote" if use_remote else "local",
                exc,
                exc_info=True,
            )
            if _current_model is None:
                _current_model = _build_local_llm()
                _model_config_hash = new_hash

        return _current_model


def get_embeddings() -> Embeddings:
    """Return the active embedding model, reinitialising when config changes."""
    global _current_embeddings, _embeddings_config_hash

    with _lock:
        try:
            new_hash = _emb_config_hash()
        except Exception:
            if _current_embeddings is None:
                _current_embeddings = _build_local_embeddings()
            return _current_embeddings

        if _current_embeddings is not None and new_hash == _embeddings_config_hash:
            return _current_embeddings

        use_remote = _rc().get("use_remote_models", "false").lower() == "true"

        try:
            _current_embeddings = (
                _build_remote_embeddings() if use_remote else _build_local_embeddings()
            )
            _embeddings_config_hash = new_hash
        except Exception as exc:
            logger.error(
                "Failed to initialise %s embeddings: %s — falling back to local Ollama",
                "remote" if use_remote else "local",
                exc,
                exc_info=True,
            )
            if _current_embeddings is None:
                _current_embeddings = _build_local_embeddings()
                _embeddings_config_hash = new_hash

        return _current_embeddings


def force_reinit() -> None:
    """Force reinitialisation on next access (called after admin saves settings)."""
    global _model_config_hash, _embeddings_config_hash
    with _lock:
        _model_config_hash = ""
        _embeddings_config_hash = ""
    logger.info("Model config hashes cleared — will reinitialise on next access")


def get_embedding_dimensions(embeddings: Embeddings = None) -> int:
    """Detect the output dimension of an embedding model by embedding a test string.

    If no *embeddings* object is passed, uses the currently active one.
    Falls back to ``settings.DINMS`` on any error.
    """
    if embeddings is None:
        embeddings = get_embeddings()
    try:
        vec = embeddings.embed_query("dimension detection probe")
        dims = len(vec)
        logger.info("Detected embedding dimensions: %d", dims)
        return dims
    except Exception as exc:
        logger.warning("Failed to detect embedding dims (%s) — using DINMS=%d", exc, settings.DINMS)
        return settings.DINMS


def get_active_config() -> dict:
    """Return a summary of the currently active model configuration."""
    rc = _rc()
    use_remote = rc.get("use_remote_models", "false").lower() == "true"

    if use_remote:
        return {
            "mode": "remote",
            "llm_provider": rc.get("llm_provider", "openai"),
            "llm_model": rc.get("llm_model_name", "") or _LLM_DEFAULTS.get(rc.get("llm_provider", "openai"), ""),
            "embedding_provider": rc.get("embedding_provider", "openai"),
            "embedding_model": rc.get("embedding_model", "") or _EMBEDDING_DEFAULTS.get(rc.get("embedding_provider", "openai"), ""),
        }

    return {
        "mode": "local",
        "llm_provider": "ollama",
        "llm_model": rc.get("llm_model_name", "") or settings.LLM_MODEL_NAME,
        "embedding_provider": "ollama",
        "embedding_model": rc.get("embedding_model", "") or settings.EMBEDDING_MODEL,
    }
