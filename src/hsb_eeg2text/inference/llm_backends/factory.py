from __future__ import annotations

from hsb_eeg2text.config import deep_get
from hsb_eeg2text.inference.llm_backends.mock import MockBackend
from hsb_eeg2text.inference.llm_backends.openai_compatible import OpenAICompatibleBackend
from hsb_eeg2text.inference.llm_backends.transformers_local import TransformersLocalBackend


def build_llm_backend(config: dict, backend_name: str | None = None):
    backend_name = backend_name or deep_get(config, "llm.default_backend", "deepseek_api")
    backend_cfg = deep_get(config, f"llm.backends.{backend_name}")
    if not backend_cfg:
        raise ValueError(f"Unknown LLM backend config: {backend_name}")
    backend_type = backend_cfg["type"]
    if backend_type == "mock":
        return MockBackend()
    if backend_type == "openai_compatible":
        import os

        base_url = backend_cfg.get("base_url") or os.getenv(backend_cfg.get("base_url_env", ""))
        model = backend_cfg.get("model") or os.getenv(backend_cfg.get("model_env", ""))
        if not base_url or not model:
            raise RuntimeError(f"Backend {backend_name} requires base_url/model or matching environment variables.")
        return OpenAICompatibleBackend(
            base_url=base_url,
            api_key_env=backend_cfg["api_key_env"],
            model=model,
            temperature=float(deep_get(config, "llm.temperature", 0.2)),
            max_tokens=int(deep_get(config, "llm.max_tokens", 512)),
        )
    if backend_type == "transformers_local":
        return TransformersLocalBackend(
            model_name=backend_cfg["model_name"],
            dtype=backend_cfg.get("dtype", "bfloat16"),
            load_in_4bit=bool(backend_cfg.get("load_in_4bit", False)),
            max_tokens=int(deep_get(config, "llm.max_tokens", 512)),
        )
    raise ValueError(f"Unknown backend type: {backend_type}")
