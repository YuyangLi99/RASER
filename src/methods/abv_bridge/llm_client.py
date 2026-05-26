"""Shared LLM client for ABV-Bridge modules."""

from openai import OpenAI

from src.tools.llm_utils import call_chat, is_reasoning_model

_DEFAULT_API_KEY = ""
_DEFAULT_BASE_URL = "https://ki-toolbox.scc.kit.edu/api/v1"
_DEFAULT_MODEL = "kit.gpt-oss-120b"


class LLMClient:
    """Thin wrapper around KIT API with reasoning-model handling."""

    def __init__(self, api_key: str = None, model: str = None):
        import os as _os
        self.client = OpenAI(
            api_key=(api_key or _os.environ.get("HAGRID_LLM_API_KEY")
                     or _DEFAULT_API_KEY),
            base_url=(_os.environ.get("HAGRID_LLM_BASE_URL") or _DEFAULT_BASE_URL),
        )
        self.model = model or _DEFAULT_MODEL
        self.is_reasoning = is_reasoning_model(self.model)
        self.total_tokens = 0

    def call(self, prompt: str, max_tokens: int = 1500,
             system: str = None, temperature: float = 0.0) -> tuple:
        """Returns (content_str, token_count). Never raises."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        content, tokens = call_chat(self.client, self.model, messages,
                                    max_tokens=max_tokens,
                                    temperature=temperature)
        self.total_tokens += tokens
        return content, tokens
