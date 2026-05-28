from __future__ import annotations

import os

from hsb_eeg2text.inference.llm_backends.base import LLMBackend, LLMResponse


class OpenAICompatibleBackend(LLMBackend):
    name = "openai_compatible"

    def __init__(self, base_url: str, api_key_env: str, model: str, temperature: float = 0.2, max_tokens: int = 512):
        from openai import OpenAI

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"Missing API key environment variable: {api_key_env}")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str) -> LLMResponse:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You reconstruct concise English sentences from EEG-derived semantic anchors."},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return LLMResponse(text=response.choices[0].message.content or "", raw=response.model_dump())
