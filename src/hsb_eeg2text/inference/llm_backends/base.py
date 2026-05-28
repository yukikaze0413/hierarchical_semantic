from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    raw: dict


class LLMBackend:
    name = "base"

    def generate(self, prompt: str) -> LLMResponse:
        raise NotImplementedError
