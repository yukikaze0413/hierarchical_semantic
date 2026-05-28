from __future__ import annotations

import re
import json

from hsb_eeg2text.inference.llm_backends.base import LLMBackend, LLMResponse


class MockBackend(LLMBackend):
    name = "mock"

    def generate(self, prompt: str) -> LLMResponse:
        anchors = re.findall(r"> ([a-zA-Z_]+), confidence", prompt)
        if anchors:
            sentence = " ".join(anchors[:8]).replace("_", " ") + "."
        else:
            sentence = "The reconstructed sentence follows the decoded semantic anchors."
        text = json.dumps({"entities": [], "action": "", "sentence": sentence})
        return LLMResponse(text=text, raw={"backend": self.name})
