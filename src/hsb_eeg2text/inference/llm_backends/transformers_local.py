from __future__ import annotations

from hsb_eeg2text.inference.llm_backends.base import LLMBackend, LLMResponse


class TransformersLocalBackend(LLMBackend):
    name = "transformers_local"

    def __init__(self, model_name: str, dtype: str = "bfloat16", load_in_4bit: bool = False, max_tokens: int = 512):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch_dtype = getattr(torch, dtype) if hasattr(torch, dtype) else torch.bfloat16
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto",
            load_in_4bit=load_in_4bit,
            trust_remote_code=True,
        )
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> LLMResponse:
        messages = [
            {"role": "system", "content": "You reconstruct concise English sentences from EEG-derived semantic anchors."},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=self.max_tokens, do_sample=False)
        decoded = self.tokenizer.decode(output[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return LLMResponse(text=decoded, raw={"backend": self.name})
