import json

from hsb_eeg2text.inference.llm_backends.mock import MockBackend


def test_mock_backend_returns_json_sentence():
    prompt = "1. object > vehicle > ambulance, confidence: 0.900"
    response = MockBackend().generate(prompt)
    parsed = json.loads(response.text)
    assert "sentence" in parsed
    assert "ambulance" in parsed["sentence"]
