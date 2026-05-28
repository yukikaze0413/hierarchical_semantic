from hsb_eeg2text.inference.reconstruct import parse_generation


def test_parse_direct_json():
    parsed = parse_generation('{"entities": ["ambulance"], "action": "stopped", "sentence": "The ambulance stopped."}')
    assert parsed["sentence"] == "The ambulance stopped."


def test_parse_fenced_json():
    parsed = parse_generation('```json\n{"sentence": "The car waited.", "entities": ["car"], "action": "waited"}\n```')
    assert parsed["sentence"] == "The car waited."


def test_parse_wrapped_json():
    parsed = parse_generation('Here is the result: {"sentence": "The train moved.", "entities": ["train"], "action": "moved"} Thanks.')
    assert parsed["sentence"] == "The train moved."
