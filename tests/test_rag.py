import json

from hsb_eeg2text.inference.rag import retrieve_examples


def test_rag_examples_are_json_safe_and_exclude_test_sentence():
    index = [
        {
            "sentence_id": "s1",
            "sentence": "The ambulance stopped.",
            "split": "train",
            "anchors": [["object", "vehicle", "ambulance"]],
            "coarse": ["object"],
            "mid": ["vehicle"],
            "fine": ["ambulance"],
        },
        {
            "sentence_id": "s2",
            "sentence": "The gold test sentence.",
            "split": "test",
            "anchors": [["object", "vehicle", "ambulance"]],
            "coarse": ["object"],
            "mid": ["vehicle"],
            "fine": ["ambulance"],
        },
    ]
    anchors = [{"coarse": "object", "mid": "vehicle", "fine": "ambulance"}]
    examples = retrieve_examples(anchors, index, query_sentence_id="s2", allowed_splits={"train"})
    assert [item["sentence_id"] for item in examples] == ["s1"]
    json.dumps(examples)
