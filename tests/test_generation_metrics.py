from hsb_eeg2text.evaluation.metrics import reconstruction_metrics, rouge_l, token_f1


def test_token_f1_and_rouge_l():
    assert token_f1("the ambulance stopped", "ambulance stopped") > 0.75
    assert rouge_l("the ambulance stopped", "ambulance stopped") > 0.75


def test_reconstruction_metrics_basic_coverage():
    rows = [
        {
            "gold_sentence": "The ambulance stopped.",
            "generated_sentence": "The ambulance stopped.",
            "anchors": [{"coarse": "object", "mid": "vehicle", "fine": "ambulance"}],
            "retrieved_examples": [{"sentence": "The ambulance stopped."}],
        }
    ]
    metrics = reconstruction_metrics(rows)
    assert metrics["keyword_coverage"] == 1.0
    assert metrics["token_f1"] == 1.0
    assert metrics["retrieval_top5"] == 1.0
