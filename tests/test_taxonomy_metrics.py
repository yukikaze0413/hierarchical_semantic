from hsb_eeg2text.evaluation.metrics import decode_metrics
from hsb_eeg2text.taxonomy.tree import hierarchical_distance, lca_depth


def test_hierarchical_distance_rewards_near_miss():
    gold = ["object", "vehicle", "ambulance"]
    near = ["object", "vehicle", "bus"]
    far = ["emotion", "sadness", "grief"]
    assert lca_depth(gold, near) == 2
    assert hierarchical_distance(gold, near) < hierarchical_distance(gold, far)


def test_decode_metrics_topk():
    rows = [
        {
            "gold": {"coarse": "object", "mid": "vehicle", "fine": "ambulance"},
            "predictions": [
                {"coarse": "object", "mid": "vehicle", "fine": "bus"},
                {"coarse": "object", "mid": "vehicle", "fine": "ambulance"},
            ],
        }
    ]
    metrics = decode_metrics(rows)
    assert metrics["coarse_top1"] == 1.0
    assert metrics["fine_top1"] == 0.0
    assert metrics["fine_top5"] == 1.0
