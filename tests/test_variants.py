from hsb_eeg2text.taxonomy.build import randomize_hierarchy


def test_random_hierarchy_preserves_rows():
    pd = __import__("pandas")
    df = pd.DataFrame(
        [
            {"keyword": "ambulance", "coarse": "object", "mid": "vehicle", "fine": "ambulance"},
            {"keyword": "doctor", "coarse": "person", "mid": "medical", "fine": "doctor"},
        ]
    )
    out = randomize_hierarchy(df, seed=1)
    assert len(out) == len(df)
    assert set(out["fine"]) == set(df["fine"])
