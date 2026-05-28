from pathlib import Path


def test_feature_field_matching_prefers_mean_diff():
    from hsb_eeg2text.preprocessing.zuco_mat import find_feature_field

    fields = ["mean_t1_diff", "FFD_t1_diff", "mean_g2_diff"]
    assert find_feature_field(fields, "t1") == "mean_t1_diff"
    assert find_feature_field(fields, "g2") == "mean_g2_diff"


def test_stack_frequency_features_shape():
    np = __import__("pytest").importorskip("numpy")
    from hsb_eeg2text.preprocessing.zuco_mat import stack_frequency_features

    word = {f"mean_{suffix}_diff": np.arange(48, dtype=np.float32) for suffix in ["t1", "t2", "a1", "a2", "b1", "b2", "g1", "g2"]}
    eeg, matched = stack_frequency_features(word, ["t1", "t2", "a1", "a2", "b1", "b2", "g1", "g2"])
    assert eeg.shape == (48, 8, 1)
    assert matched["t1"] == "mean_t1_diff"


def test_train_zscore_fills_nan_and_standardizes():
    np = __import__("pytest").importorskip("numpy")
    from hsb_eeg2text.preprocessing.zuco_mat import ParsedWord, apply_train_zscore

    eeg_train = np.ones((48, 8, 1), dtype=np.float32)
    eeg_val = np.full((48, 8, 1), np.nan, dtype=np.float32)
    parsed = [
        ParsedWord("S", "s1", 0, "word", "word", "word", "train", eeg_train),
        ParsedWord("S", "s2", 0, "word", "word", "word", "val", eeg_val),
    ]
    out, stats = apply_train_zscore(parsed)
    assert stats["applied"]
    assert np.isfinite(out[0].eeg).all()
    assert np.isfinite(out[1].eeg).all()


def test_extract_subject_id_from_results_file():
    from hsb_eeg2text.preprocessing.zuco_mat import extract_subject_id

    assert extract_subject_id(Path("resultsZMG_SR.mat")) == "ZMG"


def test_load_mat_any_reports_both_loader_errors(tmp_path: Path):
    pytest = __import__("pytest")
    pytest.importorskip("numpy")
    from hsb_eeg2text.preprocessing.zuco_mat import load_mat_any

    bad_mat = tmp_path / "bad.mat"
    bad_mat.write_text("not a mat file", encoding="utf-8")
    try:
        load_mat_any(bad_mat)
    except RuntimeError as exc:
        text = str(exc)
        assert "h5py_error" in text
        assert "scipy_error" in text
    else:
        raise AssertionError("Expected loader failure")


def test_empty_mat_parse_fails_with_report(tmp_path: Path):
    pytest = __import__("pytest")
    pytest.importorskip("numpy")
    from hsb_eeg2text.preprocessing.zuco_mat import preprocess_zuco_mat

    cfg = {
        "paths": {
            "raw_zuco_dir": str(tmp_path / "raw"),
            "processed_zuco_dir": str(tmp_path / "processed"),
            "eeg_word_dir": str(tmp_path / "processed" / "eeg_word"),
            "reports_dir": str(tmp_path / "reports"),
        },
        "preprocessing": {"zuco_frequency_suffixes": ["t1", "t2", "a1", "a2", "b1", "b2", "g1", "g2"]},
        "data": {"train_ratio": 0.8, "val_ratio": 0.1},
    }
    (tmp_path / "raw").mkdir()
    with pytest.raises(RuntimeError):
        preprocess_zuco_mat(cfg)
    assert (tmp_path / "reports" / "zuco_mat_structure.json").exists()
