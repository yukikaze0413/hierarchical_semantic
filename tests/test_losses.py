import pytest


def test_prototype_ce_accepts_duplicate_labels():
    torch = pytest.importorskip("torch")
    from hsb_eeg2text.models.losses import prototype_ce_loss

    eeg = torch.randn(4, 8)
    prototypes = torch.randn(2, 8)
    labels = torch.tensor([0, 0, 1, 1])
    loss = prototype_ce_loss(eeg, prototypes, labels)
    assert torch.isfinite(loss)
