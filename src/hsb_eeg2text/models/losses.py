from __future__ import annotations


def symmetric_info_nce(eeg_emb, text_emb, temperature: float = 0.07):
    import torch
    import torch.nn.functional as F

    eeg_emb = F.normalize(eeg_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    logits = eeg_emb @ text_emb.T / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def prototype_ce_loss(eeg_emb, prototype_emb, labels, temperature: float = 0.07):
    import torch.nn.functional as F

    eeg_emb = F.normalize(eeg_emb, dim=-1)
    prototype_emb = F.normalize(prototype_emb, dim=-1)
    logits = eeg_emb @ prototype_emb.T / temperature
    return F.cross_entropy(logits, labels)


def hierarchy_consistency_loss(outputs: dict, batch: dict):
    import torch.nn.functional as F

    return F.cross_entropy(outputs["coarse_logits"], batch["coarse_id"]) + F.cross_entropy(outputs["mid_logits"], batch["mid_id"])
