from __future__ import annotations


class HierarchicalSemanticModel:
    def __new__(
        cls,
        eeg_encoder_name: str,
        embedding_dim: int,
        dropout: float,
        label_counts: dict[str, int],
        in_channels: int = 48,
        text_input_dim: int | None = None,
    ):
        import torch.nn as nn
        import torch.nn.functional as F

        from hsb_eeg2text.models.eeg_encoders import build_eeg_encoder

        class _Model(nn.Module):
            def __init__(self):
                super().__init__()
                text_input = text_input_dim or embedding_dim
                self.eeg_encoder = build_eeg_encoder(eeg_encoder_name, embedding_dim, dropout, in_channels)
                self.text_projection = nn.Identity() if text_input == embedding_dim else nn.Linear(text_input, embedding_dim)
                self.coarse_head = nn.Linear(embedding_dim, label_counts["coarse"])
                self.mid_head = nn.Linear(embedding_dim, label_counts["mid"])
                self.fine_head = nn.Linear(embedding_dim, label_counts["fine"])

            def forward(self, eeg):
                emb = F.normalize(self.eeg_encoder(eeg), dim=-1)
                return {
                    "embedding": emb,
                    "coarse_logits": self.coarse_head(emb),
                    "mid_logits": self.mid_head(emb),
                    "fine_logits": self.fine_head(emb),
                }

            def project_text(self, text_emb):
                return F.normalize(self.text_projection(text_emb), dim=-1)

        return _Model()
