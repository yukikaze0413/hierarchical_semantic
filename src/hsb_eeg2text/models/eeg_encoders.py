from __future__ import annotations


def build_eeg_encoder(name: str, embedding_dim: int = 256, dropout: float = 0.2, in_channels: int = 48):
    name = name.lower()
    if name == "deep4":
        return Deep4StyleCNN(in_channels=in_channels, embedding_dim=embedding_dim, dropout=dropout)
    if name == "eegnet":
        return EEGNetEncoder(in_channels=in_channels, embedding_dim=embedding_dim, dropout=dropout)
    if name in {"conformer", "eeg_conformer"}:
        return EEGConformerEncoder(in_channels=in_channels, embedding_dim=embedding_dim, dropout=dropout)
    raise ValueError(f"Unknown EEG encoder: {name}")


class Deep4StyleCNN:
    def __new__(cls, *args, **kwargs):
        import torch.nn as nn

        class _Deep4(nn.Module):
            def __init__(self, in_channels: int, embedding_dim: int, dropout: float):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ELU(),
                    nn.Dropout(dropout),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ELU(),
                    nn.Dropout(dropout),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(128, embedding_dim),
                )

            def forward(self, x):
                return self.net(x)

        return _Deep4(*args, **kwargs)


class EEGNetEncoder:
    def __new__(cls, *args, **kwargs):
        import torch.nn as nn

        class _EEGNet(nn.Module):
            def __init__(self, in_channels: int, embedding_dim: int, dropout: float):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(in_channels, 32, kernel_size=(1, 3), padding=(0, 1), bias=False),
                    nn.BatchNorm2d(32),
                    nn.ELU(),
                    nn.Conv2d(32, 64, kernel_size=(3, 1), padding=(1, 0), groups=32, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ELU(),
                    nn.Dropout(dropout),
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Linear(64, embedding_dim),
                )

            def forward(self, x):
                return self.net(x)

        return _EEGNet(*args, **kwargs)


class EEGConformerEncoder:
    def __new__(cls, *args, **kwargs):
        import torch
        import torch.nn as nn

        class _Conformer(nn.Module):
            def __init__(self, in_channels: int, embedding_dim: int, dropout: float):
                super().__init__()
                self.patch = nn.Conv2d(in_channels, embedding_dim, kernel_size=1)
                layer = nn.TransformerEncoderLayer(
                    d_model=embedding_dim,
                    nhead=4,
                    dim_feedforward=embedding_dim * 2,
                    dropout=dropout,
                    batch_first=True,
                )
                self.encoder = nn.TransformerEncoder(layer, num_layers=2)
                self.out = nn.Linear(embedding_dim, embedding_dim)

            def forward(self, x):
                z = self.patch(x)
                z = z.flatten(2).transpose(1, 2)
                z = self.encoder(z)
                return self.out(torch.mean(z, dim=1))

        return _Conformer(*args, **kwargs)
