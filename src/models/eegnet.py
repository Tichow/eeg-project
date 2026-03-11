import torch
import torch.nn as nn


class ChannelDropout(nn.Module):
    """Randomly zero out entire channels during training.

    This forces the model to learn robust features that don't rely on
    all channels being present — critical for testing on an 8-channel
    OpenBCI setup when trained on 64-channel PhysioNet data.
    """

    def __init__(self, p: float = 0.1):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, 1, n_channels, n_times)
        if not self.training or self.p == 0:
            return x
        mask = torch.bernoulli(
            torch.full((x.size(0), 1, x.size(2), 1), 1 - self.p, device=x.device)
        )
        return x * mask / (1 - self.p)


class EEGNet(nn.Module):
    """EEGNet-8,2 architecture (Lawhern et al., 2018) in PyTorch.

    Parameters
    ----------
    n_channels : number of EEG channels (64 for PhysioNet)
    n_times : number of time samples per epoch (560 for 3.5s at 160 Hz)
    n_classes : number of output classes (2 for left/right, 4 for multi)
    sfreq : sampling frequency in Hz (used to compute temporal kernel size)
    F1 : number of temporal filters
    D : depth multiplier for depthwise convolution
    F2 : number of separable filters
    dropout_rate : dropout probability
    channel_dropout : probability of masking each channel (0 to disable)
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_times: int = 560,
        n_classes: int = 2,
        sfreq: float = 160.0,
        F1: int = 16,
        D: int = 4,
        F2: int = 64,
        dropout_rate: float = 0.0,
        channel_dropout: float = 0.1,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_times = n_times
        self.n_classes = n_classes

        kernel_length = int(sfreq * 0.5)  # 0.5 seconds

        self.channel_drop = ChannelDropout(p=channel_dropout)

        # Block 1: Temporal + Spatial filtering
        self.block1 = nn.Sequential(
            # Temporal convolution
            nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            # Depthwise spatial convolution
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
        )

        # Block 2: Separable convolution
        self.block2 = nn.Sequential(
            # Depthwise temporal
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            # Pointwise
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )

        # Compute flattened size by running a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            dummy = self.block1(dummy)
            dummy = self.block2(dummy)
            flat_size = dummy.numel()

        self.classifier = nn.Linear(flat_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : (batch, 1, n_channels, n_times)

        Returns
        -------
        logits : (batch, n_classes)
        """
        x = self.channel_drop(x)
        x = self.block1(x)
        x = self.block2(x)
        x = x.flatten(1)
        return self.classifier(x)

    def freeze_feature_extractor(self):
        """Freeze block1 and block2 for fine-tuning (only classifier trains)."""
        for param in self.block1.parameters():
            param.requires_grad = False
        for param in self.block2.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters."""
        for param in self.parameters():
            param.requires_grad = True
