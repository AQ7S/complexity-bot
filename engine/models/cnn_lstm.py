"""CNN-LSTM architecture per Appendix F.

Input: (batch, 1, 60, 50)  →  output logits over 3 classes (BUY, SELL, HOLD).

Note: Uses LSTMCell loops instead of nn.LSTM to avoid fused-LSTM kernels
that DirectML does not implement (aten::_thnn_fused_lstm_cell).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

CLASSES = ("BUY", "SELL", "HOLD")
N_CLASSES = 3
SEQUENCE_LEN = 60
N_FEATURES = 50


class _LSTMCellLoop(nn.Module):
    """Drop-in replacement for ``nn.LSTM(batch_first=True, num_layers=1)``
    implemented with raw linear layers + sigmoid/tanh so it never touches
    ``aten::_thnn_fused_lstm_cell`` (absent on DirectML **and** broken for
    CPU-fallback under torch-directml).
    """

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        # Combined gates: input, forget, cell-candidate, output  (4 * hidden)
        self.W_x = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size, bias=False)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # x: (B, T, input_size)
        b, t, _ = x.shape
        h = torch.zeros(b, self.hidden_size, device=x.device, dtype=x.dtype)
        c = torch.zeros_like(h)
        outputs: list[torch.Tensor] = []
        for step in range(t):
            gates = self.W_x(x[:, step, :]) + self.W_h(h)  # (B, 4*H)
            i, f, g, o = gates.chunk(4, dim=1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            g = torch.tanh(g)
            o = torch.sigmoid(o)
            c = f * c + i * g
            h = o * torch.tanh(c)
            outputs.append(h)
        return torch.stack(outputs, dim=1), (h, c)


class CNNLSTM(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES, dropout: float = 0.3) -> None:
        super().__init__()
        self.act = nn.LeakyReLU(0.01)

        # Conv block 1
        self.c1 = nn.Conv2d(1,   32, kernel_size=3, padding=1)
        self.c2 = nn.Conv2d(32,  32, kernel_size=3, padding=1)
        self.p1 = nn.MaxPool2d(2)

        # Conv block 2
        self.c3 = nn.Conv2d(32,  64, kernel_size=3, padding=1)
        self.c4 = nn.Conv2d(64,  64, kernel_size=3, padding=1)
        self.p2 = nn.MaxPool2d(2)

        # Conv block 3
        self.c5 = nn.Conv2d(64,  128, kernel_size=3, padding=1)
        self.c6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.c7 = nn.Conv2d(128, 64,  kernel_size=1)

        # After two pools on (60, 50): height = 60/4 = 15, width = 50/4 = 12.
        # Using LSTMCell loop for DirectML compatibility.
        self.lstm1 = _LSTMCellLoop(input_size=64 * 12, hidden_size=256)
        self.dropout_lstm = nn.Dropout(0.2)
        self.lstm2 = _LSTMCellLoop(input_size=256, hidden_size=256)

        self.fc1 = nn.Linear(256, 64)
        self.dropout_fc = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1
        x = self.act(self.c1(x))
        x = self.act(self.c2(x))
        x = self.p1(x)
        # Conv block 2
        x = self.act(self.c3(x))
        x = self.act(self.c4(x))
        x = self.p2(x)
        # Conv block 3
        x = self.act(self.c5(x))
        x = self.act(self.c6(x))
        x = self.act(self.c7(x))  # (B, 64, 15, 12)

        # Flatten spatial-width into feature dim → (B, 15, 64*12)
        b, c, h, w = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, h, c * w)

        x, _ = self.lstm1(x)
        x = self.dropout_lstm(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]  # last time step

        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)
        return self.fc2(x)


def build_model(device: str | torch.device | None = None) -> CNNLSTM:
    model = CNNLSTM()
    if device is not None:
        model = model.to(device)
    return model
