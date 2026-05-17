"""Latest-checkpoint loader + single-window prediction interface."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from engine.models.cnn_lstm import CLASSES, build_model

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"


def _remap_native_lstm_to_cellloop(sd: dict) -> dict:
    out: dict = {}
    for k, v in sd.items():
        if ".weight_ih_l0" in k:
            out[k.replace(".weight_ih_l0", ".W_x.weight")] = v
        elif ".weight_hh_l0" in k:
            out[k.replace(".weight_hh_l0", ".W_h.weight")] = v
        elif ".bias_ih_l0" in k:
            base = k.replace(".bias_ih_l0", "")
            hh = sd.get(f"{base}.bias_hh_l0")
            out[f"{base}.W_x.bias"] = v + hh if hh is not None else v
        elif ".bias_hh_l0" in k:
            continue
        else:
            out[k] = v
    return out


@dataclass(frozen=True)
class Prediction:
    label: str          # BUY | SELL | HOLD
    confidence: float   # 0..1
    probs: dict[str, float]


MIN_VALID_CHECKPOINT_BYTES = 1024


def _checkpoint_is_loadable(path: Path) -> bool:
    try:
        if path.stat().st_size < MIN_VALID_CHECKPOINT_BYTES:
            return False
        torch.load(path, map_location="cpu", weights_only=False)
        return True
    except Exception:
        return False


def latest_checkpoint(model_name: str = "cnn_lstm") -> Path | None:
    if not CHECKPOINT_DIR.exists():
        return None
    candidates = sorted(
        CHECKPOINT_DIR.glob(f"{model_name}_v*.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if _checkpoint_is_loadable(candidate):
            return candidate
    return None


class CNNLSTMInferencer:
    def __init__(self, checkpoint: Path | None = None, device: str | None = None) -> None:
        if device == "dml":
            import torch_directml  # noqa: PLC0415
            self.device = torch_directml.device()
        else:
            self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = build_model(self.device)
        self.checkpoint_path = checkpoint or latest_checkpoint()
        if self.checkpoint_path is None:
            raise FileNotFoundError(
                f"No CNN-LSTM checkpoint found in {CHECKPOINT_DIR}; train one first."
            )
        state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        sd = _remap_native_lstm_to_cellloop(state["model_state"])
        self.model.load_state_dict(sd)
        self.model.eval()
        self.feature_mean = np.asarray(state.get("feature_mean", []), dtype=np.float32)
        self.feature_std = np.asarray(state.get("feature_std", []), dtype=np.float32)
        self.version = state.get("version", "unknown")

    def predict(self, window: np.ndarray) -> Prediction:
        """`window` is a (60, 50) ndarray of *raw* features (not yet normalized)."""
        if window.shape != (60, 50):
            raise ValueError(f"expected (60, 50), got {window.shape}")
        x = window.astype(np.float32, copy=False)
        if self.feature_mean.size:
            x = (x - self.feature_mean) / np.where(self.feature_std < 1e-9, 1.0, self.feature_std)
            x = np.clip(x, -10.0, 10.0)
        tensor = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).to(self.device)  # (1,1,60,50)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
        idx = int(probs.argmax())
        return Prediction(
            label=CLASSES[idx],
            confidence=float(probs[idx]),
            probs={c: float(p) for c, p in zip(CLASSES, probs)},
        )
