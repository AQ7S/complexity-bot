"""Generate the 8 bundled WAV files described in Appendix K.

All files are 16-bit PCM, 44.1 kHz, mono, soft-peak (≤ -3 dBFS), and short
(≤ 2s). No external assets — pure numpy + scipy.io.wavfile so the build is
deterministic and license-free.

Re-run safely: existing files are overwritten.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io.wavfile as wav

SR = 44100
OUT_DIR = Path(__file__).resolve().parents[1] / "engine" / "sounds"


def _normalize(sig: np.ndarray, peak_db: float = -3.0) -> np.ndarray:
    peak = 10 ** (peak_db / 20)
    m = float(np.max(np.abs(sig))) or 1.0
    return sig / m * peak


def tone(freq: float, dur_s: float, env: str = "adsr",
         shape: str = "sine") -> np.ndarray:
    n = int(SR * dur_s)
    t = np.arange(n) / SR
    if shape == "sine":
        sig = np.sin(2 * np.pi * freq * t)
    elif shape == "square":
        sig = np.sign(np.sin(2 * np.pi * freq * t))
    else:
        raise ValueError(shape)

    if env == "adsr":
        a = max(int(0.05 * n), 1)
        d = max(int(0.10 * n), 1)
        r = max(int(0.20 * n), 1)
        s = max(n - a - d - r, 1)
        s_lvl = 0.7
        envv = np.concatenate([
            np.linspace(0, 1, a),
            np.linspace(1, s_lvl, d),
            np.full(s, s_lvl),
            np.linspace(s_lvl, 0, r),
        ])
        envv = envv[:n]
        sig = sig[:len(envv)] * envv
    elif env == "exp":
        sig *= np.exp(-3 * t / dur_s)
    elif env == "sharp":
        sig *= np.exp(-12 * t / dur_s)
    elif env == "flat":
        pass
    else:
        raise ValueError(env)
    return sig


def silence(dur_s: float) -> np.ndarray:
    return np.zeros(int(SR * dur_s), dtype=np.float64)


def write(path: Path, sig: np.ndarray) -> None:
    sig = _normalize(sig)
    pcm = (sig * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    wav.write(str(path), SR, pcm)


def make_trading_open() -> np.ndarray:
    return np.concatenate([
        tone(600, 0.080, env="adsr"),
        silence(0.040),
        tone(900, 0.080, env="adsr"),
    ])


def make_profit() -> np.ndarray:
    return np.concatenate([
        tone(523, 0.100, env="adsr"),  # C5
        tone(659, 0.100, env="adsr"),  # E5
        tone(784, 0.100, env="adsr"),  # G5
    ])


def make_loss() -> np.ndarray:
    return np.concatenate([
        tone(392, 0.150, env="adsr"),  # G4
        tone(311, 0.150, env="adsr"),  # Eb4
    ])


def make_signal() -> np.ndarray:
    return tone(1000, 0.060, env="sharp")


def make_emergency() -> np.ndarray:
    chunks = []
    for _ in range(3):
        chunks.append(tone(880, 0.120, env="flat", shape="square"))
        chunks.append(silence(0.080))
    return np.concatenate(chunks)


def make_news_alert() -> np.ndarray:
    chunk = tone(440, 0.200, env="flat") + tone(480, 0.200, env="flat")
    out = []
    for _ in range(2):
        out.append(chunk)
        out.append(silence(0.200))
    return np.concatenate(out)


def make_error() -> np.ndarray:
    return tone(220, 0.250, env="adsr") + tone(233, 0.250, env="adsr")


def make_complete() -> np.ndarray:
    return tone(1320, 0.800, env="exp")


SPECS: dict[str, callable] = {
    "trading_open.wav": make_trading_open,
    "profit.wav":       make_profit,
    "loss.wav":         make_loss,
    "signal.wav":       make_signal,
    "emergency.wav":    make_emergency,
    "news_alert.wav":   make_news_alert,
    "error.wav":        make_error,
    "complete.wav":     make_complete,
}


def main() -> int:
    for name, fn in SPECS.items():
        path = OUT_DIR / name
        write(path, fn())
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
