"""VPIN — Volume-synchronized Probability of Informed Trading.

Easley, López de Prado, O'Hara (2012). Identifies toxic order flow by
estimating the fraction of trade volume coming from informed traders
inside fixed-volume buckets (not fixed-time bars).

Algorithm:
  1. Bucket the tick stream by accumulated volume. Each bucket closes
     when total volume since the last close reaches V (the "volume
     threshold").
  2. For each bucket, classify each tick's volume as buy- or sell-side
     using the *bulk volume classification* rule (Easley et al.): the
     fraction of volume attributed to buyers is a smooth function of
     the price change between the bucket's open and close, scaled by
     a recent volatility estimate.
  3. VPIN_t = mean over the last N buckets of |buy_vol − sell_vol| / V.
  4. VPIN > threshold (default 0.4) signals toxic flow → pause entries
     or widen confluence requirements.

Used as a precondition gate in consensus.evaluate(): when toxic flow is
detected, even an otherwise-perfect signal is rejected because the
adverse-selection cost of crossing the spread will eat the edge.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping


VPIN_TOXIC_THRESHOLD = 0.40
DEFAULT_BUCKET_VOLUME = 50.0  # contracts / lots in the bucket
DEFAULT_SMOOTH_WINDOW = 50    # number of buckets to average


@dataclass(frozen=True)
class VPINBucket:
    open_price: float
    close_price: float
    total_volume: float
    buy_volume: float
    sell_volume: float

    @property
    def imbalance(self) -> float:
        if self.total_volume <= 0:
            return 0.0
        return abs(self.buy_volume - self.sell_volume) / self.total_volume


def _phi_std_normal_cdf(z: float) -> float:
    """Standard normal CDF — used by Easley bulk-volume classification."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def bulk_volume_classify(
    open_price: float,
    close_price: float,
    total_volume: float,
    sigma: float,
) -> tuple[float, float]:
    """Easley et al. (2012) bulk-volume classification.

    Returns (buy_volume, sell_volume) with buy_volume + sell_volume =
    total_volume. The buy-side fraction is Phi((close - open) / sigma),
    where Phi is the standard-normal CDF.
    """
    if total_volume <= 0 or sigma <= 0:
        return total_volume / 2.0, total_volume / 2.0
    z = (close_price - open_price) / sigma
    buy_frac = _phi_std_normal_cdf(z)
    buy_vol = total_volume * buy_frac
    return buy_vol, total_volume - buy_vol


def build_volume_buckets(
    ticks: Iterable[Mapping],
    bucket_volume: float = DEFAULT_BUCKET_VOLUME,
    *,
    sigma: float | None = None,
) -> list[VPINBucket]:
    """Aggregate `ticks` into fixed-volume buckets.

    Each tick must expose at least one of {bid, ask, last, close, mid} for
    its price, plus a `volume` field. Ticks are processed in order. When
    accumulated volume in the current bucket reaches `bucket_volume`, the
    bucket closes and a new one opens.

    `sigma` (price-change scale) is optional; if omitted, it is estimated
    in-line from the per-tick price differences in this stream.
    """
    bucket_volume = float(bucket_volume)
    if bucket_volume <= 0:
        raise ValueError("bucket_volume must be positive")

    prices: list[float] = []
    volumes: list[float] = []
    for t in ticks:
        v = float(t.get("volume", 1.0) or 1.0)
        if v <= 0:
            continue
        bid = float(t.get("bid", 0.0) or 0.0)
        ask = float(t.get("ask", 0.0) or 0.0)
        if "last" in t and t["last"] is not None:
            p = float(t["last"])
        elif "close" in t and t["close"] is not None:
            p = float(t["close"])
        elif "mid" in t and t["mid"] is not None:
            p = float(t["mid"])
        elif bid > 0 and ask > 0:
            p = (bid + ask) / 2.0
        elif bid > 0:
            p = bid
        elif ask > 0:
            p = ask
        else:
            continue
        prices.append(p)
        volumes.append(v)

    if not prices:
        return []
    if sigma is None or sigma <= 0:
        diffs = [prices[i] - prices[i - 1] for i in range(1, len(prices))]
        if len(diffs) >= 2:
            mean = sum(diffs) / len(diffs)
            var = sum((d - mean) ** 2 for d in diffs) / (len(diffs) - 1)
            sigma = max(math.sqrt(var), 1e-9)
        else:
            sigma = 1e-6

    buckets: list[VPINBucket] = []
    i = 0
    n = len(prices)
    while i < n:
        cur_vol = 0.0
        open_p = prices[i]
        close_p = prices[i]
        while i < n and cur_vol < bucket_volume:
            need = bucket_volume - cur_vol
            v = volumes[i]
            if v <= need:
                cur_vol += v
                close_p = prices[i]
                i += 1
            else:
                # Partial fill: take only `need` from this tick, then close.
                cur_vol += need
                close_p = prices[i]
                volumes[i] = v - need
                break
        if cur_vol <= 0:
            break
        buy_vol, sell_vol = bulk_volume_classify(open_p, close_p, cur_vol, sigma)
        buckets.append(VPINBucket(
            open_price=open_p, close_price=close_p,
            total_volume=cur_vol, buy_volume=buy_vol, sell_volume=sell_vol,
        ))
    return buckets


def compute_vpin(
    ticks: Iterable[Mapping],
    *,
    bucket_volume: float = DEFAULT_BUCKET_VOLUME,
    smooth_window: int = DEFAULT_SMOOTH_WINDOW,
    sigma: float | None = None,
) -> float:
    """Return current VPIN — mean imbalance over the last `smooth_window` buckets."""
    buckets = build_volume_buckets(ticks, bucket_volume=bucket_volume, sigma=sigma)
    if not buckets:
        return 0.0
    recent = buckets[-smooth_window:]
    return sum(b.imbalance for b in recent) / len(recent)


def vpin_gate(
    vpin_score: float,
    *,
    threshold: float = VPIN_TOXIC_THRESHOLD,
) -> bool:
    """Return True iff order flow is acceptable (VPIN below threshold)."""
    return vpin_score < threshold


def vpin_regime(vpin_score: float) -> str:
    if vpin_score < 0.20:
        return "BENIGN"
    if vpin_score < VPIN_TOXIC_THRESHOLD:
        return "ELEVATED"
    return "TOXIC"
