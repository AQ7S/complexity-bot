"""13×13 rolling correlation matrix on log returns.

Inputs:
    closes: dict[symbol -> pd.Series of close prices, indexed by ts]
    The 13 symbol order is taken from the dict's key order — pass an OrderedDict
    or use SYMBOL_NAMES from engine.config.symbols for canonical ordering.

Output:
    DataFrame of shape (n_symbols, n_symbols) with values in [-1, 1], symmetric,
    diagonal = 1.0. NaN in any input pair is propagated to that cell as 0.0.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def returns_matrix(closes: Mapping[str, pd.Series], window: int = 200) -> pd.DataFrame:
    """Align closes on a common index, take log returns, keep last `window` rows."""
    df = pd.DataFrame({sym: s.astype(float) for sym, s in closes.items()})
    df = df.sort_index().ffill().dropna(how="all")
    rets = np.log(df / df.shift(1))
    return rets.tail(window)


def correlation_matrix(closes: Mapping[str, pd.Series], *, window: int = 200) -> pd.DataFrame:
    """Pearson correlation across the last `window` log returns."""
    rets = returns_matrix(closes, window=window)
    if rets.shape[0] < 2:
        n = rets.shape[1]
        return pd.DataFrame(np.eye(n), index=rets.columns, columns=rets.columns)
    corr = rets.corr().fillna(0.0)
    np.fill_diagonal(corr.values, 1.0)
    return corr


def correlated_pairs(corr: pd.DataFrame, threshold: float = 0.80) -> list[tuple[str, str, float]]:
    """Upper-triangle pairs with |corr| >= threshold."""
    out: list[tuple[str, str, float]] = []
    cols = list(corr.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            v = float(corr.at[a, b])
            if abs(v) >= threshold:
                out.append((a, b, v))
    return out
