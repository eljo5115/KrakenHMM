"""Hidden Markov Model utilities for price series.

This module fits Gaussian HMMs on percentage returns (not log-returns).
Features per observation (aligned to returns) are:
  - pct return: (p_t / p_{t-1}) - 1  (primary signal)
  - volume log-change: diff(log(volume)) if volumes provided else 0
  - price vs SMA: (price - sma)/sma (sma over `sma_window` ending at the price)

The implementation adapts the number of features and model complexity based
on available sample size to avoid degenerate fits on very small histories.
"""

from typing import Sequence, Tuple, Optional
import numpy as np
from hmmlearn.hmm import GaussianHMM


# small epsilon to avoid division by zero
_EPS = 1e-8


class HMMModel:
    def __init__(self, n_states: int = 3, random_state: Optional[int] = None):
        self.n_states = n_states
        self.random_state = random_state
        self.model = None
        self.n_components_used = None
        self.covariance_type_used = None
        self.features = None
        self.returns = None
        self.states = None

    def fit(self, prices: Sequence[float], volumes: Optional[Sequence[float]] = None, sma_window: int = 5):
        prices = np.asarray(prices, dtype=float)
        n = len(prices)
        if n < 3:
            raise ValueError("Need at least 3 price points to fit HMM")

        # percentage returns (length n-1): pct_t = p_{t+1}/p_t - 1
        pct_returns = (prices[1:] / (prices[:-1] + _EPS)) - 1.0

        # volume-derived signal aligned to returns
        if volumes is not None:
            vols = np.asarray(volumes, dtype=float)
            if len(vols) != n:
                vol_changes = np.zeros_like(pct_returns)
            else:
                vol_changes = np.diff(vols + _EPS)
        else:
            vol_changes = np.zeros_like(pct_returns)

        # SMA-based feature: for each return index i (corresponds to price index i+1)
        sma_window = max(1, int(sma_window))
        sma_pct = np.zeros_like(pct_returns)
        for i in range(len(pct_returns)):
            price_idx = i + 1
            start = max(0, price_idx - sma_window + 1)
            window = prices[start : price_idx + 1]
            sma = float(window.mean()) if len(window) > 0 else float(prices[price_idx])
            sma_pct[i] = (prices[price_idx] - sma) / (sma + _EPS)

        # Feature selection: for very small sample sizes, use fewer features
        n_obs = len(pct_returns)
        if n_obs < 30:
            X = pct_returns.reshape(-1, 1)
        elif n_obs < 80:
            X = np.column_stack([pct_returns, vol_changes])
        else:
            X = np.column_stack([pct_returns, vol_changes, sma_pct])

        # choose number of components adaptively based on samples available
        n_samples, n_features = X.shape
        max_comps = max(2, n_samples // 3)
        n_components = min(self.n_states, max_comps)

        # covariance heuristic
        cov_type = "full" if n_samples >= max(60, 10 * n_features) else "diag"

        # instantiate and fit
        model = GaussianHMM(n_components=n_components, covariance_type=cov_type, random_state=self.random_state, n_iter=200)
        try:
            model.fit(X)
        except Exception:
            # fallback to simpler model
            model = GaussianHMM(n_components=2, covariance_type="diag", random_state=self.random_state, n_iter=100)
            model.fit(X)

        # regularize covariances to avoid nearly-zero variances
        min_covar = 1e-8
        if model.covariance_type == "full":
            for k in range(model.n_components):
                cov = model.covars_[k]
                # add tiny floor to diagonal
                diag_idx = np.arange(cov.shape[0])
                cov[diag_idx, diag_idx] += min_covar
                model.covars_[k] = cov
        else:
            model.covars_ = np.maximum(model.covars_, min_covar)

        # attach
        self.model = model
        self.n_components_used = model.n_components
        self.covariance_type_used = model.covariance_type
        self.features = X
        self.returns = pct_returns
        self.states = self.model.predict(X)

    def state_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return mean and sample-std of percentage returns per hidden state."""
        if not hasattr(self, "returns") or not hasattr(self, "states"):
            return np.zeros(self.n_states), np.zeros(self.n_states)

        n_comp = self.n_components_used or self.n_states
        means = np.zeros(n_comp)
        stds = np.zeros(n_comp)
        overall_std = float(np.std(self.returns, ddof=1)) if len(self.returns) > 1 else float(_EPS)
        overall_std = max(overall_std, float(_EPS))
        for s in range(n_comp):
            vals = self.returns[self.states == s].ravel()
            if len(vals) > 0:
                means[s] = float(np.mean(vals))
                stds[s] = float(np.std(vals, ddof=1)) if len(vals) > 1 else overall_std
            else:
                means[s] = 0.0
                stds[s] = overall_std
        return means, stds

    def predict_state(self, recent_prices: Sequence[float], recent_volumes: Optional[Sequence[float]] = None, sma_window: int = 5) -> int:
        prices = np.asarray(recent_prices, dtype=float)
        if len(prices) < 2:
            raise ValueError("need at least 2 prices to predict state")

        pct_returns = (prices[1:] / (prices[:-1] + _EPS)) - 1.0
        if recent_volumes is not None and len(recent_volumes) == len(prices):
            vol_changes = np.diff(np.log(np.asarray(recent_volumes) + _EPS))
        else:
            vol_changes = np.zeros_like(pct_returns)

        sma_window = max(1, int(sma_window))
        sma_pct = np.zeros_like(pct_returns)
        for i in range(len(pct_returns)):
            price_idx = i + 1
            start = max(0, price_idx - sma_window + 1)
            window = prices[start : price_idx + 1]
            sma = float(window.mean()) if len(window) > 0 else float(prices[price_idx])
            sma_pct[i] = (prices[price_idx] - sma) / (sma + _EPS)

        # Mirror feature selection used in fit
        n_obs = len(pct_returns)
        if n_obs < 30:
            X = pct_returns.reshape(-1, 1)
        elif n_obs < 80:
            X = np.column_stack([pct_returns, vol_changes])
        else:
            X = np.column_stack([pct_returns, vol_changes, sma_pct])

        return int(self.model.predict(X)[-1])
