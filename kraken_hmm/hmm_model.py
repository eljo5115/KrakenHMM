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
import contextlib
import os


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

        # log returns (length n-1): r_t = ln(p_{t+1}/p_t)
        # use small epsilon to avoid division by zero
        log_returns = np.log((prices[1:] + _EPS) / (prices[:-1] + _EPS))

        # volume-derived signal aligned to returns: use log-volume-change when available
        if volumes is not None:
            vols = np.asarray(volumes, dtype=float)
            if len(vols) != n:
                vol_changes = np.zeros_like(log_returns)
            else:
                # prefer log-differences for scale invariance
                vol_changes = np.diff(np.log(vols + _EPS))
        else:
            vol_changes = np.zeros_like(log_returns)

        # SMA-based feature: for each return index i (corresponds to price index i+1)
        sma_window = max(1, int(sma_window))
        sma_pct = np.zeros_like(log_returns)
        for i in range(len(log_returns)):
            price_idx = i + 1
            start = max(0, price_idx - sma_window + 1)
            window = prices[start : price_idx + 1]
            sma = float(window.mean()) if len(window) > 0 else float(prices[price_idx])
            sma_pct[i] = (prices[price_idx] - sma) / (sma + _EPS)

        # Additional technical indicators (appended after original features when sample size allows)
        # 1) RSI (relative strength index) computed on pct_returns using a short window
        # 2) Force Index: (price change) * volume at the current bar (aligned to pct_returns)
        # These are optional and included only when there are enough observations to avoid degenerate fits.
        def compute_rsi(returns: np.ndarray, window: int = 14) -> np.ndarray:
            rsi = np.zeros_like(returns)
            gains = np.where(returns > 0, returns, 0.0)
            losses = np.where(returns < 0, -returns, 0.0)
            for i in range(len(returns)):
                start = max(0, i - window + 1)
                g = gains[start : i + 1]
                l = losses[start : i + 1]
                avg_gain = g.mean() if g.size > 0 else 0.0
                avg_loss = l.mean() if l.size > 0 else _EPS
                rs = avg_gain / max(avg_loss, _EPS)
                rsi[i] = 100.0 - (100.0 / (1.0 + rs))
            return rsi

        def compute_force_index(prices_arr: np.ndarray, vols_arr: Optional[np.ndarray]) -> np.ndarray:
            # Force Index for index i corresponds to (price_{i+1} - price_i) * volume_{i+1}
            fi = np.zeros_like(log_returns)
            if vols_arr is None or len(vols_arr) != n:
                return fi
            for i in range(len(log_returns)):
                price_idx = i + 1
                fi[i] = (prices_arr[price_idx] - prices_arr[price_idx - 1]) * vols_arr[price_idx]
            # scale/normalize small values
            return fi

        # compute RSI on log returns for stability
        rsi = compute_rsi(log_returns, window=14)
        fi = compute_force_index(prices, np.asarray(volumes, dtype=float) if volumes is not None else None)

        # Feature selection: for very small sample sizes, use fewer features.
        # Preserve original ordering for backward compatibility: [pct_returns, vol_changes, sma_pct]
        # Append additional indicators (rsi, force index) only when there are many observations.
        # note: use log_returns internally for training features
        n_obs = len(log_returns)
        if n_obs < 30:
            X = log_returns.reshape(-1, 1)
        elif n_obs < 80:
            X = np.column_stack([log_returns, vol_changes])
        elif n_obs < 200:
            X = np.column_stack([log_returns, vol_changes, sma_pct])
        else:
            # include RSI and Force Index when we have ample observations
            X = np.column_stack([log_returns, vol_changes, sma_pct, rsi, fi])

        # choose number of components adaptively based on samples available
        n_samples, n_features = X.shape
        max_comps = max(2, n_samples // 3)
        n_components = min(self.n_states, max_comps)

        # covariance heuristic
        cov_type = "full" if n_samples >= max(60, 10 * n_features) else "diag"

        # instantiate and fit with more iterations and a tolerance to improve convergence
        try:
            model = GaussianHMM(n_components=n_components, covariance_type=cov_type, random_state=self.random_state, n_iter=500, tol=1e-4)
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.filterwarnings('ignore', message='Some rows of transmat_ have zero sum.*')
                _warnings.filterwarnings('ignore', message='Model is not converging.*')
                _warnings.filterwarnings('ignore')
                # suppress noisy stdout/stderr from hmmlearn during fit
                with open(os.devnull, 'w') as devnull:
                    with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                        model.fit(X)
        except Exception:
            # fallback to simpler model with reasonable settings
            try:
                model = GaussianHMM(n_components=2, covariance_type="diag", random_state=self.random_state, n_iter=200, tol=1e-4)
                import warnings as _warnings
                with _warnings.catch_warnings():
                    _warnings.filterwarnings('ignore', message='Some rows of transmat_ have zero sum.*')
                    _warnings.filterwarnings('ignore', message='Model is not converging.*')
                    _warnings.filterwarnings('ignore')
                    with open(os.devnull, 'w') as devnull:
                        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                            model.fit(X)
            except Exception:
                # last-resort: instantiate without fitting (keep previous behavior safe)
                model = GaussianHMM(n_components=max(2, min(4, n_components)), covariance_type="diag", random_state=self.random_state, n_iter=50)

        # regularize covariances to avoid nearly-zero variances
        min_covar = 1e-8
        try:
            covs = model.covars_
            # handle common covariance shapes depending on covariance_type
            if model.covariance_type == "full":
                # expected shape: (n_components, n_dim, n_dim)
                covs = np.asarray(covs)
                if covs.ndim == 3:
                    for k in range(model.n_components):
                        cov = covs[k]
                        diag_idx = np.arange(cov.shape[0])
                        cov[diag_idx, diag_idx] += min_covar
                        covs[k] = cov
                else:
                    # fallback: build small full covariances from diag of existing covs
                    covs = np.tile(np.atleast_2d(np.maximum(np.var(X, axis=0, ddof=1), min_covar)), (model.n_components, 1, 1))
                model.covars_ = covs
            elif model.covariance_type == "diag":
                # expected shape: (n_components, n_dim)
                covs = np.asarray(covs)
                if covs.ndim == 2 and covs.shape[1] == n_features:
                    covs = np.maximum(covs, min_covar)
                else:
                    # construct reasonable per-component diag covariances from empirical feature variances
                    feat_var = np.maximum(np.var(X, axis=0, ddof=1), min_covar)
                    covs = np.tile(feat_var, (model.n_components, 1))
                model.covars_ = covs
            else:
                # other types (spherical, tied) - try to coerce
                covs = np.asarray(covs)
                if covs.ndim == 1:
                    # spherical: one variance per component
                    covs = np.maximum(covs, min_covar)
                    model.covars_ = covs
                else:
                    # fallback to per-component diag using empirical variances
                    feat_var = np.maximum(np.var(X, axis=0, ddof=1), min_covar)
                    model.covars_ = np.tile(feat_var, (model.n_components, 1))
        except Exception:
            # best-effort: if anything unexpected happened, set safe default diag covariances
            try:
                feat_var = np.maximum(np.var(X, axis=0, ddof=1), min_covar)
                model.covars_ = np.tile(feat_var, (model.n_components, 1))
            except Exception:
                # last resort: small constant
                model.covars_ = np.full((model.n_components, max(1, n_features)), min_covar)

        # sanitize transition matrix and start probabilities to avoid zero-sum rows
        try:
            if hasattr(model, 'transmat_'):
                tm = model.transmat_
                # replace any all-zero row with a small uniform distribution
                row_sums = tm.sum(axis=1)
                zero_rows = (row_sums == 0)
                if zero_rows.any():
                    n_comp = model.n_components
                    small = 1.0 / float(n_comp)
                    for i in range(n_comp):
                        if row_sums[i] == 0:
                            tm[i, :] = small
                    # renormalize rows
                    tm = tm / tm.sum(axis=1, keepdims=True)
                    model.transmat_ = tm
            if hasattr(model, 'startprob_'):
                sp = model.startprob_
                if not np.isfinite(sp).all() or sp.sum() == 0:
                    # fallback to uniform start probabilities
                    model.startprob_ = np.full(model.n_components, 1.0 / float(model.n_components))
                else:
                    # ensure normalization
                    model.startprob_ = model.startprob_ / float(model.startprob_.sum())
        except Exception:
            # best-effort: don't fail training because of sanitization
            pass

        # attach
        self.model = model
        self.n_components_used = model.n_components
        self.covariance_type_used = model.covariance_type
        self.features = X
        # store internal returns as log-returns
        self.returns = log_returns
        self.states = self.model.predict(X)

    def state_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return mean and sample-std of percentage returns per hidden state."""
        if not hasattr(self, "returns") or not hasattr(self, "states"):
            return np.zeros(self.n_states), np.zeros(self.n_states)

        # convert stored log-returns back to arithmetic pct returns for downstream bounds
        n_comp = self.n_components_used or self.n_states
        means = np.zeros(n_comp)
        stds = np.zeros(n_comp)
        if not hasattr(self, "returns") or len(self.returns) == 0:
            return means, stds
        arith_returns = np.exp(self.returns) - 1.0
        overall_std = float(np.std(arith_returns, ddof=1)) if len(arith_returns) > 1 else float(_EPS)
        overall_std = max(overall_std, float(_EPS))
        for s in range(n_comp):
            vals = arith_returns[self.states == s].ravel()
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

        # compute log returns for prediction
        log_returns = np.log((prices[1:] + _EPS) / (prices[:-1] + _EPS))
        # Match the same volume-derived feature used in fit (log-diff when possible)
        if recent_volumes is not None and len(recent_volumes) == len(prices):
            vol_changes = np.diff(np.log(np.asarray(recent_volumes) + _EPS))
        else:
            vol_changes = np.zeros_like(log_returns)

        sma_window = max(1, int(sma_window))
        sma_pct = np.zeros_like(log_returns)
        for i in range(len(log_returns)):
            price_idx = i + 1
            start = max(0, price_idx - sma_window + 1)
            window = prices[start : price_idx + 1]
            sma = float(window.mean()) if len(window) > 0 else float(prices[price_idx])
            sma_pct[i] = (prices[price_idx] - sma) / (sma + _EPS)

        # Build RSI and Force Index aligned to the returns (only used when enough observations)
        def compute_rsi_local(returns: np.ndarray, window: int = 14) -> np.ndarray:
            rsi_local = np.zeros_like(returns)
            gains = np.where(returns > 0, returns, 0.0)
            losses = np.where(returns < 0, -returns, 0.0)
            for i in range(len(returns)):
                start = max(0, i - window + 1)
                g = gains[start : i + 1]
                l = losses[start : i + 1]
                avg_gain = g.mean() if g.size > 0 else 0.0
                avg_loss = l.mean() if l.size > 0 else _EPS
                rs = avg_gain / max(avg_loss, _EPS)
                rsi_local[i] = 100.0 - (100.0 / (1.0 + rs))
            return rsi_local

        def compute_force_local(prices_arr: np.ndarray, vols_arr: Optional[np.ndarray]) -> np.ndarray:
            fi_local = np.zeros_like(log_returns)
            if vols_arr is None or len(vols_arr) != len(prices_arr):
                return fi_local
            for i in range(len(log_returns)):
                price_idx = i + 1
                fi_local[i] = (prices_arr[price_idx] - prices_arr[price_idx - 1]) * vols_arr[price_idx]
            return fi_local

        rsi_local = compute_rsi_local(log_returns, window=14)
        vols_arr = np.asarray(recent_volumes, dtype=float) if recent_volumes is not None else None
        fi_local = compute_force_local(prices, vols_arr)

        # Mirror feature selection used in fit
        n_obs = len(log_returns)
        if n_obs < 30:
            X = log_returns.reshape(-1, 1)
        elif n_obs < 80:
            X = np.column_stack([log_returns, vol_changes])
        elif n_obs < 200:
            X = np.column_stack([log_returns, vol_changes, sma_pct])
        else:
            X = np.column_stack([log_returns, vol_changes, sma_pct, rsi_local, fi_local])

        return int(self.model.predict(X)[-1])

    def partial_update(self, prices: Sequence[float], volumes: Optional[Sequence[float]] = None, sma_window: int = 5, window: int = 200, n_iter: int = 20):
        """Online-ish update: warm-start EM using recent `window` prices.

        This method attempts to avoid full reinitialization by creating a new
        GaussianHMM instance with the same topology as the currently-fitted
        model, copying over parameters, and calling fit with init_params set
        to '' so it continues EM from the provided parameters. This performs
        a small number of iterations (`n_iter`) on the recent window to adapt
        to new data while keeping cost low.

        Note: this is a best-effort approach — it depends on hmmlearn allowing
        fit() to proceed without reinitializing parameters when init_params is
        empty. If no model exists yet, this behaves like a normal fit.
        """
        prices = list(prices)
        n = len(prices)
        if n < 3:
            return

        # compute features identical to fit, but only for the requested window
        start_idx = max(0, n - window)
        window_prices = prices[start_idx:]
        window_vols = None
        if volumes is not None:
            vols = list(volumes)
            window_vols = vols[start_idx:]

        # build feature matrix X using same logic as fit
        # reuse local computation by calling internal fit steps, but duplicate here to avoid refactor
        prices_arr = np.asarray(window_prices, dtype=float)
        n_w = len(prices_arr)
        log_returns = np.log((prices_arr[1:] + _EPS) / (prices_arr[:-1] + _EPS))

        if window_vols is not None and len(window_vols) == n_w:
            vols_arr = np.asarray(window_vols, dtype=float)
            vol_changes = np.diff(np.log(vols_arr + _EPS))
        else:
            vol_changes = np.zeros_like(log_returns)

        sma_window = max(1, int(sma_window))
        sma_pct = np.zeros_like(log_returns)
        for i in range(len(log_returns)):
            price_idx = i + 1
            start = max(0, price_idx - sma_window + 1)
            window2 = prices_arr[start : price_idx + 1]
            sma = float(window2.mean()) if len(window2) > 0 else float(prices_arr[price_idx])
            sma_pct[i] = (prices_arr[price_idx] - sma) / (sma + _EPS)

        # compute RSI and Force Index
        def compute_rsi_local(returns: np.ndarray, window_r: int = 14) -> np.ndarray:
            rsi_local = np.zeros_like(returns)
            gains = np.where(returns > 0, returns, 0.0)
            losses = np.where(returns < 0, -returns, 0.0)
            for i in range(len(returns)):
                start = max(0, i - window_r + 1)
                g = gains[start : i + 1]
                l = losses[start : i + 1]
                avg_gain = g.mean() if g.size > 0 else 0.0
                avg_loss = l.mean() if l.size > 0 else _EPS
                rs = avg_gain / max(avg_loss, _EPS)
                rsi_local[i] = 100.0 - (100.0 / (1.0 + rs))
            return rsi_local

        def compute_force_local(prices_arr: np.ndarray, vols_arr_local: Optional[np.ndarray]) -> np.ndarray:
            fi_local = np.zeros_like(log_returns)
            if vols_arr_local is None or len(vols_arr_local) != len(prices_arr):
                return fi_local
            for i in range(len(log_returns)):
                price_idx = i + 1
                fi_local[i] = (prices_arr[price_idx] - prices_arr[price_idx - 1]) * vols_arr_local[price_idx]
            return fi_local

        rsi_local = compute_rsi_local(log_returns, window_r=14)
        fi_local = compute_force_local(prices_arr, np.asarray(window_vols, dtype=float) if window_vols is not None else None)

        n_obs = len(log_returns)
        if n_obs < 30:
            X = log_returns.reshape(-1, 1)
        elif n_obs < 80:
            X = np.column_stack([log_returns, vol_changes])
        elif n_obs < 200:
            X = np.column_stack([log_returns, vol_changes, sma_pct])
        else:
            X = np.column_stack([log_returns, vol_changes, sma_pct, rsi_local, fi_local])

        # If no existing model, just fit normally on window
        if self.model is None:
            try:
                self.fit(prices, volumes=volumes, sma_window=sma_window)
            except Exception:
                return
            return

        # create a new HMM with same topology and copy parameters
        try:
            from hmmlearn.hmm import GaussianHMM

            n_comp = getattr(self.model, 'n_components', self.n_states)
            cov_type = getattr(self.model, 'covariance_type', 'diag')
            new_model = GaussianHMM(n_components=n_comp, covariance_type=cov_type, random_state=self.random_state, n_iter=n_iter)
            # prevent reinitialization of params so EM continues from current params
            new_model.init_params = ''
            # copy parameters if present
            if hasattr(self.model, 'startprob_'):
                new_model.startprob_ = np.array(self.model.startprob_)
            if hasattr(self.model, 'transmat_'):
                new_model.transmat_ = np.array(self.model.transmat_)
            if hasattr(self.model, 'means_'):
                new_model.means_ = np.array(self.model.means_)
            if hasattr(self.model, 'covars_'):
                new_model.covars_ = np.array(self.model.covars_)

            # fit on X to adapt params (safely, suppress convergence warnings)
            try:
                import warnings as _warnings
                with _warnings.catch_warnings():
                    _warnings.filterwarnings('ignore')
                    new_model.fit(X)
            except Exception:
                # if incremental EM fails, keep the existing model and return
                return

            # replace current model
            self.model = new_model
            # recompute stored features and returns/states using entire provided prices
            try:
                # rebuild features for whole provided prices set
                self.fit(prices, volumes=volumes, sma_window=sma_window)
            except Exception:
                # if rebuilding fails, at least keep the incremental model
                pass
        except Exception:
            # fallback: full fit on provided prices
            try:
                self.fit(prices, volumes=volumes, sma_window=sma_window)
            except Exception:
                return
