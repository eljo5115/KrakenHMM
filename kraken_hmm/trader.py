"""Trading logic that ranks assets by Sharpe ratio computed from HMM predicted states.

This module provides an async Trader class that consumes price updates, fits
HMM models per asset, computes per-state mean returns, computes a simple
Sharpe-like metric, and selects top-N assets to allocate capital to.
"""

from typing import Dict, List, Optional
import asyncio
import numpy as np
import math
from dataclasses import dataclass, field

from .hmm_model import HMMModel
import os
import pickle
import json
import time
import datetime


@dataclass
class Position:
    pair: str
    qty: float = 0.0
    entry_price: Optional[float] = None
    stop_price: Optional[float] = None
    take_price: Optional[float] = None


class Trader:
    def __init__(self, total_capital: float = 1000.0, n_assets: int = 3, n_states: int = 3):
        self.total_capital = total_capital
        self.n_assets = n_assets
        self.n_states = n_states

        # per-pair time series
        self.prices: Dict[str, List[float]] = {}
        # per-pair timestamps (parallel to prices)
        self.times: Dict[str, List[float]] = {}
        # optional per-pair volumes
        self.volumes: Dict[str, List[float]] = {}
        self.models: Dict[str, HMMModel] = {}
        self.positions: Dict[str, Position] = {}

        # config
        self.min_history = 100
        self.recent_window = 10
        # order execution controls
        self.execute_orders = False
        self.api_key = None
        self.api_secret = None
        # trade logging (rotate by day). Set either trade_log_file (exact file)
        # or trade_log_dir (directory) to control where logs are written.
        # Default is a repository-local ./logs directory (hardcoded)
        self.trade_log_dir = "logs"
        self.trade_log_file = None
        # snapshot of the last allocation set that was executed (pair->capital)
        # used to ensure allocations are applied only once per unique set.
        self._last_executed_allocs = None
        # date (UTC YYYYMMDD) when allocations were last applied. Used to
        # implement the once-per-day train->buy cycle.
        self._last_allocation_date = None
        # cache last written json line to avoid duplicate consecutive writes
        self._last_written_line = None
        # persistent state file to survive restarts (positions, allocation snapshot, last allocation date)
        self.state_file = os.path.join("state", "trader_state.json")

    def add_price(self, pair: str, price: float, volume: Optional[float] = None, ts: Optional[float] = None):
        if price is None:
            return
        arr = self.prices.setdefault(pair, [])
        arr.append(price)
        # timestamp for this price tick
        if ts is None:
            ts = time.time()
        tarr = self.times.setdefault(pair, [])
        tarr.append(float(ts))
        if volume is not None:
            varr = self.volumes.setdefault(pair, [])
            varr.append(volume)
        # keep history bounded
        if len(arr) > 2000:
            del arr[0:-2000]
        # keep volumes bounded to same length
        if volume is not None:
            if len(self.volumes.get(pair, [])) > 2000:
                del self.volumes[pair][0:-2000]

    def seed_history(self, pair: str, closes: List[float], volumes: Optional[List[float]] = None, times: Optional[List[float]] = None):
        """Seed the trader's internal history for a pair with historical daily data.

        `closes` should be chronological (oldest first). `volumes` if provided
        must have the same length as `closes`.
        After seeding, the method will attempt to fit the model for the pair.
        """
        if not closes:
            return
        self.prices[pair] = list(map(float, closes))
        # seed timestamps: use provided times if available and len matches, otherwise generate daily-spaced timestamps
        if times is not None and len(times) == len(closes):
            self.times[pair] = list(map(float, times))
        else:
            self.times[pair] = [float(time.time()) - (len(closes) - i) * 86400 for i in range(len(closes))]
        if volumes is not None and len(volumes) == len(closes):
            self.volumes[pair] = list(map(float, volumes))
        else:
            # ensure volumes dict has an entry so downstream code can read it
            self.volumes.setdefault(pair, [])
        # bound lengths
        if len(self.prices[pair]) > 2000:
            self.prices[pair] = self.prices[pair][-2000:]
        if len(self.volumes.get(pair, [])) > 2000:
            self.volumes[pair] = self.volumes[pair][-2000:]

        # attempt immediate fit when seeded
        try:
            self.ensure_model(pair)
            vols = self.volumes.get(pair)
            self.models[pair].fit(self.prices[pair][-self.min_history :], volumes=(vols[-self.min_history :] if vols else None), sma_window=min(10, self.recent_window))
        except Exception:
            # if fit fails, keep history for later attempts
            pass

    def ensure_model(self, pair: str):
        if pair in self.models:
            return
        self.models[pair] = HMMModel(n_states=self.n_states, random_state=42)

    def try_fit(self, pair: str):
        prices = self.prices.get(pair, [])
        if len(prices) < self.min_history:
            return False
        self.ensure_model(pair)
        try:
            vols = self.volumes.get(pair)
            self.models[pair].fit(prices[-self.min_history:], volumes=(vols[-self.min_history:] if vols is not None else None), sma_window=min(10, self.recent_window))
            return True
        except Exception:
            return False

    def compute_sharpe_for_pair(self, pair: str) -> Optional[float]:
        """Return a Sharpe-like score for the pair.

        Primary source: HMM predicted state's mean/std. If the HMM reports a
        (near-)zero std (degenerate fit on very small samples), fall back to
        sample mean/std computed from recent log-returns. Small stds are
        clamped to EPS to avoid division-by-zero and to produce a usable
        diagnostic in debug mode.

        Returns None when not enough data is available for either method.
        """
        EPS = 1e-8
        prices = self.prices.get(pair, [])
        if len(prices) < 2:
            return None

        # prefer HMM-based metric when a model exists and enough history
        model = self.models.get(pair)
        if model is not None and len(prices) >= self.min_history:
            recent = prices[-self.recent_window :]
            recent_vols = None
            vols = self.volumes.get(pair)
            if vols is not None and len(vols) >= 2:
                recent_vols = vols[-self.recent_window :]
            try:
                state = model.predict_state(recent, recent_vols)
                means, stds = model.state_stats()
                if state < len(means):
                    mean = float(means[state])
                    std = float(stds[state])
                    if not math.isfinite(std) or std <= EPS:
                        # degenerate HMM state std -> fallback to sample returns
                        raise ValueError("degenerate state std")
                    return mean / max(std, EPS)
            except Exception:
                # fall through to fallback estimator below
                pass

        # Fallback: compute sample mean/std from recent log-returns
        if len(prices) < 3:
            return None
        recent_prices = np.array(prices[-self.recent_window :], dtype=float)
        pctrets = (recent_prices[1:] / (recent_prices[:-1] + EPS)) - 1.0
        if len(pctrets) < 2:
            return None
        m = float(np.mean(pctrets))
        s = float(np.std(pctrets, ddof=1))
        if not math.isfinite(s) or s <= EPS:
            return None
        return m / max(s, EPS)

    def rank_assets(self) -> List[str]:
        scores = {p: self.compute_sharpe_for_pair(p) for p in self.models.keys()}
        # filter out None scores and sort descending
        filtered = {p: s for p, s in scores.items() if s is not None}
        ranked = sorted(filtered.items(), key=lambda kv: kv[1], reverse=True)
        return [p for p, s in ranked[: self.n_assets]]

    def allocate(self) -> Dict[str, float]:
        """Simple equal-weight allocation among top assets.

        Returns dict pair->allocation (capital amount).
        """
        top = self.rank_assets()
        if not top:
            return {}
        per = self.total_capital / len(top)
        return {p: per for p in top}

    # Placeholder order execution
    async def execute_allocations(self, allocs: Dict[str, float]):
        """Pretend to place orders - replace with REST order calls to Kraken.

        This function updates positions map as if orders filled immediately.
        """
        # if order execution is enabled and API credentials provided, place real market orders
        if self.execute_orders and self.api_key and self.api_secret:
            try:
                from .api import place_market_order

                for pair, cap in allocs.items():
                    price = self.prices.get(pair, [])[-1]
                    if price is None or price <= 0:
                        continue
                    qty = cap / price
                    # log planned trade (live mode)
                    plan = {
                        "ts": int(time.time()),
                        "type": "plan",
                        "mode": "live",
                        "pair": pair,
                        "qty": qty,
                        "price": price,
                        "capital": cap,
                        "would_execute": True,
                    }
                    try:
                        self._write_trade(plan)
                    except Exception:
                        pass
                    # place market buy order (Kraken expects pair without slash in our helper)
                    try:
                        result = await place_market_order(pair, qty, side="buy", api_key=self.api_key, api_secret=self.api_secret)
                        # assume order filled at current price; determine thresholds from model
                        stop_price = None
                        take_price = None
                        model = self.models.get(pair)
                        if model is not None:
                            try:
                                means, stds = model.state_stats()
                                # use predicted state for the most recent window if available
                                try:
                                    state = model.predict_state(self.prices.get(pair, [])[-self.recent_window :], recent_volumes=self.volumes.get(pair, [])[-self.recent_window :])
                                except Exception:
                                    state = None
                                if state is not None and state < len(stds):
                                    s = float(stds[state])
                                    s = max(s, 1e-8)
                                    stop_price = price * (1.0 - s)
                                    take_price = price * (1.0 + 2.0 * s)
                            except Exception:
                                pass
                        # If the API returned order_info, try to extract executed details
                        executed_price = price
                        executed_qty = qty
                        executed_cost = None
                        executed_fee = None
                        order_info = None
                        if isinstance(result, dict):
                            order_info = result.get("order_info") or result.get("orderinfo") or result.get("order_info")
                        if order_info is None and isinstance(result, dict):
                            # sometimes place_market_order returns {'add_result': ..., 'order_info': {...}}
                            order_info = result.get("order_info")
                        if order_info is None and isinstance(result, dict):
                            # fall back: check nested 'order_info'
                            order_info = result.get("orderinfo")

                        if order_info and isinstance(order_info, dict):
                            # QueryOrders returns a dict keyed by txid; take first entry
                            try:
                                first = next(iter(order_info.values()))
                                # vol_exec, cost, fee, price may be present
                                ve = first.get("vol_exec") or first.get("vol_exec")
                                if ve is not None:
                                    executed_qty = float(ve)
                                cost = first.get("cost")
                                if cost is not None:
                                    executed_cost = float(cost)
                                fee = first.get("fee")
                                if fee is not None:
                                    # fee may be a string like '0.0002'
                                    executed_fee = float(fee)
                                pprice = first.get("price")
                                if pprice is not None:
                                    executed_price = float(pprice)
                            except Exception:
                                pass

                        self.positions[pair] = Position(pair=pair, qty=executed_qty, entry_price=executed_price, stop_price=stop_price, take_price=take_price)
                        # log the executed buy (live) with any available exchange-reported fills
                        rec = {
                            "ts": int(time.time()),
                            "type": "buy",
                            "mode": "live",
                            "pair": pair,
                            "qty": executed_qty,
                            "price": executed_price,
                            "capital": cap,
                            "stop_price": stop_price,
                            "take_price": take_price,
                            "cost": executed_cost,
                            "fee": executed_fee,
                            "order_info": order_info,
                            "raw_add_result": result.get("add_result") if isinstance(result, dict) else result,
                        }
                        try:
                            self._write_trade(rec)
                        except Exception:
                            pass
                        # persist state after successful buy
                        try:
                            self.save_state()
                        except Exception:
                            pass
                    except Exception:
                        # on failure, skip this allocation
                        continue
            except Exception:
                # fallback to simulated fills if API helper not available
                for pair, cap in allocs.items():
                    price = self.prices.get(pair, [])[-1]
                    if price is None:
                        continue
                    qty = cap / price
                    # log planned trade (fallback simulated)
                    plan = {
                        "ts": int(time.time()),
                        "type": "plan",
                        "mode": "fallback_simulated",
                        "pair": pair,
                        "qty": qty,
                        "price": price,
                        "capital": cap,
                        "would_execute": False,
                    }
                    try:
                        self._write_trade(plan)
                    except Exception:
                        pass
                    self.positions[pair] = Position(pair=pair, qty=qty, entry_price=price)
                    # log fallback/simulated buy
                    rec = {
                        "ts": int(time.time()),
                        "type": "buy",
                        "mode": "fallback_simulated",
                        "pair": pair,
                        "qty": qty,
                        "price": price,
                        "capital": cap,
                    }
                    try:
                        self._write_trade(rec)
                    except Exception:
                        pass
                    # persist state after simulated/fallback buy
                    try:
                        self.save_state()
                    except Exception:
                        pass
        else:
            # simulated fills
            for pair, cap in allocs.items():
                price = self.prices.get(pair, [])[-1]
                if price is None:
                    continue
                qty = cap / price
                # compute thresholds from model if possible
                stop_price = None
                take_price = None
                model = self.models.get(pair)
                if model is not None:
                    try:
                        means, stds = model.state_stats()
                        try:
                            state = model.predict_state(self.prices.get(pair, [])[-self.recent_window :], recent_volumes=self.volumes.get(pair, [])[-self.recent_window :])
                        except Exception:
                            state = None
                        if state is not None and state < len(stds):
                            s = float(stds[state])
                            s = max(s, 1e-8)
                            stop_price = price * (1.0 - s)
                            take_price = price * (1.0 + 2.0 * s)
                    except Exception:
                        pass
                self.positions[pair] = Position(pair=pair, qty=qty, entry_price=price, stop_price=stop_price, take_price=take_price)
                # log the simulated buy
                rec = {
                    "ts": int(time.time()),
                    "type": "buy",
                    "mode": "simulated",
                    "pair": pair,
                    "qty": qty,
                    "price": price,
                    "capital": cap,
                    "stop_price": stop_price,
                    "take_price": take_price,
                }
                try:
                    self._write_trade(rec)
                except Exception:
                    pass
                # single simulated buy record written above
                try:
                    self.save_state()
                except Exception:
                    pass

    # High level tick - called when new data arrives
    async def on_tick(self, pair: str, price: float, volume: Optional[float] = None):
        self.add_price(pair, price, volume)

        # periodically (or on each tick) compute allocations and execute
        # Only execute buys for pairs that we do not already hold. This
        # prevents the trader from re-buying the same planned allocations
        # on every incoming tick; once bought, positions are monitored for
        # stop/take events and will not be re-entered until a new
        # allocation set differs.
        # Allocation logic with re-entry allowed.
        # Determine today's UTC date string.
        utc_day = datetime.datetime.utcnow().strftime("%Y%m%d")
        allocs = self.allocate()
        if not allocs:
            # nothing to do
            pass
        else:
            # desired allocations that are not currently held
            missing = {p: cap for p, cap in allocs.items() if p not in self.positions}

            # First allocation of the day: apply entire allocation set once
            if self._last_allocation_date != utc_day:
                # if the allocation set differs from last executed, execute missing
                if allocs != self._last_executed_allocs:
                    if missing:
                        await self.execute_allocations(missing)
                    try:
                        self._last_executed_allocs = dict(allocs)
                    except Exception:
                        self._last_executed_allocs = allocs
                self._last_allocation_date = utc_day
            else:
                # same UTC day: allow re-entry for any desired pair we don't hold
                if missing:
                    await self.execute_allocations(missing)
                    # after re-entry, update last_executed_allocs to current desired set
                    try:
                        self._last_executed_allocs = dict(allocs)
                    except Exception:
                        self._last_executed_allocs = allocs

        # Check open positions for stop-loss / take-profit triggers for this pair
        pos = self.positions.get(pair)
        if pos is not None and pos.qty and pos.entry_price is not None:
            # evaluate stop / take if defined
            cur_price = price
            triggered = False
            if pos.stop_price is not None and cur_price <= pos.stop_price:
                triggered = True
                reason = "stop"
            elif pos.take_price is not None and cur_price >= pos.take_price:
                triggered = True
                reason = "take"

            if triggered:
                # close the position via market sell
                qty = pos.qty
                try:
                    result = None
                    mode = "simulated"
                    if self.execute_orders and self.api_key and self.api_secret:
                        from .api import place_market_order

                        try:
                            result = await place_market_order(pair, qty, side="sell", api_key=self.api_key, api_secret=self.api_secret)
                            mode = "live"
                        except Exception:
                            # if sell fails, keep position open
                            return
                    # simulated sell / or after successful sell: remove position
                    print(f"Closing position {pair} at {cur_price} due to {reason}")
                    # log the sell
                    entry = pos.entry_price
                    pnl = None
                    pnl_pct = None
                    try:
                        if entry is not None:
                            pnl = float((cur_price - float(entry)) * float(qty))
                            pnl_pct = float((cur_price / float(entry)) - 1.0)
                    except Exception:
                        pnl = None
                        pnl_pct = None
                    rec = {
                        "ts": int(time.time()),
                        "type": "sell",
                        "mode": mode,
                        "pair": pair,
                        "qty": qty,
                        "price": cur_price,
                        "entry_price": entry,
                        "reason": reason,
                        "pnl": pnl,
                        "pnl_pct": pnl_pct,
                        "result": str(result) if result is not None else None,
                    }
                    try:
                        self._write_trade(rec)
                    except Exception:
                        pass
                    self.positions.pop(pair, None)
                    # persist state after sell/close
                    try:
                        self.save_state()
                    except Exception:
                        pass
                except Exception:
                    # any error: keep position
                    pass

    def save_models(self, models_dir: str = "models"):
        """Persist fitted models and training metadata to disk.

        Creates `models_dir` and writes one pickle per pair plus a small
        metadata JSON with history length and timestamp.
        """
        os.makedirs(models_dir, exist_ok=True)
        for pair, model in self.models.items():
            safe = pair.replace("/", "_")
            pfile = os.path.join(models_dir, f"{safe}.pkl")
            meta_file = os.path.join(models_dir, f"{safe}_meta.json")
            try:
                with open(pfile, "wb") as f:
                    pickle.dump(model, f)
                meta = {
                    "pair": pair,
                    "history_len": len(self.prices.get(pair, [])),
                    "timestamp": int(time.time()),
                }
                # include a small slice of the last trained prices/volumes for reproducibility
                meta["prices_tail"] = self.prices.get(pair, [])[-10:]
                meta["volumes_tail"] = self.volumes.get(pair, [])[-10:]
                meta["times_tail"] = self.times.get(pair, [])[-10:]
                with open(meta_file, "w") as f:
                    json.dump(meta, f)
            except Exception:
                # best-effort: continue saving others
                continue

    def save_state(self):
        """Persist trader runtime state (positions, last executed allocs, last allocation date).

        This writes an atomic JSON file under `self.state_file` so the process can
        be restarted without re-buying already-held positions.
        """
        os.makedirs(os.path.dirname(self.state_file) or '.', exist_ok=True)
        tmp = self.state_file + '.tmp'
        state = {
            'positions': {p: {'qty': pos.qty, 'entry_price': pos.entry_price, 'stop_price': pos.stop_price, 'take_price': pos.take_price} for p, pos in self.positions.items()},
            'last_executed_allocs': self._last_executed_allocs,
            'last_allocation_date': self._last_allocation_date,
        }
        with open(tmp, 'w') as f:
            json.dump(state, f)
        os.replace(tmp, self.state_file)

    def load_state(self):
        """Load saved state if available. Restores `positions`, `._last_executed_allocs`, and allocation date."""
        if not os.path.exists(self.state_file):
            return
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            posd = state.get('positions', {})
            self.positions = {}
            for p, info in posd.items():
                self.positions[p] = Position(pair=p, qty=float(info.get('qty', 0.0)), entry_price=info.get('entry_price'), stop_price=info.get('stop_price'), take_price=info.get('take_price'))
            self._last_executed_allocs = state.get('last_executed_allocs')
            self._last_allocation_date = state.get('last_allocation_date')
        except Exception:
            # ignore load errors and continue with empty state
            return

    def load_models(self, models_dir: str = "models"):
        """Load pickled models and metadata from disk if present.

        This will populate `self.models` and lightly restore `prices`/`volumes`
        from the metadata files where available.
        """
        if not os.path.isdir(models_dir):
            return
        for fname in os.listdir(models_dir):
            if not fname.endswith(".pkl"):
                continue
            pfile = os.path.join(models_dir, fname)
            safe = fname[:-4]
            pair = safe.replace("_", "/")
            try:
                with open(pfile, "rb") as f:
                    model = pickle.load(f)
                self.models[pair] = model
                # try to load metadata
                meta_file = os.path.join(models_dir, f"{safe}_meta.json")
                if os.path.exists(meta_file):
                    with open(meta_file, "r") as fm:
                        meta = json.load(fm)
                    prices_tail = meta.get("prices_tail") or []
                    vols_tail = meta.get("volumes_tail") or []
                    times_tail = meta.get("times_tail") or []
                    if prices_tail:
                        self.prices[pair] = list(map(float, prices_tail))
                    if vols_tail:
                        self.volumes[pair] = list(map(float, vols_tail))
                    if times_tail:
                        self.times[pair] = list(map(float, times_tail))
            except Exception:
                # ignore single-file errors
                continue
    def _get_trade_log_path(self) -> str:
        """Return current trade log path; rotate by UTC day when trade_log_file not set."""
        if self.trade_log_file:
            return self.trade_log_file
        # use UTC date for rotation
        date = datetime.datetime.utcnow().strftime("%Y%m%d")
        fname = f"trades-{date}.jsonl"
        return os.path.join(self.trade_log_dir, fname)

    def _write_trade(self, rec: Dict):
        """Append a single trade record (dict) as JSONL to the rotating log path."""
        path = self._get_trade_log_path()
        # ensure directory exists
        d = os.path.dirname(path) or "."
        os.makedirs(d, exist_ok=True)
        line = json.dumps(rec)
        # attempt to initialize last-written cache from file if unknown
        if self._last_written_line is None and os.path.exists(path):
            try:
                # read last ~8KB and take last non-empty line
                with open(path, "rb") as f:
                    f.seek(0, os.SEEK_END)
                    size = f.tell()
                    to_read = min(8192, size)
                    f.seek(size - to_read)
                    tail = f.read().decode(errors="ignore")
                    lines = [l for l in tail.splitlines() if l.strip()]
                    if lines:
                        self._last_written_line = lines[-1]
            except Exception:
                self._last_written_line = None

        # skip writing if identical to last written line (consecutive duplicate)
        if self._last_written_line == line:
            return

        with open(path, "a") as f:
            f.write(line + "\n")
        self._last_written_line = line
