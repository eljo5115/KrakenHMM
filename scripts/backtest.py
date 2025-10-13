#!/usr/bin/env python3
"""Simple backtester for KrakenHMM models.

This script fetches daily close & volume series (or reads from local CSVs),
fits per-pair HMM models using `kraken_hmm.hmm_model.HMMModel`, and runs a
simple backtest: each day it ranks assets by the trader's Sharpe-style score,
allocates equal capital to top-N, buys at close, and closes positions when
stop/take thresholds (derived from model state std) are hit.

Usage examples:
    # fetch historical data from Kraken REST (requires network + aiohttp)
    python scripts/backtest.py --pairs LINK/USD FIL/USD --days 365 --initial-capital 1000

    # run a quick local backtest using cached models/meta (no network)
    python scripts/backtest.py --use-models-only --pairs XBT/USD --days 60

This tool is intentionally small and easy to extend. It avoids placing any
live orders.
"""
import argparse
import asyncio
import time
import math
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

from kraken_hmm.hmm_model import HMMModel
from kraken_hmm.trader import Trader

# Try to import vectorbt for interactive backtests; fall back if unavailable
try:
    import vectorbt as vbt
    VBT_AVAILABLE = True
except Exception:
    VBT_AVAILABLE = False


async def fetch_hist(pairs: List[str], days: int = 365):
    try:
        from kraken_hmm.api import fetch_daily_close_volume

        print(f"Fetching {days} days of daily close data for: {pairs} from Kraken REST...")
        hist = await fetch_daily_close_volume(pairs, days=days)
        return hist
    except Exception as e:
        raise RuntimeError(f"Failed to fetch historical data: {e}")


def fit_models_from_hist(hist: Dict[str, Dict[str, List[float]]], n_states: int = 3) -> Dict[str, HMMModel]:
    models: Dict[str, HMMModel] = {}
    for p, series in hist.items():
        closes = series.get("close", [])
        vols = series.get("volume", [])
        if not closes or len(closes) < 10:
            continue
        model = HMMModel(n_states=n_states, random_state=42)
        try:
            model.fit(closes, volumes=vols if vols else None, sma_window=5)
            models[p] = model
        except Exception as e:
            print(f"Failed to fit model for {p}: {e}")
    return models


def run_backtest(hist: Dict[str, Dict[str, List[float]]], models: Dict[str, HMMModel], initial_capital: float = 1000.0, n_assets: int = 3):
    # Use Trader object to simulate the live trading logic more closely.
    pairs = sorted(models.keys())
    if not pairs:
        print("No models available for backtest")
        return

    lengths = [len(hist[p].get("close", [])) for p in pairs]
    T = min(lengths)
    print(f"Running Trader-driven walk-forward backtest for {len(pairs)} pairs over {T} days (no look-ahead)")

    # create and configure a Trader instance that will simulate live behavior
    trader = Trader(total_capital=initial_capital, n_assets=n_assets, n_states=3)
    trader.execute_orders = False
    trader.trade_log_dir = "logs"
    trader.min_history = 60  # let backtest seed models with a reasonable window
    trader.recent_window = 10

    # seed initial history up to init_window for each pair so models can be fit
    init_window = max(60, int(0.3 * T))
    for p in pairs:
        closes = hist[p]["close"][:init_window]
        vols = hist[p].get("volume")[:init_window]
        times = hist[p].get("time")[:init_window]
        try:
            trader.seed_history(p, closes, volumes=vols if vols else None, times=times if times else None)
        except Exception:
            pass

    # capital accounting for the replay (Trader does not maintain a changing cash balance)
    capital = float(initial_capital)
    trade_log = []

    # walk-forward day-by-day: add today's price for all pairs, let trader decide once per day
    for t in range(init_window, T):
        # add today's price tick for every pair (simulate end-of-day tick)
        for p in pairs:
            price = hist[p]["close"][t]
            vol = None
            if hist[p].get("volume"):
                vol = hist[p]["volume"][t]
            trader.add_price(p, price, volume=vol, ts=(hist[p].get("time", [None] * len(hist[p]["close"]) ) or [None])[t])

        # compute desired allocations and execute missing buys using Trader
        allocs = trader.allocate()
        # ensure trader uses requested managed set
        missing = {p: cap for p, cap in allocs.items() if p not in trader.positions}
        executed = {}
        try:
            # execute_allocations is async; run it to completion
            executed = asyncio.get_event_loop().run_until_complete(trader.execute_allocations(missing))
        except Exception:
            try:
                executed = asyncio.new_event_loop().run_until_complete(trader.execute_allocations(missing))
            except Exception:
                executed = {}

        # update replay capital & collect trade_log entries from executed mapping
        for p, cap in executed.items():
            # deduct allocated capital
            try:
                capital -= float(cap)
            except Exception:
                pass
            # the Trader.write_trade created persistent records; also mirror them in our in-memory trade_log
            # attempt to read last line from the trader's log file for this day if present
            # fallback: append a simple buy record
            trade_log.append({"ts": t, "type": "buy", "pair": p, "qty": trader.positions.get(p).qty if trader.positions.get(p) else None, "price": trader.prices.get(p)[-1]})

        # check stops/takes for open positions using today's price and close them if triggered
        for p, pos in list(trader.positions.items()):
            cur_price = trader.prices.get(p, [])[-1]
            if pos.stop_price is not None and cur_price <= pos.stop_price:
                reason = "stop"
            elif pos.take_price is not None and cur_price >= pos.take_price:
                reason = "take"
            else:
                reason = None
            if reason:
                # simulate sell: compute proceeds and update capital
                qty = pos.qty
                proceeds = qty * cur_price
                capital += proceeds
                pnl = None
                try:
                    if pos.entry_price is not None:
                        pnl = (cur_price - float(pos.entry_price)) * float(qty)
                except Exception:
                    pnl = None
                rec = {"ts": t, "type": "sell", "pair": p, "qty": qty, "price": cur_price, "reason": reason, "pnl": pnl}
                trade_log.append(rec)
                # remove position
                try:
                    trader.positions.pop(p, None)
                except Exception:
                    pass

        # online update models with data up to and including today's price (warm-start)
        for p in pairs:
            try:
                if p in trader.models:
                    trader.models[p].partial_update(hist[p]["close"][: t + 1], volumes=hist[p].get("volume")[: t + 1], sma_window=5, window=200, n_iter=10)
            except Exception:
                pass

    # at end, liquidate remaining positions at final prices
    final_prices = {p: hist[p]["close"][T - 1] for p in pairs}
    for p, pos in list(trader.positions.items()):
        exit_price = final_prices[p]
        proceeds = pos.qty * exit_price
        try:
            pnl = proceeds - (pos.qty * pos.entry_price)
        except Exception:
            pnl = None
        capital += proceeds
        trade_log.append({"ts": T - 1, "type": "sell", "pair": p, "qty": pos.qty, "price": exit_price, "reason": "liquidate", "pnl": pnl})

    total_return = (capital - initial_capital) / initial_capital
    print(f"Backtest (Trader-driven) complete. Initial capital={initial_capital:.2f} Final capital={capital:.2f} Return={total_return*100:.2f}%")
    # per-pair PnL
    per_pair = {}
    for rec in trade_log:
        p = rec.get("pair")
        per_pair.setdefault(p, 0.0)
        if rec.get("type") == "sell":
            per_pair[p] += float(rec.get("pnl", 0.0) or 0.0)

    print("Per-pair PnL:")
    for p, v in per_pair.items():
        print(f"  {p}: {v:.2f}")


def run_backtest_vectorbt(hist: Dict[str, Dict[str, List[float]]], models: Dict[str, HMMModel], initial_capital: float = 1000.0, n_assets: int = 3, out_html: Optional[str] = None):
    """Run a vectorbt backtest by constructing daily equal-weight allocations to top-N HMM-ranked pairs.

    This creates a price DataFrame aligned across pairs, builds a daily weights
    DataFrame where each day the top-N pairs get equal weight, and then uses
    vbt.Portfolio.from_weights to compute performance. Outputs an interactive
    HTML file if `out_html` is provided.
    """
    if not VBT_AVAILABLE:
        raise RuntimeError("vectorbt not available in the environment")

    # suppress noisy vectorbt/pandas aggregation warnings locally
    warnings.filterwarnings("ignore", "Object has multiple columns.*", category=UserWarning)
    warnings.filterwarnings("ignore", "Only one column is allowed.*", category=UserWarning)

    pairs = sorted(models.keys())
    if not pairs:
        raise RuntimeError("No models available for vectorbt backtest")

    # align histories to min length
    lengths = [len(hist[p].get("close", [])) for p in pairs]
    T = min(lengths)
    # build price DataFrame (oldest->newest)
    idx = None
    data = {}
    for p in pairs:
        closes = np.array(hist[p]["close"][-T:]).astype(float)
        times = hist[p].get("time") or []
        if times and len(times) >= T:
            idx = pd.to_datetime(np.array(times[-T:]), unit='s')
        data[p] = closes
    if idx is None:
        idx = pd.RangeIndex(start=0, stop=T, step=1)
    price_df = pd.DataFrame(data, index=idx)

    # build weights DataFrame by daily ranking
    weights = pd.DataFrame(0.0, index=price_df.index, columns=price_df.columns)
    for i, ts in enumerate(price_df.index):
        # compute per-pair HMM-based score using recent window ending at i
        scores = {}
        for p in pairs:
            model = models[p]
            # build recent window (up to 10 bars) from aligned price_df
            start = max(0, i - 9)
            recent = price_df[p].iloc[start : i + 1].dropna().values.tolist()
            if len(recent) < 2:
                continue
            try:
                state = model.predict_state(recent)
                means, stds = model.state_stats()
                if state is not None and state < len(means) and float(stds[state]) > 1e-8:
                    scores[p] = float(means[state]) / float(stds[state])
            except Exception:
                continue

        if not scores:
            continue
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top = [p for p, s in ranked[:n_assets]]
        if not top:
            continue
        w = 1.0 / len(top)
        for p in top:
            weights.at[ts, p] = w

    # create vectorbt portfolio - try multiple constructors to be compatible
    port = None
    # try from_weights if available
    if hasattr(vbt.Portfolio, 'from_weights'):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Object has multiple columns. Aggregating using")
                warnings.filterwarnings("ignore", message="Only one column is allowed")
                port = vbt.Portfolio.from_weights(price_df, weights, init_cash=initial_capital)
        except Exception:
            port = None

    # try from_orders: generate explicit signed per-day orders that mirror the
    # walk-forward rebalancer used in run_backtest so vectorbt sees the same
    # individual buy/sell events. We simulate cash and positions day-by-day and
    # record executed volumes (positive=buy, negative=sell) into `orders`.
    if port is None and hasattr(vbt.Portfolio, 'from_orders'):
        try:
            orders = pd.DataFrame(0.0, index=price_df.index, columns=price_df.columns)

            # simple deterministic rebalancer state to produce signed orders
            capital = float(initial_capital)
            positions = {p: 0.0 for p in price_df.columns}

            for ts in price_df.index:
                # current prices as floats
                day_prices = price_df.loc[ts].to_dict()

                # market value of current holdings
                mv = 0.0
                for p, qty in positions.items():
                    price = day_prices.get(p, np.nan)
                    if np.isnan(price) or price <= 0:
                        continue
                    mv += qty * float(price)

                portfolio_value = capital + mv

                # target set inferred from weights (weights already computed by ranking)
                top = [p for p in price_df.columns if float(weights.at[ts, p]) > 0]
                target_value = portfolio_value / len(top) if top else 0.0

                # compute adjustments and execute (deterministic, same rules as run_backtest)
                for p in price_df.columns:
                    price = day_prices.get(p, np.nan)
                    if np.isnan(price) or price <= 0:
                        continue
                    cur_val = positions.get(p, 0.0) * float(price)
                    desired = target_value if p in top else 0.0
                    diff = desired - cur_val
                    qty_diff = diff / float(price)
                    if abs(qty_diff) <= 1e-12:
                        continue

                    if qty_diff > 0:
                        # buy up to available capital
                        buy_qty = min(qty_diff, capital / float(price)) if float(price) > 0 else 0.0
                        if buy_qty <= 0:
                            continue
                        positions[p] = positions.get(p, 0.0) + buy_qty
                        capital -= buy_qty * float(price)
                        orders.at[ts, p] = float(orders.at[ts, p]) + buy_qty
                    else:
                        sell_qty = min(-qty_diff, positions.get(p, 0.0))
                        if sell_qty <= 0:
                            continue
                        positions[p] = positions.get(p, 0.0) - sell_qty
                        capital += sell_qty * float(price)
                        orders.at[ts, p] = float(orders.at[ts, p]) - sell_qty

            # when price_df has multiple columns, vectorbt treats each column as a
            # separate portfolio if init_cash is scalar. To mirror our single
            # pooled cash that the rebalancer used, split init_cash evenly per
            # column so the sum of per-column values approximates the pooled PV.
            try:
                ncols = max(1, len(price_df.columns))
                per_col_init = float(initial_capital) / float(ncols)
            except Exception:
                per_col_init = float(initial_capital)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Object has multiple columns. Aggregating using")
                warnings.filterwarnings("ignore", message="Only one column is allowed")
                port = vbt.Portfolio.from_orders(price_df, orders, init_cash=per_col_init)
            # persist orders for auditing
            try:
                ts0 = int(time.time())
                orders_file = f"logs/backtest-{ts0}-orders.csv"
                orders.to_csv(orders_file)
                print(f"Saved generated orders to {orders_file}")
            except Exception:
                pass

            # dump diagnostics about the vectorbt portfolio internals to a log
            try:
                dbg_lines = []
                dbg_lines.append(f"port type: {type(port)}")
                # port.stats end value if available
                try:
                    stats = port.stats()
                    dbg_lines.append(f"port.stats End Value: {stats.get('End Value') if isinstance(stats, dict) else stats}")
                except Exception as e:
                    dbg_lines.append(f"port.stats() failed: {e}")

                for attr in ('value', 'total_value', 'cash', 'cash_series'):
                    try:
                        if hasattr(port, attr):
                            v = getattr(port, attr)
                            dbg_lines.append(f"attr {attr} type: {type(v)}")
                            # attempt to materialize small sample
                            try:
                                sample = None
                                if callable(v):
                                    try:
                                        vv = v()
                                    except Exception:
                                        vv = v
                                else:
                                    vv = v
                                if isinstance(vv, pd.Series):
                                    sample = list(vv.iloc[:5].values)
                                elif isinstance(vv, pd.DataFrame):
                                    sample = {c: list(vv[c].iloc[:5].values) for c in vv.columns[:3]}
                                elif isinstance(vv, (list, tuple, np.ndarray)):
                                    sample = list(vv[:5])
                                else:
                                    sample = repr(vv)[:200]
                                dbg_lines.append(f"attr {attr} sample: {sample}")
                            except Exception as e:
                                dbg_lines.append(f"attr {attr} sample failed: {e}")
                        else:
                            dbg_lines.append(f"attr {attr} not present")
                    except Exception as e:
                        dbg_lines.append(f"error while inspecting {attr}: {e}")

                # orders/trades summary
                try:
                    if hasattr(port, 'orders'):
                        try:
                            n_orders = len(port.orders.records) if hasattr(port.orders, 'records') else None
                        except Exception:
                            n_orders = None
                        dbg_lines.append(f"port.orders type: {type(port.orders)}, records: {n_orders}")
                except Exception:
                    pass

                tsdbg = int(time.time())
                dbgfile = f"logs/backtest-{tsdbg}-vbt-debug.txt"
                with open(dbgfile, 'w') as f:
                    f.write('\n'.join(str(x) for x in dbg_lines))
                print(f"Saved vectorbt debug to {dbgfile}")
                # try to dump orders/trades readable records if available
                try:
                    if hasattr(port, 'orders') and hasattr(port.orders, 'records_readable'):
                        ords = port.orders.records_readable
                        df_ords = pd.DataFrame(ords)
                        ordfile = f"logs/backtest-{tsdbg}-vbt-orders.csv"
                        df_ords.to_csv(ordfile, index=False)
                        print(f"Saved vectorbt orders to {ordfile}")
                except Exception:
                    pass
                try:
                    if hasattr(port, 'trades') and hasattr(port.trades, 'records_readable'):
                        tr = port.trades.records_readable
                        df_tr = pd.DataFrame(tr)
                        trfile = f"logs/backtest-{tsdbg}-vbt-trades.csv"
                        df_tr.to_csv(trfile, index=False)
                        print(f"Saved vectorbt trades to {trfile}")
                except Exception:
                    pass
            except Exception:
                pass

            # compute replayed portfolio value from the same `orders` to compare
            try:
                def compute_pv_from_orders_local(orders_df: pd.DataFrame, price_df: pd.DataFrame, init_cash: float):
                    pos = {p: 0.0 for p in price_df.columns}
                    cash = float(init_cash)
                    vals = []
                    for idx in price_df.index:
                        if idx in orders_df.index:
                            row = orders_df.loc[idx]
                            for p in price_df.columns:
                                try:
                                    q = float(row.get(p, 0.0))
                                except Exception:
                                    q = 0.0
                                if abs(q) <= 1e-12:
                                    continue
                                price = float(price_df.at[idx, p])
                                pos[p] = pos.get(p, 0.0) + q
                                cash -= q * price
                        mv = 0.0
                        for p in price_df.columns:
                            try:
                                price = float(price_df.at[idx, p])
                            except Exception:
                                price = np.nan
                            if price is None or np.isnan(price):
                                continue
                            mv += pos.get(p, 0.0) * price
                        vals.append(cash + mv)
                    return pd.Series(vals, index=price_df.index)

                pv_replay = compute_pv_from_orders_local(orders, price_df, float(initial_capital))
                # try to extract vectorbt's value series
                vbt_series = None
                try:
                    if hasattr(port, 'value'):
                        candidate = getattr(port, 'value')
                        if callable(candidate):
                            try:
                                candidate = candidate()
                            except Exception:
                                pass
                        # If candidate is a DataFrame with per-asset columns, sum to total value
                        if isinstance(candidate, pd.DataFrame):
                            try:
                                # Sum per-asset columns to produce total portfolio value
                                vbt_series = candidate.sum(axis=1)
                            except Exception:
                                vbt_series = None
                        elif isinstance(candidate, dict):
                            try:
                                # dict of series/arrays per asset -> build DataFrame then sum
                                df_cand = pd.DataFrame(candidate)
                                vbt_series = df_cand.sum(axis=1)
                            except Exception:
                                vbt_series = None
                        elif isinstance(candidate, pd.Series):
                            vbt_series = candidate
                        elif isinstance(candidate, (np.ndarray, list, tuple)) and len(candidate) == len(price_df.index):
                            vbt_series = pd.Series(list(candidate), index=price_df.index)
                except Exception:
                    vbt_series = None

                vbt_end = None
                replay_end = None
                if vbt_series is not None:
                    try:
                        vbt_end = float(vbt_series.iloc[-1])
                    except Exception:
                        vbt_end = None
                try:
                    replay_end = float(pv_replay.iloc[-1])
                except Exception:
                    replay_end = None

                print(f"Vectorbt end value (extracted): {vbt_end}")
                print(f"Replay end value (from orders): {replay_end}")

                # choose the series to plot: prefer vectorbt aggregation when available
                try:
                    if vbt_series is not None:
                        pv_for_plot = vbt_series
                    else:
                        pv_for_plot = pv_replay
                except Exception:
                    pv_for_plot = pv_replay

                # save per-day differences for debugging
                try:
                    ts1 = int(time.time())
                    difffile = f"logs/backtest-{ts1}-pv-diff.csv"
                    df_diff = pd.DataFrame({'price_df_idx': price_df.index})
                    if vbt_series is not None:
                        df_diff['vbt_value'] = list(vbt_series.values)
                    df_diff['replay_value'] = list(pv_replay.values)
                    if vbt_series is not None:
                        df_diff['diff'] = df_diff['vbt_value'] - df_diff['replay_value']
                    df_diff.to_csv(difffile, index=False)
                    print(f"Saved PV diff to {difffile}")
                except Exception:
                    pass
            except Exception:
                pass
        except Exception:
            port = None

    # fallback: from_signals by setting entry when weight>0 and exit when weight==0
    if port is None and hasattr(vbt.Portfolio, 'from_signals'):
        try:
            entries = weights > 0
            exits = weights.shift(-1).fillna(False) == 0
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Object has multiple columns. Aggregating using")
                warnings.filterwarnings("ignore", message="Only one column is allowed")
                port = vbt.Portfolio.from_signals(price_df, entries, exits, init_cash=initial_capital)
        except Exception:
            port = None

    if port is None:
        raise RuntimeError("Failed to construct vectorbt Portfolio with available constructors")

    # save interactive plot to HTML using plotly
    if out_html:
        try:
            fig = port.plot()
            fig.write_html(out_html)
            print(f"Saved interactive portfolio plot to {out_html}")
        except Exception as e:
            print("Failed to write HTML plot:", e)

    # compare expected orders count from the orders DataFrame (if present)
    try:
        if 'orders' in locals() and orders is not None:
            expected = int((orders.abs() > 1e-12).sum().sum())
            actual = None
            # try a few ways to get vectorbt's recorded orders/trades
            try:
                if hasattr(port, 'orders') and hasattr(port.orders, 'records_readable'):
                    actual = len(port.orders.records_readable)
                elif hasattr(port, 'orders') and hasattr(port.orders, 'records'):
                    actual = len(port.orders.records)
            except Exception:
                actual = None
            if actual is None:
                try:
                    if hasattr(port, 'trades') and hasattr(port.trades, 'records_readable'):
                        actual = len(port.trades.records_readable)
                    elif hasattr(port, 'trades') and hasattr(port.trades, 'records'):
                        actual = len(port.trades.records)
                except Exception:
                    actual = None

            print(f"Orders generated (nonzero cells): {expected}")
            if actual is not None:
                print(f"Vectorbt recorded orders/trades: {actual}")
            else:
                print("Could not determine recorded orders/trades from vectorbt object")
    except Exception:
        pass

    # create a custom HTML report: multi-asset price chart with buy/sell markers
    try:
        if 'orders' in locals() and orders is not None:
            ts1 = int(time.time())
            report_file = f"logs/backtest-{ts1}-report.html"

            # compute portfolio value by replaying orders against price_df to ensure
            # the plotted end value matches the orders we generated and passed to
            # vectorbt. This avoids any aggregation/column-selection differences
            # from vectorbt's internal `value` representation.
            pv_series = None
            try:
                def compute_pv_from_orders(orders_df: pd.DataFrame, price_df: pd.DataFrame, init_cash: float):
                    # positions in base asset units
                    pos = {p: 0.0 for p in price_df.columns}
                    cash = float(init_cash)
                    vals = []
                    for idx in price_df.index:
                        # apply orders at this timestamp if present
                        if idx in orders_df.index:
                            row = orders_df.loc[idx]
                            for p in price_df.columns:
                                try:
                                    q = float(row.get(p, 0.0))
                                except Exception:
                                    q = 0.0
                                if abs(q) <= 1e-12:
                                    continue
                                price = float(price_df.at[idx, p])
                                # update position and cash (buys are +q, sells are -q)
                                pos[p] = pos.get(p, 0.0) + q
                                cash -= q * price
                        # compute total value
                        mv = 0.0
                        for p in price_df.columns:
                            try:
                                price = float(price_df.at[idx, p])
                            except Exception:
                                price = np.nan
                            if price is None or np.isnan(price):
                                continue
                            mv += pos.get(p, 0.0) * price
                        vals.append(cash + mv)
                    return pd.Series(vals, index=price_df.index)

                pv_series = compute_pv_from_orders(orders, price_df, float(initial_capital))
            except Exception:
                pv_series = None

            # build subplot: prices (top) and portfolio value (bottom if available)
            rows = 2 if pv_series is not None else 1
            specs = [[{"secondary_y": False}], [{"secondary_y": False}]] if rows == 2 else [[{"secondary_y": False}]]
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.06, specs=specs)

            # add price traces
            for col in price_df.columns:
                fig.add_trace(go.Scatter(x=price_df.index, y=price_df[col], mode='lines', name=str(col)), row=1, col=1)

            # add buy/sell markers from orders DataFrame — collect markers with hover text
            orders_nonzero = orders.stack().reset_index()
            orders_nonzero.columns = ['time', 'pair', 'qty']
            markers = []
            for _, r in orders_nonzero.iterrows():
                t = r['time']
                p = r['pair']
                q = float(r['qty'])
                side = 'BUY' if q > 0 else 'SELL'
                if p not in price_df.columns:
                    continue
                try:
                    x = t
                    y = float(price_df.at[x, p])
                except Exception:
                    try:
                        x = price_df.index[int(t)]
                        y = float(price_df.iloc[int(t)][p])
                    except Exception:
                        continue
                msize = 8 + min(20, abs(q) * 5 if abs(q) < 10 else 20)
                hover = f"{side} {p}<br>Qty: {q:.6g}<br>Price: {y:.6g}<br>Time: {x}"
                markers.append({"x": x, "y": y, "pair": p, "side": side, "qty": q, "size": msize, "hover": hover})

            # Build table rows for bottom table (include trade value = qty * price)
            table_rows = []
            try:
                tbl = orders_nonzero.copy()
                tbl['side'] = tbl['qty'].apply(lambda q: 'BUY' if float(q) > 0 else 'SELL')
                tbl['qty_abs'] = tbl['qty'].abs()
                # compute price at time and trade value
                vals = []
                for _, r2 in tbl.iterrows():
                    t2 = r2['time']
                    p2 = r2['pair']
                    q2 = float(r2['qty'])
                    # attempt to lookup price
                    price2 = None
                    try:
                        price2 = float(price_df.at[t2, p2])
                    except Exception:
                        try:
                            price2 = float(price_df.iloc[int(t2)][p2])
                        except Exception:
                            price2 = None
                    if price2 is None or (isinstance(price2, float) and (np.isnan(price2) or price2 == 0)):
                        trade_value = ''
                    else:
                        trade_value = f"{abs(q2 * price2):.6g}"
                    vals.append(trade_value)
                tbl['trade_value'] = vals
                # ensure time formatting
                try:
                    if len(price_df.index) and isinstance(price_df.index[0], pd.Timestamp):
                        tbl['time'] = tbl['time'].apply(lambda t: pd.to_datetime(t, unit='s') if isinstance(t, (int, float)) else t)
                except Exception:
                    pass
                table_rows = tbl[['time', 'pair', 'side', 'qty_abs', 'trade_value']].astype(str).values.tolist()
            except Exception:
                table_rows = []

            # create subplots: if pv_series exists we want 3 rows (price, pv, table), else 2 rows (price, table)
            try:
                if pv_series is not None:
                    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03, specs=[[{"secondary_y": False}], [{"secondary_y": False}], [{"type": "table"}]])
                    # add price traces to row 1
                    for col in price_df.columns:
                        fig.add_trace(go.Scatter(x=price_df.index, y=price_df[col], mode='lines', name=str(col)), row=1, col=1)
                    # add markers to row 1
                    for m in markers:
                        symbol = 'triangle-up' if m['side'] == 'BUY' else 'triangle-down'
                        color = 'green' if m['side'] == 'BUY' else 'red'
                        fig.add_trace(go.Scatter(x=[m['x']], y=[m['y']], mode='markers', marker=dict(symbol=symbol, color=color, size=m['size']), hovertemplate=m['hover'], showlegend=False), row=1, col=1)
                    # add pv traces to row 2
                    fig.add_trace(go.Scatter(x=pv_series.index, y=pv_series.values, mode='lines', name='Replay PV', line=dict(color='blue')), row=2, col=1)
                    try:
                        if 'vbt_series' in locals() and vbt_series is not None:
                            fig.add_trace(go.Scatter(x=vbt_series.index, y=vbt_series.values, mode='lines', name='VectorBT PV (agg)', line=dict(color='black', dash='dash')), row=2, col=1)
                    except Exception:
                        pass
                    # add table to row 3
                    if table_rows:
                        header = dict(values=['Time', 'Pair', 'Side', 'Qty', 'Value'], fill_color='paleturquoise', align='left')
                        cell = dict(values=list(zip(*table_rows)), fill_color='white', align='left')
                        fig.add_trace(go.Table(header=header, cells=cell), row=3, col=1)
                else:
                    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, specs=[[{"secondary_y": False}], [{"type": "table"}]])
                    for col in price_df.columns:
                        fig.add_trace(go.Scatter(x=price_df.index, y=price_df[col], mode='lines', name=str(col)), row=1, col=1)
                    for m in markers:
                        symbol = 'triangle-up' if m['side'] == 'BUY' else 'triangle-down'
                        color = 'green' if m['side'] == 'BUY' else 'red'
                        fig.add_trace(go.Scatter(x=[m['x']], y=[m['y']], mode='markers', marker=dict(symbol=symbol, color=color, size=m['size']), hovertemplate=m['hover'], showlegend=False), row=1, col=1)
                    if table_rows:
                        header = dict(values=['Time', 'Pair', 'Side', 'Qty', 'Value'], fill_color='paleturquoise', align='left')
                        cell = dict(values=list(zip(*table_rows)), fill_color='white', align='left')
                        fig.add_trace(go.Table(header=header, cells=cell), row=2, col=1)
            except Exception:
                # fallback: use original fig if something failed
                pass

            fig.update_layout(title_text='Backtest: prices with buy/sell markers', height=700)
            # annotate with vectorbt's End Value (from port.stats) so the chart
            # explicitly shows the same final value that vectorbt reports on CLI.
            try:
                vbt_end_val = None
                try:
                    stats = port.stats()
                    # stats may be a Series-like; try to parse End Value
                    if isinstance(stats, dict):
                        vbt_end_val = stats.get('End Value')
                    else:
                        try:
                            vbt_end_val = stats.get('End Value') if hasattr(stats, 'get') else None
                        except Exception:
                            vbt_end_val = None
                except Exception:
                    vbt_end_val = None

                if vbt_end_val is not None:
                    try:
                        vbt_end_val = float(vbt_end_val)
                        # if portfolio subplot exists, add hline there, else add to price subplot
                        annot_y = vbt_end_val
                        if rows == 2:
                            fig.add_hline(y=vbt_end_val, line_dash='dash', line_color='black', row=2, col=1)
                            fig.add_annotation(text=f"Vectorbt End Value: {vbt_end_val:.2f}", xref='paper', x=0.99, yref='y domain', y=0.01, showarrow=False, row=2, col=1)
                        else:
                            # place annotation on top plot
                            fig.add_hline(y=vbt_end_val, line_dash='dash', line_color='black', row=1, col=1)
                            fig.add_annotation(text=f"Vectorbt End Value: {vbt_end_val:.2f}", xref='paper', x=0.99, yref='y domain', y=0.01, showarrow=False, row=1, col=1)
                    except Exception:
                        pass
                # annotate replay end value as well
                try:
                    replay_end_val = float(pv_series.iloc[-1])
                    if rows == 2:
                        fig.add_annotation(text=f"Replay End Value: {replay_end_val:.2f}", xref='paper', x=0.01, yref='y domain', y=0.01, showarrow=False, row=2, col=1)
                    else:
                        fig.add_annotation(text=f"Replay End Value: {replay_end_val:.2f}", xref='paper', x=0.01, yref='y domain', y=0.01, showarrow=False, row=1, col=1)
                except Exception:
                    pass

                try:
                    # adjust x-axis range to focus on traded date range +/- 2 days (or +/-2 index units for numeric index)
                    try:
                        trade_idx = None
                        if len(markers) > 0:
                            xs = [m['x'] for m in markers]
                            try:
                                xs_dt = pd.to_datetime(xs)
                                trade_idx = xs_dt
                            except Exception:
                                trade_idx = xs
                        else:
                            try:
                                trade_idx = pd.to_datetime(orders_nonzero['time'])
                            except Exception:
                                trade_idx = list(orders_nonzero['time']) if len(orders_nonzero) > 0 else None

                        if trade_idx is not None and len(trade_idx) > 0:
                            if isinstance(trade_idx[0], pd.Timestamp) or pd.api.types.is_datetime64_any_dtype(getattr(trade_idx, 'dtype', None)):
                                xmin = pd.to_datetime(trade_idx).min()
                                xmax = pd.to_datetime(trade_idx).max()
                                margin = pd.Timedelta(days=2)
                                fig.update_xaxes(range=[xmin - margin, xmax + margin])
                            else:
                                xmin = min(trade_idx)
                                xmax = max(trade_idx)
                                margin = 2
                                fig.update_xaxes(range=[xmin - margin, xmax + margin])
                    except Exception:
                        pass
                    fig.write_html(report_file)
                    print(f"Saved HTML report to {report_file}")
                except Exception as e:
                    print("Failed to write custom HTML report:", e)
            except Exception as e:
                print("Error annotating report with vectorbt end value:", e)
    except Exception:
        pass

    # return portfolio and data for further inspection
    return port, price_df, weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", help="Pairs to backtest", default=["XBT/USD", "ETH/USD", "LINK/USD", "FIL/USD"]) 
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--n-assets", type=int, default=3)
    parser.add_argument("--use-models-only", action="store_true", help="Do not fetch data from Kraken; require local model fits in models_dir/meta")
    args = parser.parse_args()

    pairs = args.pairs
    days = args.days

    loop = asyncio.get_event_loop()
    use_vbt = VBT_AVAILABLE and not args.use_models_only
    out_html = None
    if use_vbt:
        ts = int(time.time())
        out_html = f"logs/backtest-{ts}.html"

    if args.use_models_only:
        # attempt to load models from models/<PAIR>_meta.json for prices
        hist = {}
        for p in pairs:
            safe = p.replace("/", "_")
            try:
                import json, os
                meta_file = os.path.join("models", f"{safe}_meta.json")
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                closes = meta.get("prices_tail", [])
                times = meta.get("times_tail", [])
                vols = meta.get("volumes_tail", [])
                hist[p] = {"time": times, "close": closes, "volume": vols}
            except Exception as e:
                print(f"Failed to load meta for {p}: {e}")
        models = fit_models_from_hist(hist)
        if use_vbt:
            try:
                port, price_df, weights = run_backtest_vectorbt(hist, models, initial_capital=args.initial_capital, n_assets=args.n_assets, out_html=out_html)
                print(port.stats())
            except Exception as e:
                print("vectorbt path failed, falling back:", e)
                run_backtest(hist, models, initial_capital=args.initial_capital, n_assets=args.n_assets)
        else:
            run_backtest(hist, models, initial_capital=args.initial_capital, n_assets=args.n_assets)
    else:
        hist = loop.run_until_complete(fetch_hist(pairs, days=days))
        models = fit_models_from_hist(hist)
        if use_vbt:
            try:
                port, price_df, weights = run_backtest_vectorbt(hist, models, initial_capital=args.initial_capital, n_assets=args.n_assets, out_html=out_html)
                print(port.stats())
            except Exception as e:
                print("vectorbt path failed, falling back:", e)
                run_backtest(hist, models, initial_capital=args.initial_capital, n_assets=args.n_assets)
        else:
            run_backtest(hist, models, initial_capital=args.initial_capital, n_assets=args.n_assets)


if __name__ == "__main__":
    main()
