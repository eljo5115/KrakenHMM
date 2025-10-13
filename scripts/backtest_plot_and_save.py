#!/usr/bin/env python3
"""Backtest runner that plots results and saves final HMM models.

This script will attempt to fetch historical daily close series from Kraken.
If network or dependencies are unavailable it will fall back to loading the
small `*_meta.json` tails from the `models/` directory (best-effort).

Artifacts produced:
 - plots saved to `logs/backtest-<timestamp>-equity.png` and `...-pnl.png`
 - fitted HMM pickles and meta saved to `models/<SAFE>.pkl` and `models/<SAFE>_meta.json`

"""
import argparse
import asyncio
import os
import time
import json
import pickle
from typing import List, Dict

import numpy as np

from kraken_hmm.hmm_model import HMMModel


async def fetch_hist(pairs: List[str], days: int = 365):
    try:
        from kraken_hmm.api import fetch_daily_close_volume

        print(f"Fetching {days} days of daily data for: {pairs}...")
        hist = await fetch_daily_close_volume(pairs, days=days)
        return hist
    except Exception as e:
        print("Failed to fetch history from Kraken REST (will try models meta):", e)
        return None


def load_hist_from_meta(pairs: List[str]):
    hist = {}
    for p in pairs:
        safe = p.replace("/", "_")
        meta_file = os.path.join("models", f"{safe}_meta.json")
        if not os.path.exists(meta_file):
            print(f"No meta for {p} at {meta_file}")
            continue
        try:
            with open(meta_file, "r") as f:
                meta = json.load(f)
            closes = meta.get("prices_tail", [])
            vols = meta.get("volumes_tail", [])
            times = meta.get("times_tail", [])
            hist[p] = {"time": times, "close": closes, "volume": vols}
        except Exception as e:
            print(f"Failed to read meta for {p}: {e}")
    return hist


def fit_models(hist: Dict[str, Dict], n_states: int = 3) -> Dict[str, HMMModel]:
    models = {}
    for p, series in hist.items():
        closes = series.get("close", [])
        vols = series.get("volume", [])
        if not closes or len(closes) < 3:
            print(f"Skipping {p}: insufficient history ({len(closes)})")
            continue
        model = HMMModel(n_states=n_states, random_state=42)
        try:
            model.fit(closes, volumes=(vols if vols else None), sma_window=min(10, max(1, len(closes)//10)))
            models[p] = model
            print(f"Fitted model for {p}")
        except Exception as e:
            print(f"Failed to fit model for {p}: {e}")
    return models


def run_simulation(hist: Dict[str, Dict], models: Dict[str, HMMModel], initial_capital: float = 1000.0, n_assets: int = 3):
    # align lengths
    pairs = sorted(models.keys())
    if not pairs:
        raise RuntimeError("No models to simulate")
    lengths = [len(hist[p]["close"]) for p in pairs]
    T = min(lengths)
    print(f"Simulating {len(pairs)} pairs for {T} days")

    capital = initial_capital
    equity_curve = []
    positions = {}
    trade_log = []

    for t in range(T):
        # price snapshot
        prices = {p: hist[p]["close"][t] for p in pairs}

        # compute per-pair sharpe-like from HMM
        scores = {}
        for p in pairs:
            model = models[p]
            recent_start = max(0, t - 9)
            recent = hist[p]["close"][recent_start:t+1]
            try:
                state = model.predict_state(recent)
                means, stds = model.state_stats()
                if state is not None and state < len(means) and stds[state] > 1e-8:
                    scores[p] = float(means[state]) / float(stds[state])
            except Exception:
                continue

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top = [p for p, s in ranked[:n_assets]]

        # allocate equally among top that we don't already hold
        new_buys = [p for p in top if p not in positions]
        if new_buys:
            per = capital / len(new_buys)
            for p in new_buys:
                price = prices[p]
                qty = per / price if price and price > 0 else 0.0
                # determine stop/take
                stop = None
                take = None
                try:
                    state = models[p].predict_state(hist[p]["close"][max(0, t-9):t+1])
                    means, stds = models[p].state_stats()
                    if state is not None and state < len(stds):
                        s = max(1e-8, float(stds[state]))
                        stop = price * (1.0 - s)
                        take = price * (1.0 + 2.0 * s)
                except Exception:
                    pass
                positions[p] = {"qty": qty, "entry_price": price, "stop": stop, "take": take}
                trade_log.append({"ts": t, "type": "buy", "pair": p, "qty": qty, "price": price})
                capital -= qty * price

        # check stops/takes
        to_close = []
        for p, pos in list(positions.items()):
            price = prices[p]
            if pos.get("stop") is not None and price <= pos["stop"]:
                to_close.append((p, "stop"))
            elif pos.get("take") is not None and price >= pos["take"]:
                to_close.append((p, "take"))

        for p, reason in to_close:
            pos = positions.pop(p)
            exit_price = prices[p]
            proceeds = pos["qty"] * exit_price
            pnl = proceeds - (pos["qty"] * pos["entry_price"])
            capital += proceeds
            trade_log.append({"ts": t, "type": "sell", "pair": p, "qty": pos["qty"], "price": exit_price, "reason": reason, "pnl": pnl})

        # compute current equity (cash + mark-to-market of positions)
        mv = sum(pos["qty"] * prices[p] for p, pos in positions.items())
        equity = capital + mv
        equity_curve.append(equity)

    # liquidate remaining positions at final price
    final_prices = {p: hist[p]["close"][T-1] for p in pairs}
    for p, pos in list(positions.items()):
        exit_price = final_prices[p]
        proceeds = pos["qty"] * exit_price
        pnl = proceeds - (pos["qty"] * pos["entry_price"])
        capital += proceeds
        trade_log.append({"ts": T-1, "type": "sell", "pair": p, "qty": pos["qty"], "price": exit_price, "reason": "liquidate", "pnl": pnl})

    return equity_curve, trade_log, models


def save_models_and_meta(models: Dict[str, HMMModel], hist: Dict[str, Dict], out_dir: str = "models"):
    os.makedirs(out_dir, exist_ok=True)
    for p, model in models.items():
        safe = p.replace("/", "_")
        pfile = os.path.join(out_dir, f"{safe}.pkl")
        meta_file = os.path.join(out_dir, f"{safe}_meta.json")
        try:
            with open(pfile, "wb") as f:
                pickle.dump(model, f)
            meta = {
                "pair": p,
                "history_len": len(hist.get(p, {}).get("close", [])),
                "timestamp": int(time.time()),
                "prices_tail": hist.get(p, {}).get("close", [])[-50:],
                "volumes_tail": hist.get(p, {}).get("volume", [])[-50:],
            }
            # include state stats
            try:
                means, stds = model.state_stats()
                meta["state_means"] = [float(x) for x in means]
                meta["state_stds"] = [float(x) for x in stds]
            except Exception:
                pass
            with open(meta_file, "w") as f:
                json.dump(meta, f)
            print(f"Saved model+meta for {p} -> {pfile}, {meta_file}")
        except Exception as e:
            print(f"Failed to save model for {p}: {e}")


def plot_results(equity_curve, trade_log, out_prefix: str):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print("matplotlib not available, skipping plots:", e)
        return

    os.makedirs(os.path.dirname(out_prefix) or '.', exist_ok=True)
    # equity curve
    plt.figure(figsize=(10, 5))
    plt.plot(equity_curve)
    plt.title("Equity Curve")
    plt.xlabel("Day")
    plt.ylabel("Equity")
    eq_file = f"{out_prefix}-equity.png"
    plt.savefig(eq_file)
    plt.close()
    print("Saved equity plot:", eq_file)

    # per-pair cumulative PnL over time
    # build per-pair cumulative
    per_pair = {}
    for rec in trade_log:
        if rec.get("type") == "sell":
            p = rec.get("pair")
            per_pair.setdefault(p, 0.0)
            per_pair[p] += float(rec.get("pnl", 0.0) or 0.0)

    if per_pair:
        plt.figure(figsize=(10, 5))
        pairs = list(per_pair.keys())
        vals = [per_pair[p] for p in pairs]
        plt.bar(pairs, vals)
        plt.title("Per-pair cumulative PnL")
        plt.ylabel("PnL")
        pnl_file = f"{out_prefix}-pnl.png"
        plt.savefig(pnl_file)
        plt.close()
        print("Saved per-pair PnL plot:", pnl_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", nargs="+", default=["XBT/USD", "LINK/USD", "FIL/USD"], help="Pairs to backtest")
    parser.add_argument("--days", type=int, default=365)
    parser.add_argument("--initial-capital", type=float, default=1000.0)
    parser.add_argument("--n-assets", type=int, default=3)
    parser.add_argument("--out-prefix", type=str, default=None, help="Output prefix for plots (defaults to logs/backtest-<ts>)")
    args = parser.parse_args()

    pairs = args.pairs
    days = args.days
    loop = asyncio.get_event_loop()

    hist = loop.run_until_complete(fetch_hist(pairs, days=days))
    if hist is None:
        hist = load_hist_from_meta(pairs)
        if not hist:
            raise SystemExit("No historical data available (neither fetch nor models meta)")

    models = fit_models(hist)
    equity_curve, trade_log, models = run_simulation(hist, models, initial_capital=args.initial_capital, n_assets=args.n_assets)

    ts = int(time.time())
    out_prefix = args.out_prefix or f"logs/backtest-{ts}"
    plot_results(equity_curve, trade_log, out_prefix)

    # save final models & meta
    save_models_and_meta(models, hist)

    print("Backtest finished. Final equity:", equity_curve[-1] if equity_curve else args.initial_capital)


if __name__ == "__main__":
    main()
