"""Top-level runner for KrakenHMM trader.

This file is meant to be run directly from the project root:

    python run_trader.py

Heavy imports (aiohttp, numpy, hmmlearn) are performed inside main so the
module can be imported for quick checks without those dependencies installed.
"""

import argparse
import asyncio
import signal
from typing import List
import time
try:
    from aiohttp import web
except Exception:
    web = None


PAIRS: List[str] = [
    "XBT/USD",
    "ETH/USD",
    "LTC/USD",
    "XRP/USD",
    "XMR/USD",
    "BCH/USD",
    "ADA/USD",
    "DOT/USD",
    "LINK/USD",
    "SOL/USD",
    "UNI/USD",
    "AAVE/USD",
    "MKR/USD",
    "COMP/USD",
    "SUSHI/USD",
    "ALGO/USD",
    "ATOM/USD",
    "FIL/USD",
    "TRX/USD",
    "EOS/USD",
    "XTZ/USD",
]


async def main(total_capital: float = 1000.0, n_assets: int = 5, debug: bool = False, simulate: bool = False, max_ticks: int = 0):
    # Delay heavy imports until runtime so module import is lightweight
    try:
        from kraken_hmm.api import ticker_stream
    except Exception:
        ticker_stream = None
    from kraken_hmm.trader import Trader

    trader = Trader(total_capital=total_capital, n_assets=n_assets)
    # allow runner to configure persistent state file
    trader.state_file = getattr(main, "state_file", trader.state_file)
    # restore previously persisted positions/state if present
    try:
        trader.load_state()
    except Exception:
        pass
    # configure order execution flags on the trader instance (set via CLI attrs on main)
    trader.execute_orders = getattr(main, "execute_orders", False)
    trader.api_key = getattr(main, "api_key", None)
    trader.api_secret = getattr(main, "api_secret", None)
    # trade log location
    trader.trade_log_dir = getattr(main, "trade_log_dir", ".")
    trader.trade_log_file = getattr(main, "trade_log_file", None)

    # Environment fallback for production secrets/flags (safer than exposing on CLI)
    import os

    if not trader.api_key:
        trader.api_key = os.environ.get("KRAKEN_API_KEY")
    if not trader.api_secret:
        trader.api_secret = os.environ.get("KRAKEN_API_SECRET")
    # allow enabling live execution via environment (values: 1/true/yes)
    if not trader.execute_orders:
        exec_env = os.environ.get("EXECUTE_ORDERS") or os.environ.get("RUN_TRADER_EXECUTE_ORDERS")
        if exec_env and str(exec_env).lower() in ("1", "true", "yes", "on"):
            trader.execute_orders = True
    # allow total capital override via env
    if getattr(main, "total_capital", None) is None:
        try:
            env_cap = os.environ.get("TOTAL_CAPITAL")
            if env_cap is not None:
                trader.total_capital = float(env_cap)
        except Exception:
            pass

    # optional seeding from REST OHLC data to warm up HMMs (one-time at startup)
    seed_history = getattr(main, "seed_history", False)
    seed_days = getattr(main, "seed_days", 365)
    models_dir = getattr(main, "models_dir", "models")
    # track whether we've printed an initial debug snapshot (avoid repeating it every tick)
    initial_debug_printed = False
    if seed_history:
        try:
            from kraken_hmm.api import fetch_daily_close_volume

            print(f"Seeding history from REST for past {seed_days} days...")
            hist = await fetch_daily_close_volume(PAIRS, days=seed_days)
            for p, series in hist.items():
                closes = series.get("close", [])
                vols = series.get("volume", [])
                if closes:
                    # seed via Trader helper to ensure alignment & fitting
                    trader.seed_history(p, closes, vols)
            print("Seeding complete. Fitting models...")
            # fit models now that all histories are seeded
            for p in list(trader.prices.keys()):
                try:
                    trader.ensure_model(p)
                    vols = trader.volumes.get(p)
                    trader.models[p].fit(trader.prices[p][-seed_days:], volumes=(vols[-seed_days:] if vols else None), sma_window=min(10, trader.recent_window))
                except Exception:
                    trader.models.pop(p, None)
            # persist fitted models
            trader.save_models(models_dir=models_dir)
            print(f"Models saved to {models_dir}")
            # If debug mode is enabled, print one initial HMM diagnostic snapshot here
            if debug:
                diagnostics = []
                for p in sorted(trader.prices.keys()):
                    history_len = len(trader.prices.get(p, []))
                    has_model = p in trader.models
                    print(f"pair={p} history={history_len} model={'yes' if has_model else 'no'}")

                for p, model in trader.models.items():
                    prices = trader.prices.get(p, [])
                    try:
                        means, stds = model.state_stats()
                        state = None
                        if len(prices) >= trader.recent_window + 1:
                            try:
                                vols = trader.volumes.get(p)
                                recent_vols = vols[-trader.recent_window:] if vols is not None else None
                                state = model.predict_state(prices[-trader.recent_window:], recent_volumes=recent_vols)
                            except Exception:
                                state = None
                        sharpe = trader.compute_sharpe_for_pair(p)
                        diagnostics.append((p, sharpe, state, means, stds))
                    except Exception:
                        continue

                if diagnostics:
                    diagnostics.sort(key=lambda x: (x[1] is None, x[1] if x[1] is not None else -999.0), reverse=True)
                    print("\n--- HMM Initial Debug Snapshot (one-time) ---")
                    for p, score, state, means, stds in diagnostics:
                        score_s = f"{score:.6f}" if score is not None else "n/a"
                        s = f"pair={p} sharpe={score_s}"
                        if state is not None:
                            s += f" predicted_state={state}"
                        s += f" means={['{:.4e}'.format(m) for m in means]} stds={['{:.4e}'.format(sv) for sv in stds]}"
                        print(s)
                initial_debug_printed = True
        except Exception as e:
            print("Failed to seed history from REST:", e)

        # start a background daily retrain task (re-fetch yearly data and re-fit at midnight)
        async def daily_retrain_task():
            import datetime

            async def sleep_until_midnight():
                now = datetime.datetime.now()
                tomorrow = now + datetime.timedelta(days=1)
                midnight = datetime.datetime(year=tomorrow.year, month=tomorrow.month, day=tomorrow.day)
                secs = (midnight - now).total_seconds()
                await asyncio.sleep(secs)

            while True:
                await sleep_until_midnight()
                try:
                    print("Daily retrain: fetching latest yearly data and refitting models...")
                    hist = await fetch_daily_close_volume(PAIRS, days=seed_days)
                    for p, series in hist.items():
                        closes = series.get("close", [])
                        vols = series.get("volume", [])
                        if closes:
                            # replace history and re-fit
                            trader.seed_history(p, closes, vols)
                            trader.ensure_model(p)
                            vols = trader.volumes.get(p)
                            trader.models[p].fit(trader.prices[p][-seed_days:], volumes=(vols[-seed_days:] if vols else None), sma_window=min(10, trader.recent_window))
                    trader.save_models(models_dir=models_dir)
                    print("Daily retrain complete.")
                except Exception as e:
                    print("Daily retrain failed:", e)

        # schedule background task
        asyncio.create_task(daily_retrain_task())

    # If we're debugging with a simulated stream, relax fitting requirements so
    # HMMs can be created and inspected quickly without needing long histories.
    if debug and simulate:
        trader.min_history = 20
        trader.recent_window = 5

    stop = asyncio.Event()

    def _sig(_s, _f):
        stop.set()

    loop = asyncio.get_event_loop()
    try:
        loop.add_signal_handler(signal.SIGINT, lambda: _sig(None, None))
        loop.add_signal_handler(signal.SIGTERM, lambda: _sig(None, None))
    except NotImplementedError:
        # Windows or environments where loop signal handlers not supported
        pass

    # choose stream source: live websocket or simulated
    stream = None
    # choose stream source: REST, websocket, or simulated
    use_rest = False
    try:
        # CLI flag will set an attribute on main() caller if present
        use_rest = getattr(main, "use_rest", False)
    except Exception:
        use_rest = False

    if use_rest and ticker_stream is not None:
        # import rest stream from api
        try:
            from kraken_hmm.api import rest_ticker_stream

            stream = rest_ticker_stream(PAIRS, interval=1.0)
        except Exception:
            stream = simulated_ticker_stream(PAIRS, interval=0.05, max_ticks=max_ticks)
    elif simulate or ticker_stream is None:
        stream = simulated_ticker_stream(PAIRS, interval=0.05, max_ticks=max_ticks)
    else:
        stream = ticker_stream(PAIRS)

    # If enabled, start a tiny HTTP API for health and positions
    api_runner = None
    if getattr(main, "enable_api", False) and web is not None:
        app = web.Application()

        async def health(request):
            return web.json_response({"ok": True})

        async def positions(request):
            # expose current positions and their thresholds
            out = {}
            for p, pos in trader.positions.items():
                out[p] = {
                    "qty": pos.qty,
                    "entry_price": pos.entry_price,
                    "stop_price": pos.stop_price,
                    "take_price": pos.take_price,
                }
            return web.json_response(out)

        app.add_routes([web.get('/health', health), web.get('/positions', positions)])
        api_runner = web.AppRunner(app)
        await api_runner.setup()
        site = web.TCPSite(api_runner, '0.0.0.0', getattr(main, 'http_port', 8080))
        await site.start()
        # mark readiness and print a clear readiness line so external checks
        # can detect the API is up (useful for service managers / probes)
        print(f"HTTP API listening on 0.0.0.0:{getattr(main, 'http_port', 8080)}")
        # expose a small attribute so tests/other code can detect API readiness
        try:
            trader.api_ready = True
        except Exception:
            pass
        # create a simple readiness file inside the state dir so systemd or
        # external monitors can check for API readiness without hitting HTTP.
        try:
            import pathlib

            state_dir = pathlib.Path(trader.state_file).parent if trader.state_file else pathlib.Path("state")
            state_dir.mkdir(parents=True, exist_ok=True)
            ready_file = state_dir / "api.ready"
            ready_file.write_text(str(int(time.time())))
            # remember path for cleanup
            trader._api_ready_file = str(ready_file)
        except Exception:
            pass
    elif getattr(main, "enable_api", False) and web is None:
        # user asked to enable API but aiohttp was not importable at module
        # load time. Print a clear warning so the operator can install deps.
        print("--enable-api requested but aiohttp is not available; HTTP API disabled")

    async for tick in stream:
        pair = tick.get("pair")
        price = tick.get("price")
        volume = tick.get("volume")
        ts = tick.get("time") or int(time.time())
        if pair and price:
            if debug:
                # In debug mode: update series and attempt to fit models early (if possible)
                trader.add_price(pair, price, volume, ts=ts)
                # do not call try_fit per-tick when we seeded at startup; only allow per-tick fits
                # when no initial seeding occurred (keeps training one-time)
                if not initial_debug_printed:
                    trader.try_fit(pair)

                # Try to initialize/fit models for pairs that lack models but have some history
                for p, prices in list(trader.prices.items()):
                    if p not in trader.models and len(prices) >= 3:
                        # create and try fitting with available history (up to min_history)
                        try:
                            trader.ensure_model(p)
                            hist = prices[-min(len(prices), trader.min_history):]
                            trader.models[p].fit(hist)
                        except Exception:
                            # if fit fails, remove model entry to keep diagnostics clean
                            trader.models.pop(p, None)

                # If we already printed the initial (one-time) debug snapshot after seeding,
                # avoid reprinting full per-model diagnostics every tick. Print a concise line instead.
                if initial_debug_printed:
                    print(f"tick pair={pair} price={price:.6f} history={len(trader.prices.get(pair, []))}")
                    # In debug mode, show what trades WOULD be made (planned allocations)
                    # only when execution is disabled.
                    if debug and not trader.execute_orders:
                        allocs = trader.allocate()
                        if allocs:
                            print("--- Planned Trades (debug, no execution) ---")
                            # decide whether to persist plans: only when the allocation set
                            # changed since the last logged set. This avoids writing identical
                            # plan entries every tick during debug.
                            last_logged = getattr(trader, "_last_logged_allocs", None)
                            should_log = last_logged != allocs
                            for ap, cap in allocs.items():
                                last_prices = trader.prices.get(ap, [])
                                if not last_prices:
                                    continue
                                last_price = last_prices[-1]
                                qty = cap / last_price if last_price and last_price > 0 else 0.0
                                sharpe = trader.compute_sharpe_for_pair(ap)
                                sharpe_s = f"{sharpe:.6f}" if sharpe is not None else "n/a"
                                print(f"plan pair={ap} capital={cap:.2f} price={last_price:.6f} qty={qty:.6f} sharpe={sharpe_s}")
                            # persist planned trades only once per unique allocation set
                            if should_log:
                                try:
                                    for ap, cap in allocs.items():
                                        last_prices = trader.prices.get(ap, [])
                                        if not last_prices:
                                            continue
                                        last_price = last_prices[-1]
                                        qty = cap / last_price if last_price and last_price > 0 else 0.0
                                        sharpe = trader.compute_sharpe_for_pair(ap)
                                        plan_rec = {
                                            "ts": int(ts),
                                            "type": "plan",
                                            "mode": "debug",
                                            "pair": ap,
                                            "qty": qty,
                                            "price": last_price,
                                            "capital": cap,
                                            "sharpe": sharpe,
                                        }
                                        trader._write_trade(plan_rec)
                                    # update last-logged snapshot
                                    trader._last_logged_allocs = dict(allocs)
                                except Exception:
                                    pass
                        else:
                            print("(no planned trades)")
                else:
                    # collect diagnostics across models
                    diagnostics = []
                    for p in sorted(trader.prices.keys()):
                        history_len = len(trader.prices.get(p, []))
                        has_model = p in trader.models
                        line = f"pair={p} history={history_len} model={'yes' if has_model else 'no'}"
                        print(line)

                    for p, model in trader.models.items():
                        prices = trader.prices.get(p, [])
                        try:
                            means, stds = model.state_stats()
                            # predict state if enough recent data
                            state = None
                            if len(prices) >= trader.recent_window + 1:
                                try:
                                    vols = trader.volumes.get(p)
                                    recent_vols = vols[-trader.recent_window:] if vols is not None else None
                                    state = model.predict_state(prices[-trader.recent_window:], recent_volumes=recent_vols)
                                except Exception:
                                    state = None
                            sharpe = trader.compute_sharpe_for_pair(p)
                            diagnostics.append((p, sharpe, state, means, stds))
                        except Exception:
                            continue

                    # print ranking sorted by sharpe
                    if diagnostics:
                        # sort by sharpe, putting None scores last
                        diagnostics.sort(key=lambda x: (x[1] is None, x[1] if x[1] is not None else -999.0), reverse=True)
                        print("\n--- HMM Debug Snapshot ---")
                        for p, score, state, means, stds in diagnostics:
                            score_s = f"{score:.6f}" if score is not None else "n/a"
                            s = f"pair={p} sharpe={score_s}"
                            if state is not None:
                                s += f" predicted_state={state}"
                            s += f" means={['{:.4e}'.format(m) for m in means]} stds={['{:.4e}'.format(sv) for sv in stds]}"
                            print(s)
                    else:
                        print("(no fitted HMM models yet — collecting history)")
            else:
                await trader.on_tick(pair, price, volume)

        # check if we should stop (signal received)
        if stop.is_set():
            break

    # main loop has exited: persist state and shut down HTTP API if running
    try:
        try:
            trader.save_state()
        except Exception:
            pass
        if api_runner is not None:
            try:
                # remove readiness file if present
                try:
                    rf = getattr(trader, "_api_ready_file", None)
                    if rf:
                        import os

                        try:
                            os.remove(rf)
                        except Exception:
                            pass
                except Exception:
                    pass
                await api_runner.cleanup()
            except Exception:
                pass
    finally:
        pass


async def simulated_ticker_stream(pairs: List[str], interval: float = 0.25, max_ticks: int = 0):
    """Async generator that yields simulated ticker updates for debug/testing.

    Prices follow an independent small random walk per pair.
    If max_ticks>0, stop after that many ticks.
    """
    import random
    prices = {p: 100.0 + random.uniform(-1.0, 1.0) for p in pairs}
    # synthetic volumes (start around 1000)
    volumes = {p: 1000.0 + random.uniform(-100.0, 100.0) for p in pairs}
    tick = 0
    while True:
        await asyncio.sleep(interval)
        for p in pairs:
            # small random walk
            drift = random.gauss(0, 0.001)
            prices[p] = max(0.0001, prices[p] * (1.0 + drift))
            # small volume noise
            volumes[p] = max(0.1, volumes[p] * (1.0 + random.gauss(0, 0.01)))
            yield {"pair": p, "price": prices[p], "volume": volumes[p]}
        tick += 1
        if max_ticks and tick >= max_ticks:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run KrakenHMM trader")
    parser.add_argument("--total-capital", type=float, default=1000.0, help="Total capital to allocate")
    parser.add_argument("--n-assets", type=int, default=5, help="Number of assets to trade")
    parser.add_argument("--debug", action="store_true", help="Debug mode: print HMM outputs and rankings, do not execute trades")

    parser.add_argument("--simulate", action="store_true", help="Use simulated ticker stream instead of connecting to Kraken")
    parser.add_argument("--max-ticks", type=int, default=0, help="Stop after this many ticks when simulating (0 = unlimited)")
    parser.add_argument("--use-rest", action="store_true", help="Use Kraken REST polling for ticker data instead of websocket or simulation")
    parser.add_argument("--seed-history", action="store_true", help="Fetch recent OHLC from Kraken REST and seed trader history before streaming")
    parser.add_argument("--seed-days", type=int, default=365, help="How many past days to fetch for seeding/retraining")
    parser.add_argument("--models-dir", type=str, default="models", help="Directory to save/load fitted models")
    parser.add_argument("--execute-orders", action="store_true", help="Enable live order execution via Kraken API")
    parser.add_argument("--api-key", type=str, default=None, help="Kraken API key for placing orders")
    parser.add_argument("--api-secret", type=str, default=None, help="Kraken API secret (base64) for placing orders")
    parser.add_argument("--trade-log-dir", type=str, default="logs", help="Directory where daily trade logs are written")
    parser.add_argument("--trade-log-file", type=str, default=None, help="Exact trade log file to write (disables rotation)")
    parser.add_argument("--enable-api", action="store_true", help="Start a small HTTP API for health/positions")
    parser.add_argument("--http-port", type=int, default=8080, help="Port for HTTP API")
    parser.add_argument("--state-file", type=str, default="state/trader_state.json", help="Path to trader state file")

    args = parser.parse_args()
    # attach flag to main so it can be picked up inside (avoids changing signature)
    setattr(main, "use_rest", args.use_rest)
    setattr(main, "seed_history", args.seed_history)
    setattr(main, "seed_days", args.seed_days)
    setattr(main, "models_dir", args.models_dir)
    setattr(main, "execute_orders", args.execute_orders)
    setattr(main, "api_key", args.api_key)
    setattr(main, "api_secret", args.api_secret)
    setattr(main, "trade_log_dir", args.trade_log_dir)
    setattr(main, "trade_log_file", args.trade_log_file)
    setattr(main, "enable_api", args.enable_api)
    setattr(main, "http_port", args.http_port)
    setattr(main, "state_file", args.state_file)
    asyncio.run(
        main(
            total_capital=args.total_capital,
            n_assets=args.n_assets,
            debug=args.debug,
            simulate=args.simulate,
            max_ticks=args.max_ticks,
        )
    )
