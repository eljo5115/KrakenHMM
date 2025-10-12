## Quick orientation

This project implements an algorithmic trader for Kraken using per-asset HMMs.
Key runtime pieces:

- `run_trader.py` — top-level runner and CLI. Defers heavy imports until runtime.
- `kraken_hmm/api.py` — websocket/REST helpers and private order helpers (signing).
- `kraken_hmm/hmm_model.py` — HMM feature engineering and wrapper around `hmmlearn.GaussianHMM`.
- `kraken_hmm/trader.py` — core trading logic: time series storage, model lifecycle, ranking, allocation, state persistence, and trade logging.
- `kraken_hmm/utils.py` — small helpers (sharpe / stats).

Read these files together to understand data flow: `api.py` supplies ticks -> `trader.Trader.add_price` stores series -> `hmm_model.HMMModel.fit` consumes recent prices -> `trader.compute_sharpe_for_pair` and `allocate` decide targets -> `trader.execute_allocations` (placeholder or live via `api.place_market_order`).

## Important contracts & shapes

- Streaming tick: dict with at least `{'pair': str, 'price': float, 'volume': float, 'time': int}` (see `api.ticker_stream` and `rest_ticker_stream`).
- Saved model files: `models/<PAIR_SAFE>.pkl` (pickle of HMMModel) with `models/<PAIR_SAFE>_meta.json` containing short `prices_tail`, `volumes_tail` and `history_len`.
- Persistent state: JSON at `state/trader_state.json` with `positions`, `_last_executed_allocs`, and `last_allocation_date`. `Trader.save_state` writes atomically via a `.tmp` then `os.replace`.

## Project-specific patterns and gotchas

- Pair namespace: the code uses human-readable pairs like `XBT/USD`. When persisted to filenames, `/` is replaced with `_` (e.g. `XBT_USD.pkl`). REST helpers often remove the slash when querying Kraken.
- HMM features are percentage returns, optional volume-change, and price-vs-SMA. `HMMModel.fit` adapts the feature set by sample size to avoid degenerate fits. See `kraken_hmm/hmm_model.py` for exact feature ordering — AI changes must preserve the feature ordering to remain compatible with existing pickles.
- Model pickles are loaded with `pickle.load` in `Trader.load_models`. Do NOT unpickle files from untrusted sources.
- Order signing: `api.place_market_order` expects `api_secret` to be the base64-encoded secret (Kraken API style). Live trading only runs when `--execute-orders` and valid `KRAKEN_API_KEY`/`KRAKEN_API_SECRET` (env or CLI) are provided.
- The runner defers heavy imports (aiohttp, numpy, hmmlearn) until `main` so the module can be imported without installing deps — preserve that pattern when refactoring tests/tools.

## How to run locally (examples)

- Fast debug simulation (no network, small history):

```bash
/home/eli/KrakenHMM/.venv/bin/python run_trader.py --simulate --debug --max-ticks 100
```

- Seed historical daily OHLC (requires network + aiohttp) and save models:

```bash
/home/eli/KrakenHMM/.venv/bin/python run_trader.py --seed-history --seed-days 365 --models-dir models
```

- Start live (WARNING: real money) — use `scripts/trader.env` to store secrets and run with `--execute-orders`:

```bash
# export or place KRAKEN_API_KEY / KRAKEN_API_SECRET (secret must be base64)
/home/eli/KrakenHMM/.venv/bin/python run_trader.py --execute-orders --api-key $KRAKEN_API_KEY --api-secret $KRAKEN_API_SECRET
```

## Developer workflows & debugging aids

- Short-circuit heavy deps by importing `run_trader` (module load) — heavy modules are only imported inside `main`.
- Use `--simulate` and `--debug` to get reproducible diagnostics (the runner prints HMM stats and planned allocations). `Trader.min_history` and `recent_window` are relaxed when debug+simulate is enabled.
- Enable a tiny HTTP API for health and positions with `--enable-api` and `--http-port`. The runner also writes a readiness file `state/api.ready` when the API is up (useful for systemd/service checks).

## Files & locations to reference when changing behavior

- Live execution and signing: `kraken_hmm/api.py::place_market_order`.
- Model training and feature choices: `kraken_hmm/hmm_model.py`.
- Allocation, persistence, and trade-logging: `kraken_hmm/trader.py` (search functions: `allocate`, `execute_allocations`, `save_state`, `load_state`, `_write_trade`).
- Runner flags and orchestration: `run_trader.py` and `scripts/run_trader_service.sh` (service wrapper).

## Safety & tests

- There are no unit tests in the repo — prefer small, focused tests when changing model serialization, order signing, or persistence.
- Never load remote model pickles during CI; add a contract test that verifies `HMMModel.fit` roundtrips to a pickle you control.

If any section is unclear or you want me to expand examples (e.g., show how to add a unit test for `place_market_order` or a CI job that lints/validates saved models), tell me which area to expand.
