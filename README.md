# KrakenHMM
An algorithmic trader for Kraken Exchange based on a Hidden Markov Model.

## Quickstart

Install dependencies from `requirements.txt` then run the example trader:

```bash
python run_trader.py
```

Files of interest:

- `kraken_hmm/api.py` - Kraken websocket helper to stream ticker data
- `kraken_hmm/hmm_model.py` - HMM model wrapper using `hmmlearn`
- `kraken_hmm/trader.py` - Trading decision logic that ranks assets by Sharpe-like metric
- `kraken_hmm/utils.py` - Small helpers (sharpe, stats)
-- `run_trader.py` - Example runner script (top-level)

Notes: This is a starting point and uses placeholders for order execution. Replace execute_allocations with real REST calls and add authentication before running live.
