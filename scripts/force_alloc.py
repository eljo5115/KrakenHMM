#!/usr/bin/env python3
"""Force an allocation with synthetic histories and write simulated plan/buy records.

This script is fast and local — it doesn't contact Kraken. It uses the same
logging path as the trader so you can inspect the day's JSONL file.
"""
import asyncio
import sys
import os

# ensure repo root is on sys.path when running from scripts/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from run_trader import PAIRS
from kraken_hmm.trader import Trader

if __name__ == "__main__":
    trader = Trader(total_capital=1000.0, n_assets=5)
    # relax history requirements for quick testing
    trader.min_history = 5
    trader.recent_window = 3
    # seed small synthetic histories for the first N pairs
    for p in PAIRS[:8]:
        # produce 10 synthetic close prices with small drift
        prices = [100.0 + (i * 0.2) for i in range(10)]
        for price in prices:
            trader.add_price(p, price)
        # create model entry (no need to fit)
        trader.ensure_model(p)

    allocs = trader.allocate()
    print("Computed allocations:", allocs)

    # run simulated execution (this will write plan + simulated buy records)
    asyncio.run(trader.execute_allocations(allocs))
    print("Simulated execution complete. Check logs/trades-<UTCdate>.jsonl for entries.")
