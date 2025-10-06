#!/usr/bin/env python3
"""Summarize trades recorded as JSONL files (daily rotation).

Usage:
    python scripts/summarize_trades.py --log-dir ./logs

This script aggregates buys and sells and prints per-pair realized P&L and totals.
"""
import argparse
import json
import os
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--log-dir", default='.', help="Directory containing trades-YYYYMMDD.jsonl files or a single file")
    p.add_argument("--file", default=None, help="Specific file to parse (overrides --log-dir)")
    return p.parse_args()


def load_records(paths):
    for path in paths:
        try:
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except Exception:
                        continue
        except Exception:
            continue


def summarize(records):
    # track realized P&L by pair
    pnl_by_pair = defaultdict(float)
    trades_by_pair = defaultdict(int)
    # naive inventory tracking: accumulate buys and reduce on sells using entry_price in logs
    # prefer 'sell' records that include 'pnl'
    for r in records:
        t = r.get('type')
        pair = r.get('pair')
        if not pair:
            continue
        if t == 'sell':
            pnl = r.get('pnl')
            if pnl is not None:
                pnl_by_pair[pair] += float(pnl)
            trades_by_pair[pair] += 1
    return pnl_by_pair, trades_by_pair


def main():
    args = parse_args()
    paths = []
    if args.file:
        paths = [args.file]
    else:
        # scan directory for files starting with trades-
        for fn in os.listdir(args.log_dir):
            if fn.startswith('trades-') and fn.endswith('.jsonl'):
                paths.append(os.path.join(args.log_dir, fn))
        # fallback: include a trades.jsonl file if present
        if not paths:
            maybe = os.path.join(args.log_dir, 'trades.jsonl')
            if os.path.exists(maybe):
                paths.append(maybe)

    records = list(load_records(paths))
    pnl_by_pair, trades_by_pair = summarize(records)

    total = sum(pnl_by_pair.values())
    print(f"Analyzed {len(records)} trade records across {len(pnl_by_pair)} pairs")
    print("Per-pair realized P&L:")
    for pair, pnl in sorted(pnl_by_pair.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {pair}: {pnl:.2f} ({trades_by_pair.get(pair,0)} trades)")
    print("---")
    print(f"Total realized P&L: {total:.2f}")


if __name__ == '__main__':
    main()
