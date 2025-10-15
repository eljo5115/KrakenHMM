#!/usr/bin/env python3
"""
Run a live reconciliation against Kraken to sync local `state/trader_state.json`
with exchange balances using Trader.reconcile_positions.

Usage:
  # provide keys via env
  KRAKEN_API_KEY=... KRAKEN_API_SECRET=... ./scripts/reconcile_with_exchange.py

  # or via CLI args
  ./scripts/reconcile_with_exchange.py --api-key KEY --api-secret SECRET

The script will:
 - load existing state
 - instantiate Trader (lightweight)
 - call reconcile_positions(api_key, api_secret)
 - print the updated local positions

WARNING: This contacts the exchange and requires valid API credentials.
"""
import os
import sys
import argparse
import asyncio
import json
from pathlib import Path
import time


def load_env_files():
    """Load simple KEY=VALUE pairs from project .env and scripts/trader.env into os.environ.

    This is a tiny, dependency-free replacement for python-dotenv so you can
    keep API keys out of the command line. Existing environment variables are
    not overwritten.
    """
    root = Path(__file__).resolve().parent.parent
    candidates = [root / '.env', root / 'scripts' / 'trader.env']
    for p in candidates:
        try:
            if not p.exists():
                continue
            with open(p, 'r') as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith('#'):
                        continue
                    if '=' not in line:
                        continue
                    k, v = line.split('=', 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if not k:
                        continue
                    # do not overwrite existing env vars
                    if os.getenv(k) is None:
                        os.environ[k] = v
        except Exception:
            # best-effort loader; ignore parse errors
            continue

ROOT = Path(__file__).resolve().parent.parent
STATE_PATH = ROOT / 'state' / 'trader_state.json'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--api-key', help='Kraken API key (env KRAKEN_API_KEY)', default=os.getenv('KRAKEN_API_KEY'))
    p.add_argument('--api-secret', help='Kraken API secret (env KRAKEN_API_SECRET)', default=os.getenv('KRAKEN_API_SECRET'))
    p.add_argument('--pairs', nargs='*', help='Optional list of pairs to reconcile (e.g., ADA/USD LINK/USD)')
    p.add_argument('--dry-run', action='store_true', help='Do not persist state changes (state file will be restored). Note: trade logs may still be written by the trader.')
    p.add_argument('--no-prune', action='store_false', dest='prune_zero', help='Do not remove zero-qty positions after reconciliation')
    p.set_defaults(prune_zero=True)
    return p.parse_args()


async def main():
    # Load .env / scripts/trader.env into environment (if present)
    load_env_files()
    args = parse_args()
    api_key = args.api_key
    api_secret = args.api_secret
    if not api_key or not api_secret:
        print('API key/secret not provided. Set KRAKEN_API_KEY and KRAKEN_API_SECRET env vars or pass --api-key/--api-secret.')
        return

    # Ensure project root is on sys.path so we can import kraken_hmm without requiring PYTHONPATH
    sys.path.insert(0, str(ROOT))
    # Import lazily to preserve module load behavior
    from kraken_hmm.trader import Trader

    trader = Trader()
    # load existing persisted state so reconcile updates the same structure
    trader.load_state()
    trader.load_models()

    pairs = args.pairs if args.pairs else None
    print('Running live reconciliation against exchange for', 'all stored positions' if pairs is None else f'pairs={pairs}')

    # If dry-run requested, backup state file so we can restore later
    state_backup = None
    if args.dry_run and STATE_PATH.exists():
        state_backup = STATE_PATH.with_suffix('.bak.tmp')
        try:
            with open(STATE_PATH, 'rb') as src, open(state_backup, 'wb') as dst:
                dst.write(src.read())
        except Exception as e:
            print('Warning: could not create state backup for dry-run:', e)

    try:
        await trader.reconcile_positions(api_key=api_key, api_secret=api_secret, pairs=pairs)
    except Exception as e:
        print('Reconciliation failed:', e)
        # restore state backup if dry-run and backup exists
        if args.dry_run and state_backup and state_backup.exists():
            try:
                with open(state_backup, 'rb') as src, open(STATE_PATH, 'wb') as dst:
                    dst.write(src.read())
                print('State restored from dry-run backup.')
            except Exception:
                print('Failed to restore state from dry-run backup.')
        return

    # After reconciliation: ensure positions have sensible entry/stop/take values
    # Default behavior: if a position has no entry_price, set it from recent prices
    # (or model metadata loaded into trader.prices) and then populate thresholds
    # from models (stop/take). Log any changes and persist state.
    try:
        from inspect import iscoroutinefunction
        # set entry prices when missing
        updated_entry = []
        for p, pos in list(trader.positions.items()):
            if pos is None:
                continue
            try:
                has_entry = pos.entry_price is not None
            except Exception:
                has_entry = False
            if not has_entry:
                # try to derive a base price from trader.prices (models may have populated prices_tail)
                last_prices = trader.prices.get(p, []) or []
                base_price = None
                if last_prices:
                    try:
                        base_price = float(last_prices[-1])
                    except Exception:
                        base_price = None
                # if we cannot find a base price, skip
                if base_price is None:
                    continue
                try:
                    pos.entry_price = float(base_price)
                    rec = {
                        'ts': int(time.time()),
                        'type': 'set_entry',
                        'pair': p,
                        'entry_price': pos.entry_price,
                        'method': 'reconcile_default_from_price',
                    }
                    try:
                        trader._write_trade(rec)
                    except Exception:
                        pass
                    updated_entry.append(p)
                except Exception:
                    pass
        # after setting entry prices, attempt to populate thresholds from models
        try:
            # populate_thresholds_from_models is async
            await trader.populate_thresholds_from_models(pairs=None)
        except Exception:
            pass
        if updated_entry:
            try:
                trader.save_state()
            except Exception:
                pass
            print('Set entry_price for positions:', updated_entry)
    except Exception as e:
        print('Error while setting default entry prices from models/prices:', e)

    # If prune-zero requested, remove zero-qty positions from trader and persist
    if args.prune_zero:
        try:
            removed = []
            # iterate over a copy of keys to avoid mutation during iteration
            for p in list(trader.positions.keys()):
                pos = trader.positions.get(p)
                try:
                    qty = float(pos.qty)
                except Exception:
                    qty = None
                if qty is None:
                    continue
                if qty == 0.0:
                    # write a small prune record then remove
                    rec = {
                        'ts': int(time.time()),
                        'type': 'prune_zero',
                        'pair': p,
                        'old_qty': float(pos.qty),
                        'new_qty': 0.0,
                        'note': 'Pruned zero-qty position after reconciliation'
                    }
                    try:
                        trader._write_trade(rec)
                    except Exception:
                        pass
                    trader.positions.pop(p, None)
                    removed.append(p)
            if removed:
                try:
                    trader.save_state()
                except Exception as e:
                    print('Warning: could not save state after pruning:', e)
                print('Pruned zero-qty positions:', removed)
        except Exception as e:
            print('Error during prune-zero step:', e)

    # If dry-run requested, restore state file from backup (note: trade logs may have entries)
    if args.dry_run and state_backup and state_backup.exists():
        try:
            with open(state_backup, 'rb') as src, open(STATE_PATH, 'wb') as dst:
                dst.write(src.read())
            print('State restored from dry-run backup. Note: trade logs may still contain reconcile records.')
        except Exception as e:
            print('Failed to restore state from dry-run backup:', e)

    # print summary of positions after reconciliation
    try:
        state = {}
        if STATE_PATH.exists():
            with open(STATE_PATH, 'r') as f:
                state = json.load(f)
        positions = state.get('positions', {})
        if not positions:
            print('No positions recorded after reconciliation.')
        else:
            print('Positions after reconciliation:')
            for p, info in positions.items():
                print(f"  {p}: qty={info.get('qty')} entry={info.get('entry_price')} stop={info.get('stop_price')} take={info.get('take_price')}")
    except Exception as e:
        print('Could not read state file after reconcile:', e)


if __name__ == '__main__':
    asyncio.run(main())
