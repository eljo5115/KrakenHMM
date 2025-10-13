#!/usr/bin/env python3
"""Load models from models/ and call Trader.log_hmm_decision to append rows to logs/hmm_decisions.csv"""
import os
import sys
sys.path.insert(0, os.path.abspath('.'))
from kraken_hmm.trader import Trader
from kraken_hmm.hmm_model import HMMModel
import pickle

MODELS_DIR = os.path.join('models')
if not os.path.isdir(MODELS_DIR):
    print('no models dir')
    raise SystemExit(1)

trader = Trader()
# ensure logs dir exists
os.makedirs(trader.trade_log_dir, exist_ok=True)

for fname in sorted(os.listdir(MODELS_DIR)):
    if not fname.endswith('.pkl'):
        continue
    safe = fname[:-4]
    pair = safe.replace('_', '/')
    pfile = os.path.join(MODELS_DIR, fname)
    try:
        with open(pfile, 'rb') as f:
            model = pickle.load(f)
        trader.models[pair] = model
        # try to seed prices from meta if present
        meta_file = os.path.join(MODELS_DIR, f"{safe}_meta.json")
        if os.path.exists(meta_file):
            import json
            with open(meta_file, 'r') as fm:
                meta = json.load(fm)
            prices_tail = meta.get('prices_tail') or []
            trader.prices[pair] = list(map(float, prices_tail))
        # write decision line
        trader.log_hmm_decision(pair)
        print('wrote', pair)
    except Exception as e:
        print('failed', pair, e)
