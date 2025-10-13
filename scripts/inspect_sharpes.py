#!/usr/bin/env python3
"""Inspect saved HMM models and print predicted-state scores.

This script loads each `models/*_.pkl` and its corresponding `_meta.json` (if present)
and computes a simple per-pair score = mean_state / std_state for the predicted state
using the `prices_tail` from the metadata as the recent window.
"""
import os
import pickle
import json
from kraken_hmm.hmm_model import HMMModel

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
MODELS_DIR = os.path.abspath(MODELS_DIR)

if not os.path.isdir(MODELS_DIR):
    print('No models directory found at', MODELS_DIR)
    raise SystemExit(1)

for fname in sorted(os.listdir(MODELS_DIR)):
    if not fname.endswith('.pkl'):
        continue
    safe = fname[:-4]
    pair = safe.replace('_', '/')
    pfile = os.path.join(MODELS_DIR, fname)
    meta_file = os.path.join(MODELS_DIR, f"{safe}_meta.json")
    prices_tail = []
    if os.path.exists(meta_file):
        try:
            with open(meta_file, 'r') as f:
                meta = json.load(f)
            prices_tail = meta.get('prices_tail') or []
        except Exception as e:
            print(f"{pair}: failed to read meta {meta_file}: {e}")

    try:
        with open(pfile, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"{pair}: failed to load model {pfile}: {e}")
        continue

    try:
        means, stds = model.state_stats()
        n_comp = getattr(model, 'n_components_used', None) or getattr(model, 'n_components', None) or 0
        hist_len = len(prices_tail)
        state = None
        score = None
        mean = None
        std = None
        if prices_tail and len(prices_tail) >= 2:
            recent = prices_tail[-min(len(prices_tail), 10):]
            try:
                state = model.predict_state(recent)
            except Exception:
                state = None
        if state is not None and state < len(means):
            mean = float(means[state])
            std = float(stds[state]) if float(stds[state]) != 0 else float(1e-12)
            if std != 0:
                score = mean / std
        print(f"{pair}: n_comp={n_comp} history_len={hist_len} state={state} mean={mean} std={std} score={score}")
    except Exception as e:
        print(f"{pair}: error computing stats: {e}")
