#!/usr/bin/env python3
"""Simple JSONL deduper: collapse consecutive identical lines.

Usage:
    python scripts/dedupe_trades.py <input.jsonl> [--inplace]

By default writes <input>.dedup.jsonl next to the input. Use --inplace to
replace the original file (atomic write).
"""
import sys
import os
import argparse

def dedupe_file(infile: str, outpath: str):
    prev = None
    written = 0
    with open(infile, 'r') as fi, open(outpath, 'w') as fo:
        for line in fi:
            # strip trailing newline for comparison but preserve original formatting
            cur = line.rstrip('\n')
            if cur == prev:
                continue
            fo.write(cur + '\n')
            prev = cur
            written += 1
    return written


def main():
    p = argparse.ArgumentParser()
    p.add_argument('input', help='Input JSONL file')
    p.add_argument('--inplace', action='store_true', help='Replace input file with deduped output')
    args = p.parse_args()
    infile = args.input
    if not os.path.exists(infile):
        print(f"Input file not found: {infile}")
        sys.exit(2)
    outpath = infile + '.dedup.jsonl'
    print(f"Deduping {infile} -> {outpath} (consecutive identical lines are collapsed)")
    written = dedupe_file(infile, outpath)
    print(f"Wrote {written} lines")
    if args.inplace:
        bak = infile + '.bak'
        os.replace(infile, bak)
        os.replace(outpath, infile)
        print(f"Replaced original; backup at {bak}")

if __name__ == '__main__':
    main()
