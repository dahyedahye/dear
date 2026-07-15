#!/usr/bin/env python3
"""Tabulate AUC / Acc / R.Acc / F.Acc from the per-generator eval JSONs.

Reads ``{results_dir}/{method}/{test_dir}/fake/{gen}.json`` (written by
``scripts/eval.py``) and prints a per-generator breakdown plus the average over
generators — the numbers reported in the paper tables.

Example:
    python scripts/analyze_results.py --results_dir results --methods dear_c dear_r \\
        --test_dirs test test_processed
"""

import json
import argparse
from pathlib import Path

# 9 LDM-family generators used for the main table
FAKE_TYPES = ['sd', 'Midjourney', 'kandinsky', 'playground', 'pixelart',
              'lcm', 'flux', 'wuerstchen', 'amused']
METRICS = ['auc', 'acc', 'real_acc', 'fake_acc']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='results')
    ap.add_argument('--methods', nargs='+', required=True)
    ap.add_argument('--fake_types', nargs='+', default=FAKE_TYPES)
    ap.add_argument('--test_dirs', nargs='+', default=['test', 'test_processed'])
    ap.add_argument('--per_generator', action='store_true',
                    help='also print per-generator rows (not just the average)')
    args = ap.parse_args()

    for td in args.test_dirs:
        print(f"\n{'='*64}\n{td.upper()}  (avg over {len(args.fake_types)} generators)\n{'='*64}")
        print(f"{'method':<18}{'AUC':>8}{'Acc':>8}{'R.Acc':>8}{'F.Acc':>8}")
        print('-' * 50)
        for m in args.methods:
            sums = {k: [] for k in METRICS}
            per_gen = {}
            for ft in args.fake_types:
                jp = Path(args.results_dir) / m / td / 'fake' / f'{ft}.json'
                if jp.exists():
                    r = json.load(open(jp))
                    per_gen[ft] = r
                    for k in METRICS:
                        sums[k].append(r.get(k, float('nan')))
            avg = {k: (sum(v) / len(v) if v else float('nan')) for k, v in sums.items()}
            print(f"{m:<18}{avg['auc']:>8.2f}{avg['acc']:>8.2f}{avg['real_acc']:>8.2f}{avg['fake_acc']:>8.2f}")
            if args.per_generator:
                for ft in args.fake_types:
                    if ft in per_gen:
                        r = per_gen[ft]
                        print(f"  {ft:<16}{r['auc']:>8.2f}{r['acc']:>8.2f}"
                              f"{r['real_acc']:>8.2f}{r['fake_acc']:>8.2f}")


if __name__ == '__main__':
    main()
