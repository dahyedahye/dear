#!/usr/bin/env python3
"""Compute detection metrics from a pair of logit CSVs (real vs. fake).

Reads the per-subset CSVs written by ``bucketed_inference.py`` and reports AP,
AUC, real accuracy, fake accuracy, and balanced accuracy. A logit threshold of 0
separates real (< 0) from fake (> 0).

Example:
    python scripts/eval.py \\
        --real_csv results/dear_c/test/real/redcaps.csv \\
        --fake_csv results/dear_c/test/fake/flux.csv
"""

import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def compute_metrics(real_csv, fake_csv, score_col_idx=1):
    real_scores = pd.read_csv(real_csv).iloc[:, score_col_idx].values
    fake_scores = pd.read_csv(fake_csv).iloc[:, score_col_idx].values

    y_true = np.concatenate([np.zeros(len(real_scores)), np.ones(len(fake_scores))])
    y_scores = np.concatenate([real_scores, fake_scores])

    ap = average_precision_score(y_true, y_scores) * 100
    auc = roc_auc_score(y_true, y_scores) * 100
    real_acc = (real_scores < 0).mean() * 100
    fake_acc = (fake_scores > 0).mean() * 100
    acc = (real_acc + fake_acc) / 2

    return {'ap': ap, 'auc': auc, 'real_acc': real_acc, 'fake_acc': fake_acc, 'acc': acc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real_csv', default='results/dear_c/test/real/redcaps.csv')
    parser.add_argument('--fake_csv', default='results/dear_c/test/fake/sd.csv')
    args = parser.parse_args()

    print(args.fake_csv)
    metrics = compute_metrics(args.real_csv, args.fake_csv)
    print(f"AP: {metrics['ap']:.2f}, AUC: {metrics['auc']:.2f}, "
          f"R.Acc: {metrics['real_acc']:.2f}, F.Acc: {metrics['fake_acc']:.2f}, "
          f"Acc: {metrics['acc']:.2f}")

    json_path = f"{Path(args.fake_csv).parent}/{Path(args.fake_csv).stem}.json"
    json.dump(metrics, open(json_path, 'w'))


if __name__ == '__main__':
    main()
