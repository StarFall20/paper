"""Recreate paired MSE summaries from locally regenerated simulation trials.

The input is a generated trial table, not respondent data. It is deliberately
excluded from the public package. Reproduce it with replay_choice_power_path.py.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def summarize(directory):
    trials = pd.read_csv(directory / 'replayed_power_trial_diagnostics.csv')
    assert len(trials) == 2500
    assert not trials.duplicated(['replication', 'power']).any()
    wide = trials.pivot(index='replication', columns='power', values='MSE')
    assert len(wide) == 500
    rows = []
    for reference, comparator in [(1., .01), (1., .1), (1., .5),
                                  (.5, .01), (.1, .01), (0., .01)]:
        common = wide[[reference, comparator]].dropna()
        difference = common[comparator] - common[reference]
        assert len(common) > 1
        rows.append({
            'power_reference': reference,
            'power_comparator': comparator,
            'shared_finite_estimates': len(common),
            'MSE_reference': common[reference].mean(),
            'MSE_comparator': common[comparator].mean(),
            'mean_MSE_difference': difference.mean(),
            'difference_MCSE': difference.std(ddof=1) / np.sqrt(len(common)),
            'median_MSE_difference': difference.median(),
            'fraction_comparator_larger_MSE': difference.gt(0).mean(),
            'scope': 'paired on shared finite estimates;full availability retained separately',
        })
    frame = pd.DataFrame(rows)
    frame.to_csv(directory / 'replayed_power_paired_comparisons.csv', index=False)
    return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    frame = summarize(args.out)
    print(f'Recreated {len(frame)} paired comparisons from 500 original draws.')
