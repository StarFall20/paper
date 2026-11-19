# Nested label budgets and finite estimation in Swissmetro

This follow-up uses the [same released Swissmetro records](https://github.com/michelbierlaire/biogeme/blob/master/src/biogeme/data/data/swissmetro.dat), four- and eight-parameter target models, 600 training respondents and 240 holdout respondents as the [earlier experiment](../swissmetro_corrected_path_20261118/README.md). Each of 96 fixed random respondent orders gives nested labelled groups of 2, 5, 10, 12, 15, 20 and 40. All other respondents in the 351-person pool supply auxiliary predictions from the frozen teacher.

At unit correction the four- and eight-parameter targets have finite estimates in 95 of 96 allocations at 15 labels, and in all 96 at 20 and 40 labels. Some individual trajectories move in the other direction. Six four-parameter and seven eight-parameter fits are finite at two labels and exterior at five. There are two more finite-to-exterior transitions from five to ten labels and one from twelve to fifteen. This shift can occur when a respondent is transferred from the auxiliary sample to the labelled sample and the sample means change.

Even where every unit-weight fit is finite, its mean holdout log loss exceeds that of the same-label ridge by 0.0799 and 0.1401 at 20 labels, and 0.0246 and 0.0360 at 40 labels for the four- and eight-parameter models. The respondent-cluster variance rule selects median weight zero in every budget and specification cell. Its mean loss also exceeds the same-label ridge in every cell, including cases where it falls back to this ridge. The direct MNL using teacher-training and revealed labels has lower mean loss than both unit-weight and variance-tuned correction in every cell. Individual comparisons are retained. The design contains one fixed pool and one fixed evaluation group. Counts summarize the tested allocations.

## Reproduction

Use Python 3.12 and [the pinned libraries](requirements.txt). Set single-threaded numerical execution to match the recorded run.

    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    data=/absolute/path/to/swissmetro.dat
    python run_audit.py --input "$data" --out regenerated-four --specification canonical_4
    for start in 0 24 48 72; do
        python run_audit.py --input "$data" --out "regenerated-eight-$start" --specification mode_specific_8 --fast --start-allocation "$start" --limit 24
    done
    python combine.py --input "$data" --parts regenerated-four regenerated-eight-{0,24,48,72} --out regenerated
    python verify.py --input "$data" --results regenerated --precedent-summary ../swissmetro_corrected_path_20261118/results/summary.json --precedent-allocations ../swissmetro_corrected_path_20261118/results/allocations.csv --out regenerated/verification.json

Within Reproduction Materials, use the undated sibling folder `swissmetro_corrected_path` for the two prior files. The independent verifier reconstructs the original available alternatives, confirms the exact ordinary-separation witnesses and negative corrected recession directions, reconstructs positive masses when the native fit exists, and recalculates all forecast losses and variance weights.

- [Plan](plan.json)
- [Complete cell counts and nested transitions](results/summary.json)
- [Independent verification](results/verification.json)
- [Primary-source comparison](literature.md)
