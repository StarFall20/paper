# Reproduce the Firth choice comparison

This independent supplement replays an established comparator on the same 1000 original simulated samples. It accompanies the current compensated-logit paper and introduces no new estimator. The full results, source scope and retained diagnostic history are in [the evidence report](evidence/README.md).

Use Python 3.12 with the tested packages in requirements.txt. Run `python run_reproduction.py --out regenerated` from this directory. No network, raw pickle input or external R execution is required. The command verifies the frozen inventory, regenerates every Firth fit and optimizer start, repeats the independent checks, and compares retained scientific outputs. The six primary CSV files must match byte-for-byte. JSON records are compared as objects so filesystem enumeration order cannot affect the result.

The original nine baseline settings per condition remain frozen inputs. This command regenerates the new comparator, not all earlier estimators or all historical Monte Carlo grids. Every source, transformation and outcome is labelled accordingly. Different local modes, numerical optimizer flags and conventional Wald intervals remain reported. The current manuscript integration is still outstanding; prepared passages are in evidence/prepared_manuscript_changes.md.
