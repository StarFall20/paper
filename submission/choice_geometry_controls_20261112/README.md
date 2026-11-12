# Choice geometry controls

This exploratory reproduction package contains fixed-cluster and positive-utility-scale controls for the same compensated choice-logit objective. The49-page delivered manuscript remains a separate artifact. The new derivations are candidate extensions under proof and primary-prior audit; no completed submission or independent-method certificate is claimed.

Start with [the positive-scale argument](evidence/radial/theory.md), [exact strict boundary comparison](evidence/radial/exact_counterexample_check.json), [verified radial counts](evidence/radial/verification/phase_summary_verified.csv), [cluster controls](evidence/cluster/README.md) and [primary-source audit](evidence/radial/closest_prior_and_access.json). All raw simulation outcomes, numerical unknowns, negative directions and repaired verification tables are included. There are no private questionnaires or downloaded research-data inputs.

Use the pinned requirements with Python3.12. Commands run from this directory. Numerical verifiers need no network or external data.

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
python verify_package.py
python analysis/verify_cluster_compensation.py --root evidence/cluster --plan analysis/cluster_compensation_plan_20261112.json --out regenerated/cluster
python analysis/verify_radial_compensation.py --root evidence/radial --plan analysis/radial_compensation_plan_20261112.json --out regenerated/radial
```

These replay384stratified cluster events and all6000radial criteria, including the2042radial negative recession directions and39independent integrals. Cluster and radial covariance checks concern full population cluster scores and fixed-dimensional oracle covariance, respectively. No high-dimensional risk or confidence-interval advantage is claimed.

For complete new generation, use fresh output directories:

```sh
python analysis/cluster_compensated_boundary.py --plan analysis/cluster_compensation_plan_20261112.json --out regenerated/cluster_population
python analysis/cluster_compensation_phase.py --plan analysis/cluster_compensation_plan_20261112.json --population regenerated/cluster_population --out regenerated/cluster_phase
python analysis/replay_cluster_population_followups.py --root evidence/cluster --plan analysis/cluster_compensation_plan_20261112.json --stage refinement --out regenerated/refinement
python analysis/replay_cluster_population_followups.py --root evidence/cluster --plan analysis/cluster_compensation_plan_20261112.json --stage matched --out regenerated/matched
python analysis/radial_compensation_audit.py --plan analysis/radial_compensation_plan_20261112.json --out regenerated/radial_generation
```

The full cluster run uses1000independent model draws and24000support classifications; the radial run uses1000independent draws and6000criterion records. Conditions share random draws as stated in the plans and are not independent across dimensions or powers. Matched ordinary cluster screening remains exploratory. The strict radial analytic comparison and the milder finite simulation use different explicitly specified scale laws.

The dependency closure includes original project files for the classical attainable-moment audit and independent-row reference. They are imported for computation, not for data loading. Original PPI tuning, conic geometry, ordinary correlated-pattern capacity and scale-identification limits retain the citations in the derivations. The external cluster-PPI repository was read at a pinned commit; its code is not copied or executed here.
