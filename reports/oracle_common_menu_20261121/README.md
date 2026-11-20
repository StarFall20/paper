# Matching choice derivatives at a correctly specified model

This construction holds the observed human responses, all available menus, the auxiliary row count, an informative predictor and correction weight 0.5 fixed. The predictor is exactly the conditional choice probability in the construction, with utility slope log 2. For any preassigned finite derivative order, the three frequency designs agree through that order at the true utility coefficient. Their outcomes are an unbounded criterion, a finite unattained infimum and a unique finite solution. All labelled profiles have both outcomes and a finite ordinary logit solution.

The exact rational program verifies positive derivative orders 1 through 9. The witness counts can be large. These calculations establish finite-sample possibility under a correctly specified prediction model. They give no rate at which the selected label configurations occur.

    python verify.py --out results.json

The proof uses independence of a directional tail cost from a finite set of derivatives of the logit log partition. The result concerns the empirical corrected objective under a fixed design. It can coexist with regular asymptotic consistency when the labelled and auxiliary observations come from a common distribution.
