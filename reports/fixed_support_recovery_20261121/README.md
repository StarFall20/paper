# Fixed support and recovery from derivatives

The nine oracle constructions select the common support after the matched derivative order K is fixed. On each of those supports, exact derivatives recover the frequencies at order K + 1 under symmetry and at order K + 2 for unrestricted frequencies. The rank calculation uses rational elimination, a parity bound at lower orders and a separate full-rank verification over a prime field. This is a fixed-design qualification of the counterexamples.

Run this command from the repository root.

    python reports/fixed_support_recovery_20261121/verify.py reports/oracle_common_menu_20261121 --out recovered.json
    cmp recovered.json reports/fixed_support_recovery_20261121/results.json
