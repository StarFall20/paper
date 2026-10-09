# Candidate-preserving DCE fielding protocol

This document is for the research team. Respondents receive only the two respondent-facing Word files; the candidate summary and the purpose of the two versions are not shown.

## Randomization and record

Assign respondents independently to Version 1 or Version 2 with probability 1/2. Record a study identifier, version, block, task, the complete displayed menu, selected alternative, response time, completion state, and any prespecified exclusion flag. Do not collect names unless the approved consent explicitly covers them. Fill the team contact and ethics reference in the respondent form before fielding.

## Candidate-preserving check

The declared candidate summary for each alternative is `(data-management level + professional-support level, intelligence level + clinical-evidence level)`. Version 1 uses A `(D0,S2,I0,E2)` and B `(D1,S2,I0,E1)`. Version 2 uses A `(D1,S1,I1,E1)` and B `(D2,S1,I1,E0)`. Every task keeps the sharing attribute, price, alternative labels, opt-out option, and task order fixed. `analysis/verify_candidate_preserving_questionnaire.py` checks all 16 displayed alternative cards; the locked result is `results/questionnaire_candidate_preservation_audit.csv`.

The composite candidate summary is an exploratory index for this pilot. Its substantive interpretation must be reviewed in cognitive interviews and preregistered before a confirmatory field study. Mathematical preservation alone does not validate the four attribute labels as a measurement scale.

Before launch, run a second check against the actual survey-platform export. Require equality of the complete candidate vector for every alternative and task, positive support for both orientation versions, and a prespecified minimum cell count. A pilot should test comprehension and perceived difficulty before the main sample.

## Analysis pre-registration

The primary estimand is the difference in choice probabilities between versions conditional on the preserved candidate menu vector. Use respondent-clustered randomization inference with the version assignment as the randomization unit. Report the generalized fibre-stratified score and complete-fibre LR as parallel diagnostics; for a binary response they test the same conditional-independence null through score and deviance forms. Include a sensitivity analysis for opt-out handling, response-time exclusions, and task order. A significant version contrast rejects the conditional-sufficiency relation on this support. It does not, by itself, identify framing, complexity, scale, or a single psychological mechanism.
