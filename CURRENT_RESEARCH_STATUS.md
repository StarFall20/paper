# Current research status — 9 November 2026, after AI-pretest clarification

The source-definition barrier is resolved. The researcher supplied the exact platform service wording: AI samples answer the questionnaire for preliminary research/pretesting. Matching task notices and counts had already been inspected. The explanation itself is researcher-provided, without a freshly identified webpage claim.

## Correct analysis populations

- **Human primary analysis:** 300 human-recruitment records, 150 per version,2400 choices. These use ordinary submission metadata and the recruitment report; demographic representativeness and all eligibility claims are not certified.
- **AI pretest records:**60,30 per version. They do not enter human preference estimates, human sample size or human significance evidence.
- **Original mixed export:**360 records preserved unchanged. Earlier pooled fits are historical mixed-source diagnostics, not 360 human respondents.

The canonical definition is [data/dh36_source_classification.json](data/dh36_source_classification.json). No further device unlock or repeated provenance question is needed to classify these 60 records. The earlier general human-recruitment interpretation is superseded for these tasks.

## Completed human primary reconstruction

The [human-primary report](reports/human_primary_20261109/真人主分析与来源纠正.md) re-estimates ordinary and random-intercept models with linear fees or free observed fee-rule cells. Whole-record folds are retained. Human joint NLL per choice is 1.031745 for ordinary linear fees and 1.031899 for a normal random intercept; increased flexibility does not improve the current human prediction target.

At CNY 89, ordinary free cells estimate G2–G1 at 0.503 with interval[0.029,0.977]; random-intercept cells give 0.499,[0.024,0.975]. Conditional A/B gives 0.787,[-0.184,1.759]. The early-record 299 sensitivity remains reported without declaring that record invalid. These are exploratory model-conditional package comparisons, not pure-control effects or established WTP.

The [current human application draft](manuscript/human_service_choice_application_20261109.md) uses 300 records. Different recruitment channels administered one version each, without a retained assignment log; no causal version effect is claimed.

## AI-pretest answer pairing: retained case, independent-method positioning withdrawn

The [computational audit](reports/synthetic_pairing_audit_20261109/AI预测试配对审计.md) fixes human training/test vectors, menus, AI counts and every training task-by-version choice count. It re-pairs only AI training answers. All 50 exact marginal controls pass. For the specified MNL, coefficient point estimates and human predictions remain invariant; clustered uncertainty is not asserted invariant.

With original AI pretests added, human-target joint NLL is 1.064631 for a random service intercept and 1.049832 for the outside component. Across ten fixed re-pairings, means are 1.044161 and 1.042139. Mean outside weight drops from 11.66% to 1.14% at unchanged margins. These are consequences of the defined data operation, not a change in human tastes or ten independent AI batches.

The [current case manuscript](manuscript/ai_pretest_pairing_audit_20261109.md) develops this specific consequence for inferred participation heterogeneity and human prediction. AI pretests are not proposed as replacement respondents. The service is explicitly intended for pretesting; the note does not claim the provider promotes population inference from them.

## Human negative controls and numerical repair

The300 human-recruitment records alone supply55 additional fixed-margin controls. Independent task re-pairing leaves MNL unchanged and has a small effect on the random intercept. A deliberately extreme rank alignment gives human joint NLL1.220683 for the random intercept and1.253002 for the outside component, while MNL stays1.031745. Different source margins prevent an AI-specific causal interpretation of effect magnitudes.

All five extreme random-intercept fits hit the fixed standard-deviation upper bound7.389056. Their initial61-node quadrature was inadequate. Re-estimation at1281 nodes and verification at2561 reduce maximum training log-likelihood discrepancy below7.3e-7. Use `human_negative_control_validated_summary.csv`; lower-order stress outputs remain an audit trail. These repairs do not change the primary human analysis or original AI-pairing results.

## New within-person assumption audit

The [within-person report](reports/within_person_audit_20261109/个体内模型重建与创新边界.md) uses classical conditional multinomial logit to remove stable person-specific alternative intercepts. Free observed cells give G2 minus G1 at CNY89 of0.497299, with interval[0.025488,0.969110]. Linear fees give0.358422,[0.219558,0.497285]. Both rely on shared slopes and normalized logit scale. Cross-task identification and the bundled sharing description remain limitations.

All300 human records contribute to conditional estimation. The60 constant AI vectors have singleton count strata and contribute zero to this conditional likelihood; six objective/score checks confirm the exact bounded invariance. This is a classical property, not a new AI filter. Conditional likelihood does not evaluate unconditional participation or new-person prediction.

There are56 pairwise participation-association tests across the two versions, with22 negative sample covariances and no Holm-adjusted rejection. Nonrejection does not establish latent-model adequacy or power. Two conditional models, full enumeration, analytic derivatives and direct conditioning checks are documented. No result is selected by significance.

The [external-resource preflight](reports/within_person_audit_20261109/external_resource_preflight.json) verifies eight openly licensed files. A human DCE with299 records and1790 observed tasks is structurally readable; no external preference model or validation has been run. The inspected synthetic-response supplement lacks files needed for its script, and the PriceBench README states that its58 paired-output CSVs are uncommitted. No new AI generation or author message was made.

## Public-choice reanalysis and a new finite-solution candidate

The [new audit](reports/public_conjoint_audit_20261109/公开数据重分析与创新进展.md) uses a pinned, openly released vaccine-choice dataset. A stable-loss fit on4605 human rows finds that the released reference matches200-step Adam within1.49e-7 but differs from the converged projection by up to0.185492 in a coefficient; total NLL can fall7.013964. This is a convergence finding, not a reproduction or dismissal of all reported gains.

Component clipping of inverse-propensity-corrected one-hot targets exactly returns observed labels for labelled records and g for unlabelled records. The fixed-prediction identity passes all2000 comparisons. A500-allocation, six-method, four-budget controlled reanalysis is complete on the fixed1200-row prefix. At50 labelled choice rows, the unmodified AI correction has5/500 exterior moments and the X-only correction2/500; all seven have separate positive-gap recession-ray verification. Failure counts accompany finite-fit point-error summaries. Clipped soft training's smaller point error is retained alongside its discrepancy from the theoretical corrected score. No interval-coverage or PPI++ superiority claim is made.

The [candidate derivation](reports/public_conjoint_audit_20261109/existence_candidate_derivation.md) gives an exact finite-label probability and a frozen low-index Gaussian boundary depending on signal and Brier error.160 exact-probability and480 signal/predictor conditions use2000 original-X,Y draws each; population integration checks and the independent finite-pool argument are recorded. Gaussian lift-zonoid geometry, generic separation LPs, ordinary-logit phase transitions and corrected-risk issues are attributed prior work. The candidate's priority and substantive increment remain unproven. This is a separate estimator question, not a revival of fibre ranks or a combination of unrelated case studies.

The [working draft](manuscript/rectified_choice_finite_solution_working_draft_20261109.md) has a single research question. The next substantive test concerns established power tuning, finite auxiliary pools and valid treatment of nonfinite roots. Public source and file hashes, objective definitions, plans, numerical checks and aggregate outputs are documented; raw questionnaire exports remain private.

## Innovation and submission decision

The original fibre-information/rank claim coincides with classical lack-of-fit constructions and remains retired. [Equivalence proof](reports/method_decision_20261109/classical_equivalence_proof.md).

The [updated originality review](reports/synthetic_pairing_audit_20261109/originality_review_20261109.md) checks the closest synthetic shuffle precedent and the classical panel likelihood. Human-only fixed-margin stress controls also change panel estimates. This is generic dependence sensitivity. The exact realized DCE margins and downstream estimation remain case-specific differences; they do not establish an independent new method. The case record is retained and its independent-method positioning withdrawn. A sufficient innovative application remains unproven. The goal is not complete; see the [completion assessment](reports/synthetic_pairing_audit_20261109/completion_assessment.json).

All raw records remain private and unchanged. No new questionnaire, AI generation, purchase or external message was made. Historical drafts have supersession notices; they retain their prior information state and numerical results. Current reproduction starts from the new human-primary and pairing-audit READMEs.
