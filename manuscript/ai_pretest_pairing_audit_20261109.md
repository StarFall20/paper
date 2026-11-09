# AI pretest answer pairing and inferred participation heterogeneity: a health-service DCE case audit

*Exploratory research manuscript, 9 November 2026. A bounded computational case study; independent novelty and submission readiness remain under assessment.*

## Abstract

Synthetic pretest responses may enter a panel choice dataset with dependence patterns that differ from human submissions. We audit 300 human-recruitment records and 60 platform AI-pretest records, each containing eight health-service choices. All AI vectors select one option throughout; none of the human vectors do. Adding the original AI batch worsens human-target joint negative log scores from 1.031899 to 1.064631 for a random service-intercept model and from 1.031745 to 1.049832 for an outside-component model. A computational control re-pairs only training AI answers within version and task, preserving every task's choice counts. Ordinary MNL estimates and human predictions remain invariant. Across ten fixed re-pairings, human-target scores average 1.044161 and 1.042139 for the two panel models; the fitted outside weight falls from 11.66% to 1.14%. Thus the pairing of synthetic answers changes inferred heterogeneity and predictive performance at fixed margins. The result concerns one pretest deployment and does not establish a general AI-generator mechanism.

## 1. Question

Panel choice models use the joint sequence of an individual's answers. AI pretests provide complete answer vectors whose joint structure can differ from human submissions. The question here is what changes in panel estimation when synthetic answer pairing changes and every observed task margin remains fixed.

Recent DCE work already compares synthetic and human choices, including utility-scale differences and pretest design uses. [Califano and Caracciolo (2026)](https://doi.org/10.1016/j.foodqual.2026.105930) supplies a close comparison across three food DCEs. [Barrie and Cerina's 2026 working paper](https://csmapnyu.org/research/academic-research) also documents dependence differences in synthetic survey belief systems. Non-trading rules are established in [Kazagli and de Lapparent (2023)](https://doi.org/10.1016/j.jocm.2023.100413). The present operation isolates the consequence of answer pairing for specified panel choice models and a human prediction target. It is not a new conditioning theorem or a first claim that synthetic answers differ from humans.

## 2. Data and source definition

Two versions contribute 150 human-recruitment submissions each. The platform additionally produced 30 AI-pretest records per version. The researcher supplied the service wording explicitly describing AI samples answering questionnaires for preliminary research/pretesting. Prior authenticated inspection verified the two task counts, original source labels and AI-point records. The wording itself is researcher-provided. The AI model, prompting, persona-selection and decoding configuration are unavailable; no particular LLM is assumed.

The publication archive contains rendered service cards and records eight uploaded cards per version. The generator's representation of these cards is unavailable. Input access, parsing, profile construction and output handling remain possible explanations for the constant vectors; this audit does not attribute them to an AI preference mechanism.

The AI batch contains 15 all-A, three all-B and 42 all-outside vectors. All 300 ordinary-submission vectors change option at least once. The source definition identifies the AI records; constant choices are not used as an origin detector. Recruitment channels differ by version, so source contrasts do not identify population-level causal differences between matched human and AI respondents.

The platform describes a pretest use. The mixed-data conditions below deliberately examine the computational consequence of pooling those records for estimation; they are not a claim that the service recommends doing so. Human preference estimation uses only the 300 human-recruitment records.

## 3. Fixed computational comparison

We compare ordinary MNL, a normal random service intercept and a classical outside-component mixture, using the existing version-adjusted linear-fee utility specification. In each of the five existing record folds, menus, human training vectors and human held-out vectors remain fixed. Conditions use human training records alone, the original training AI batch, or re-paired training AI answers. The human target contains 300 original held-out records across folds.

Re-pairing independently permutes each AI task column within version and training fold. It preserves all task-by-version choice counts, sample size and source composition. Ten seeds are fixed in the exploratory plan. These controls are generated arrangements, not newly collected responses or corrected human data.

For the specified MNL, AI feature vectors are identical within version and task. Its log likelihood is a sum over those task/alternative choice counts. Preserving those cells preserves its objective and gradient, coefficient point estimates and human predictions. Record-clustered uncertainty is not claimed to remain invariant. This established invariance is an experimental control. Panel likelihoods depend on the pairing of answers and need not remain invariant. All 50 training controls pass, with maximum objective difference 4.5e-13 and gradient difference 4.2e-14.

## 4. Results

| Training condition | Model | Mean human joint NLL | Minimum | Maximum |
| --- | --- | --- | --- | --- |
| Human training only | MNL | 1.031745 | 1.031745 | 1.031745 |
| Human training only | Random service intercept | 1.031899 | 1.031899 | 1.031899 |
| Human training only | Outside component | 1.031745 | 1.031745 | 1.031745 |
| Original AI pretests added | MNL | 1.042676 | 1.042676 | 1.042676 |
| Original AI pretests added | Random service intercept | 1.064631 | 1.064631 | 1.064631 |
| Original AI pretests added | Outside component | 1.049832 | 1.049832 | 1.049832 |
| AI answers re-paired | MNL | 1.042676 | 1.042676 | 1.042676 |
| AI answers re-paired | Random service intercept | 1.044161 | 1.043981 | 1.044377 |
| AI answers re-paired | Outside component | 1.042139 | 1.041655 | 1.042524 |

With the original mixed batch, random-intercept and outside-component models predict human vectors less accurately than the same models trained on human records alone. The mixed validation target gives a different ranking: its joint scores are 1.040306 for MNL, 0.981693 for the random intercept and 0.956676 for the outside component. Evaluating only original human held-out records gives 1.042676, 1.064631 and 1.049832. A score averaged over human and AI records answers a different question from human prediction.

Re-pairing changes none of MNL's task counts or human predictions. It reduces the random-intercept human-target loss to a mean 1.044161 (range 1.043981–1.044377), and the outside-component loss to 1.042139 (1.041655–1.042524). Mean fitted outside weight across training folds falls from 0.116622 with original AI pairing to 0.011382 with re-pairing; mean random-intercept standard deviation falls from 1.355882 to 0.722955. Human-only fits have means near zero for the outside weight and 0.221594 for the random-intercept deviation. These are fitted summaries, not changes in real consumers' tastes.

The remaining human loss after re-pairing shows that changing dependence does not make the AI margins a suitable human data source. The computational comparison isolates one sensitivity of panel estimates; it does not authorize pooling repaired AI records into human inference.

## 5. Interpretation and scope

This case distinguishes task margins, answer pairing and the prediction population. The fitted heterogeneity from mixed records partly represents the construction of the synthetic pretest batch. A corresponding improvement on a mixed validation target does not establish improved human preference recovery.

Ten re-pairings measure sensitivity to the stated operation, without a permutation p-value or a claim of ten independent AI deployments. The questionnaire, service and two batches are fixed. Unreported generator settings, convenience recruitment and reused folds limit transfer. The note's proposed empirical contribution is the measured panel-estimation consequence at exactly preserved margins. Its difference from existing synthetic-response and non-trading studies requires continued assessment; no worldwide-priority or acceptance claim is made.

## Reproducibility

The package contains source classification, the fixed plan, aggregate fold scores and parameter summaries, and exact marginal-control checks. Original answer vectors, identifiers, source-detail names, timestamps, IPs and derived individual controls remain private. Re-estimation requires authorized access to the two frozen exports and existing record folds. All 300 human submissions remain in the primary sample; an independently flagged early-record sensitivity is reported separately.
