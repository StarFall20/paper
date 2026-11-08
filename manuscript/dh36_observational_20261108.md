# Observational implementation of the paired-menu questionnaire

## Data and recruitment

The archive contained 360 complete questionnaire records, with 180 records per form and eight choice tasks per record. The platform labelled 30 records in each form as AI samples. The full-source analysis retained all 360 records. A source sensitivity analysis used the 300 records without the AI label. An additional analysis excluded one earlier submission from the latter group, leaving 299 records. The AI-labelled records retained their source designation throughout the analysis.

Different recruitment channels received different questionnaire forms. Each channel contained only one form, and no assignment record was available. Form and recruitment channel consequently share one indicator, so their effects cannot be estimated separately. The empirical comparison concerns the observed cohorts. A randomized test of candidate-summary sufficiency requires within-channel assignment of forms with a documented allocation mechanism.

The observed support preserves the declared candidate menu vector across forms. This support property establishes which contrasts the questionnaire could expose under an appropriate allocation. It does not establish exchangeability of the recruited cohorts.

## Estimable choice models

For alternatives A and B, we specify the marginal utility index

\[
V_{itj}=\alpha_A\mathbf1(j=A)+\alpha_B\mathbf1(j=B)
 +\beta_p(p_{tj}-89)/10
 +\gamma_1\mathbf1(G_{tj}=G1)+\gamma_2\mathbf1(G_{tj}=G2).
\]

The opt-out index is normalized to zero and marginal choice probabilities follow the multinomial logit map. The common model contains five parameters. A seven-parameter extension adds form/channel-specific alternative constants. A ten-parameter extension allows all five coefficients to differ across the observed cohorts. Coefficients are estimated by a marginal composite likelihood; uncertainty uses participant-clustered scores and a finite-sample correction.

Data storage, consultation support, reminder mode and research evidence remain fixed within each form/alternative pair. A specification containing their raw indicators, form-specific alternative constants, price and sharing indicators has 13 columns and rank 7. The six unidentified directions prevent separate estimation of these fixed attribute effects. The alternative constants absorb the fixed service configurations and other cohort differences.

Two flexible predictive baselines estimate task-specific probabilities, either pooled across cohorts or separately by cohort. Five-fold validation assigns each participant and all eight responses to one fold. Task probability estimates use a 0.5 pseudocount per category. Prediction targets new records from these observed cohorts on the same eight menus.

## Results

The unrestricted comparison uses a 16-dimensional vector of task-specific A/B probability differences. In the full-source sample, the participant-clustered Wald statistic is 28.047 with 16 degrees of freedom. The asymptotic p-value is 0.0312 and the centered participant bootstrap p-value is 0.0312, based on 4,999 draws. In the 300 records without the AI label, the statistic is 23.553, with p-values 0.0997 and 0.0988, respectively. Excluding the earlier submission gives an asymptotic p-value of 0.0943 and a bootstrap p-value of 0.0908. Source handling changes the result at a conventional 5% threshold. None of the eight individual task comparisons passes the Holm correction in the full-source sample.

The common MNL monthly-fee coefficient is -0.2115 per CNY 10 in the full-source sample, with a participant-clustered 95% interval of [-0.2379, -0.1851]. The corresponding estimate without the AI-labelled records is -0.2337, with interval [-0.2622, -0.2053]. Lower fees are associated with greater choice probability within this model and observed support. These estimates do not identify the separate effects of the four fixed service attributes.

The common MNL model has a five-fold mean log loss of 1.04176 in the full-source sample. Adding form/channel constants gives 1.04195, and allowing all coefficients to vary by cohort gives 1.04306. The paired loss difference for the latter extension is 0.00130, with an approximate interval of [-0.00233, 0.00492] conditional on this training split. In the sample without the AI label, the corresponding losses are 1.03238, 1.03311 and 1.03398. Cohort interactions do not improve average held-out prediction in this split. These predictive results concern the observed menu support.

## Implications for the proposed design audit

The example demonstrates respondent-level data preparation, rank checks, marginal choice-model estimation and sensitivity to sample provenance. The association between forms remains confounded with recruitment channel. The empirical results consequently support an observational implementation section. Randomized evidence for or against candidate-summary sufficiency remains an open requirement. A follow-up can recruit both forms within each channel, preserve anonymous allocation records and retain all eight responses as one participant cluster.

The archive is complete at the field level. Source validity, documented assignment and parameter identification require separate evidence. Additional records from the same two non-overlapping channel/form combinations do not resolve the channel/form confounding.

All uncertainty calculations use a questionnaire record as the cluster and assume independence across records. The generation mechanism and cross-record dependence of the AI-labelled records remain unverified. Full-source p-values therefore serve as numerical sensitivity outputs for a mixed-source archive, with no population interpretation established for those AI-labelled records.
