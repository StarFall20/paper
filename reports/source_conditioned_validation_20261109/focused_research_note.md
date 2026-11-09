# Source-conditional evaluation of repeated health-service choices

*Exploratory research note, 9 November 2026. The recorded source-service definition remains unresolved. This draft makes a dataset-specific diagnostic claim and no new-method priority claim.*

## Abstract

Panel choice models can reproduce the aggregate frequency of persistent nonpurchase while allocating it to the wrong source group. We examine this distinction in 360 exported questionnaire records, each containing eight health-service choices. Sixty records with one original source label choose the same option throughout; the other 300 all change options at least once. A source-blind outside-component model predicts 42 all-outside vectors, matching the observed total. Its source-specific predictions are 35.32 for the group containing none and 6.68 for the group containing all 42. A transparent source-aware pattern benchmark reduces these predictions to 0.62 and 40.42. Its improvement over a source-adjusted ordinary model is concentrated in the 60-record group; joint scores in the other 300 records are nearly identical. These results distinguish aggregate calibration from source-conditional prediction in this dataset. They do not identify the cause of repeated choices or establish respondent origin.

## 1. Question

Repeated health-service choices combine attribute comparisons with persistent participation patterns. Their joint likelihood is sensitive to both. When source groups have different answer-vector distributions, a pooled model can improve its overall score by representing their composition. The practical question is where that improvement occurs and whether predicted persistent nonpurchase is allocated to the groups in which it is observed.

Non-trading models already represent constant choices and distinguish possible mechanisms. [Kazagli and de Lapparent (2023)](https://doi.org/10.1016/j.jocm.2023.100413) provides the relevant JOCM comparison. [Mariel and Artabe (2025)](https://doi.org/10.1016/j.reseneeco.2025.101496) tests excess frequencies of specific choice patterns. The present analysis uses these established concerns to evaluate source-conditional predictions in an existing questionnaire.

## 2. Records and comparisons

Two questionnaire versions each supply 180 records and eight choices among service A, service B and nonpurchase. Monthly fees and sharing-rule packages vary; other service-card features are represented by version-specific alternative intercepts. Different recruitment channels administered the two versions, with no retained allocation log. Version effects are channel/version associations.

The original source field labels 60 records as “AI样本”. Authenticated platform inspection confirms two 30-record task notices and matching AI-point usage. The researcher reports a human-recruitment description; its applicability to these tasks remains unverified. The analysis therefore uses source labels literally, without claiming a human-versus-generated comparison. There are 15 all-A, three all-B and 42 all-outside vectors in this group. All other 300 vectors are nonconstant. Constant choices alone establish neither invalidity nor their behavioral explanation.

Nine fixed benchmarks include ordinary MNL, a normal random service intercept, an outside-component mixture, source interactions with alternative intercepts and a four-category pattern hurdle. Linear fees and unrestricted observed fee-rule cells are fixed specification comparisons. The hurdle assigns probabilities to all-A, all-B, all-outside and nonconstant vectors. Its nonconstant component uses the correctly conditioned MNL likelihood. This is a predictive diagnostic; its conditional coefficients are not population preference or WTP estimates.

Five folds retain each record's eight choices together. Pattern probabilities are learned from training counts, with fixed symmetric Dirichlet smoothing of 0.5 and a sensitivity of 1.0. Held-out predictions use menus, fitted parameters and source metadata. Held-out answers enter scoring, with no posterior class assignment used for prediction. The folds were previously used during project exploration and provide internal validation.

## 3. Aggregate and source-specific evidence

The source-blind outside-component model predicts 42.00 all-outside vectors in total. It allocates 35.32 to the 300-record group, where none occur, and 6.68 to the 60-record group, where all 42 occur. Thus its accurate overall count coexists with substantial source-conditional errors.

The source-aware hurdle predicts 0.62 and 40.42 all-outside vectors in the two groups. Its expected total of 41.04 reflects the fixed smoothing of unobserved training categories. In the linear-fee comparison, joint negative log scores per choice are 0.956676 for the outside component and 0.877234 for the source-aware hurdle. Their marginal scores are 1.040234 and 0.992546. These scores address different distributions: complete vectors and individual choices.

Against the source-adjusted ordinary model, the hurdle's joint score changes from 0.997330 to 0.877234 in the pooled sample. Scores in the other 300 records are 1.032544 and 1.032465; their descriptive paired difference spans zero. Approximately 99.94% of the weighted pooled mean gain comes from the 60-record group. Unrestricted fee-rule cells give the same conclusion, with a near-zero difference in the other-source group.

The matched source-aware and source-blind hurdles share the same conditional utility component. Their pooled score difference, 0.056494, comes entirely from their different pattern gates by construction. It cannot be attributed to improved attribute-utility estimation. All probabilities and marginal predictions were checked against exhaustive enumeration of the 6,561 possible vectors for both versions and gate groups. The aggregate tables and numerical checks accompany the note.

## 4. Interpretation

Source-stratified evaluation changes the interpretation of the pooled predictive gain in these records. The known source field and constant-vector structure account for the main difference; a richer utility distribution is not required to reproduce it. A claim of stable preference types needs evidence beyond the pooled score and overall nonpurchase count.

The analysis concerns one questionnaire with a sharply separated source pattern. It does not identify a recruitment effect, label a record as generated, or establish how often this issue occurs in other studies. The task-specific service definition must determine the final population statement. Constant choices may arise for several reasons, and their frequency alone cannot resolve them. The reusable result is a fully specified case comparison showing how overall calibration and source-conditional allocation answer different empirical questions.

## Appendix: Conditional pattern likelihood

Let p_tj be the MNL probability for alternative j at task t. A constant-j vector has probability c_j=product_t p_tj. For a nonconstant vector y, the conditional probability is product_t p_t,y_t divided by (1−sum_j c_j). Conditional marginal probabilities are (p_tj−c_j)/(1−sum_j c_j). Training counts estimate the four pattern probabilities, either pooled or by source. This disjoint-support construction normalizes the entire vector distribution and preserves explicit prediction inputs.

The comparison uses standard conditioning and categorical smoothing. These calculations are not claimed as an independently new theorem. A publication contribution must come from a distinct, adequately documented empirical finding and its consequences for choice-model use.
