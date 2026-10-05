# Corrected benchmark note

The benchmark was rerun after two specification corrections. The model design now includes an explicit opt-out utility indicator, and the heterogeneity condition uses unobserved respondent-specific random price sensitivity. This separates latent taste heterogeneity from observed income interactions.

The corrected 30-replication results preserve the main mechanism pattern. RF-assisted specification and fully structured MNL reach mean accuracy of 0.663 and 0.663 in the combined condition, compared with 0.607 for additive MNL. Mean decision regret is 0.033 for the structured models and 0.205 for additive MNL. In the heterogeneity-only condition, all three models have similar accuracy and regret because the added variation is unobserved; this condition now serves as a boundary test for what utility-term discovery can recover.

These numbers replace the earlier assisted-specification pilot for manuscript reporting. The earlier files remain as development records and should not be cited in the paper.

A two-class Latent Class MNL extension was then fitted with the same grouped split and metrics. In the combined condition its 10-replication mean accuracy was 0.591 and mean decision regret was 0.248, compared with 0.665 and 0.036 for the RF-assisted specification. In the heterogeneity-only condition the latent-class model did not recover the unobserved continuous price variation, which keeps the boundary interpretation explicit: class segmentation and utility-term discovery address different forms of preference complexity.
