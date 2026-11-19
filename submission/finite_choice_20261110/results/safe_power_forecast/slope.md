# Local effect of power on choice forecasts

Let $H(\beta)$ be the labelled human loss, and let $P_U(\beta)$ and $P_L(\beta)$ be auxiliary and labelled losses evaluated on the frozen predictions. Write $Q_\lambda=H+\lambda(P_U-P_L)$. At a finite unpenalized labelled MNL solution $\hat\beta_0$, assume positive definite labelled curvature $J_0=\nabla^2 H(\hat\beta_0)$.

The implicit function theorem applied to the corrected score gives

$$
\left.\frac{d\hat\beta_\lambda}{d\lambda}\right|_0
=J_0^{-1}\{\nabla P_L(\hat\beta_0)-\nabla P_U(\hat\beta_0)\}.
$$

For a fixed respondent-averaged human evaluation loss $V$, the derivative is the inner product of this coefficient velocity with $\nabla V(\hat\beta_0)$. It measures the local movement of predictions on those evaluation respondents. The existence cap depends on moment support in the fitting samples. Either sign is possible for the evaluation inner product, which also uses the evaluation responses.

In the declared diagnostic, $\lambda=0.01\epsilon_*$ gives a small positive secant. The original rows reproduce all derivatives and finite secants using a second-moment Hessian and a separate implementation of the three score terms. The analytic and secant signs agree in all 656 available cases. Their magnitudes can differ at this fixed step. The comparison with the full half-cap change uses endpoint predictions and does not rely on extrapolating the linear derivative.
