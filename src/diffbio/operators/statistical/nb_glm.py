"""Differentiable Negative Binomial GLM operator.

This module provides a differentiable implementation of the negative
binomial generalized linear model for differential expression analysis,
inspired by DESeq2.

Key technique: Parameterize the NB mean through a log-linear model
and estimate dispersion parameters per gene.
"""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array, Float, PyTree


@dataclass
class NBGLMConfig(OperatorConfig):
    """Configuration for DifferentiableNBGLM.

    Attributes:
        n_features: Number of features (genes).
        n_covariates: Number of covariates in design matrix.
        estimate_dispersion: Whether to estimate dispersion parameters.
        stochastic: Whether the operator uses randomness.
        stream_name: RNG stream name (not used).
    """

    n_features: int = 2000
    n_covariates: int = 2
    estimate_dispersion: bool = True
    stochastic: bool = False
    stream_name: str | None = None


class DifferentiableNBGLM(OperatorModule):
    """Differentiable Negative Binomial GLM for differential expression.

    This operator implements a negative binomial GLM where:
    - log(mu) = X @ beta (design matrix @ coefficients)
    - P(count | mu, dispersion) = NB(count; mu, dispersion)

    Gradients flow through both the coefficient (beta) and dispersion
    parameters, enabling end-to-end learning.

    The negative binomial distribution is parameterized as:
    - mean = mu
    - variance = mu + mu^2 / dispersion

    Args:
        config: NBGLMConfig with model parameters.
        rngs: Flax NNX random number generators.
        name: Optional operator name.

    Example:
        >>> config = NBGLMConfig(n_features=2000, n_covariates=2)
        >>> glm = DifferentiableNBGLM(config, rngs=nnx.Rngs(42))
        >>> data = {"counts": counts, "design": design_row, "size_factor": sf}
        >>> result, state, meta = glm.apply(data, {}, None)
    """

    def __init__(
        self,
        config: NBGLMConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize the NB GLM operator.

        Args:
            config: NB GLM configuration.
            rngs: Random number generators for initialization.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        if rngs is None:
            rngs = nnx.Rngs(0)

        self.n_features = config.n_features
        self.n_covariates = config.n_covariates

        # Initialize coefficients (beta)
        # Shape: (n_covariates, n_features)
        key = rngs.params()
        init_beta = jax.random.normal(
            key, (config.n_covariates, config.n_features)
        ) * 0.1
        self.beta = nnx.Param(init_beta)

        # Initialize log dispersion parameters
        # Shape: (n_features,)
        # Start with dispersion = 1.0 (log_dispersion = 0)
        self.log_dispersion = nnx.Param(jnp.zeros(config.n_features))

    def get_coefficients(self) -> Float[Array, "n_covariates n_features"]:
        """Get coefficient matrix.

        Returns:
            Coefficient matrix beta (n_covariates, n_features).
        """
        return self.beta[...]

    def get_dispersion(self) -> Float[Array, "n_features"]:
        """Get dispersion parameters.

        Returns:
            Dispersion parameters (n_features,), always positive.
        """
        # Use softplus to ensure positivity
        return jax.nn.softplus(self.log_dispersion[...]) + 1e-4

    def predict_mean(
        self,
        design: Float[Array, "n_covariates"],
        size_factor: Float[Array, ""],
    ) -> Float[Array, "n_features"]:
        """Predict mean expression for a sample.

        Args:
            design: Design matrix row for this sample.
            size_factor: Library size normalization factor.

        Returns:
            Predicted mean expression (n_features,).
        """
        beta = self.get_coefficients()

        # log(mu) = design @ beta
        # design is (n_covariates,), beta is (n_covariates, n_features)
        log_mu = jnp.dot(design, beta)

        # Add size factor effect
        log_mu = log_mu + jnp.log(size_factor + 1e-8)

        # Exponentiate to get mean
        mu = jnp.exp(log_mu)

        return mu

    def negative_binomial_log_prob(
        self,
        counts: Float[Array, "n_features"],
        design: Float[Array, "n_covariates"],
        size_factor: Float[Array, ""],
    ) -> Float[Array, ""]:
        """Compute negative binomial log probability.

        Uses the parameterization where:
        - variance = mu + mu^2 / dispersion

        Args:
            counts: Observed counts.
            design: Design matrix row.
            size_factor: Size factor for normalization.

        Returns:
            Total log probability (scalar).
        """
        mu = self.predict_mean(design, size_factor)
        dispersion = self.get_dispersion()

        # Negative binomial log probability
        # NB(k; mu, r) where r = dispersion
        # log P(k) = log(Gamma(k + r)) - log(Gamma(k + 1)) - log(Gamma(r))
        #          + r * log(r / (r + mu)) + k * log(mu / (r + mu))

        r = dispersion
        k = counts

        # Compute log probability using the NB PMF
        # Use jax.scipy.special functions for numerical stability
        log_prob = (
            jax.scipy.special.gammaln(k + r)
            - jax.scipy.special.gammaln(k + 1)
            - jax.scipy.special.gammaln(r)
            + r * jnp.log(r / (r + mu + 1e-8))
            + k * jnp.log(mu / (r + mu + 1e-8) + 1e-8)
        )

        # Sum over features
        total_log_prob = jnp.sum(log_prob)

        return total_log_prob

    def batch_log_likelihood(
        self,
        counts: Float[Array, "n_samples n_features"],
        design: Float[Array, "n_samples n_covariates"],
        size_factors: Float[Array, "n_samples"],
    ) -> Float[Array, ""]:
        """Compute log likelihood for a batch of samples.

        Args:
            counts: Count matrix (n_samples, n_features).
            design: Design matrix (n_samples, n_covariates).
            size_factors: Size factors (n_samples,).

        Returns:
            Total log likelihood.
        """
        def sample_log_prob(args):
            c, d, s = args
            return self.negative_binomial_log_prob(c, d, s)

        log_probs = jax.vmap(sample_log_prob)((counts, design, size_factors))
        return jnp.sum(log_probs)

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Apply NB GLM to count data.

        This method computes the log likelihood and predicted mean
        for a given sample.

        Args:
            data: Dictionary containing:
                - "counts": Gene counts (n_features,)
                - "design": Design matrix row (n_covariates,)
                - "size_factor": Library size factor (scalar)
            state: Element state (passed through unchanged)
            metadata: Element metadata (passed through unchanged)
            random_params: Not used (deterministic operator)
            stats: Not used

        Returns:
            Tuple of (transformed_data, state, metadata):
                - transformed_data contains:
                    - "counts": Original counts
                    - "log_likelihood": Log probability of counts
                    - "predicted_mean": Predicted expression
                    - "dispersion": Dispersion parameters
                - state is passed through unchanged
                - metadata is passed through unchanged
        """
        counts = data["counts"]
        design = data["design"]
        size_factor = data["size_factor"]

        # Compute log likelihood
        log_likelihood = self.negative_binomial_log_prob(counts, design, size_factor)

        # Compute predicted mean
        predicted_mean = self.predict_mean(design, size_factor)

        # Get dispersion
        dispersion = self.get_dispersion()

        # Build output data
        transformed_data = {
            "counts": counts,
            "design": design,
            "size_factor": size_factor,
            "log_likelihood": log_likelihood,
            "predicted_mean": predicted_mean,
            "dispersion": dispersion,
        }

        return transformed_data, state, metadata
