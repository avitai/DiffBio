"""Base operator classes for DiffBio.

This module provides domain-specific base classes that operators can inherit
to get shared functionality. Following DRY principle, common patterns like
temperature-controlled smoothing, sequence validation, and VAE reparameterization
are centralized here.

Inheritance patterns:
- TemperatureOperator: For operators using logsumexp smoothing
- SequenceOperator: For operators processing one-hot encoded sequences
- EncoderDecoderOperator: For VAE-style operators
- GraphOperator: For GNN-based operators
- HMMOperator: For HMM-based operators with forward-backward

Multiple inheritance is supported:
    class MyAligner(TemperatureOperator, SequenceOperator):
        ...
"""

from typing import Any, Literal

import jax
import jax.numpy as jnp
from datarax.core.config import OperatorConfig
from datarax.core.operator import OperatorModule
from flax import nnx
from jaxtyping import Array, Float, Int, PyTree

from diffbio.constants import DEFAULT_TEMPERATURE, EPSILON
from diffbio.core.soft_ops import sorting as soft_sorting
from diffbio.utils.nn_utils import ensure_rngs, get_rng_key, init_learnable_param

__all__ = [
    "TemperatureOperator",
    "SequenceOperator",
    "EncoderDecoderOperator",
    "GraphOperator",
    "HMMOperator",
]


class TemperatureOperator(OperatorModule):
    """Base class for operators using temperature-controlled smoothing.

    Provides the soft_max method using logsumexp relaxation, which is used
    by many differentiable bioinformatics algorithms including:
    - Smith-Waterman alignment
    - Nussinov RNA folding
    - Viterbi decoding

    The temperature parameter controls the trade-off between accuracy and
    differentiability:
    - temperature -> 0: Approaches hard max (accurate but less differentiable)
    - temperature -> inf: Uniform averaging (smooth but uninformative)

    Subclasses should define their config with a 'temperature' field.
    """

    def __init__(
        self,
        config: OperatorConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize TemperatureOperator.

        Args:
            config: Configuration with 'temperature' and optionally
                'learnable_temperature' fields.
            rngs: Flax NNX random number generators.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        temperature = getattr(config, "temperature", DEFAULT_TEMPERATURE)
        is_learnable = getattr(config, "learnable_temperature", False)

        if is_learnable:
            self.temperature = init_learnable_param(temperature)
        else:
            self._temperature_value = temperature

    @property
    def _temperature(self) -> Float[Array, ""] | float:
        """Get current temperature value."""
        if hasattr(self, "temperature"):
            return self.temperature[...]
        return self._temperature_value

    def soft_max(
        self,
        values: Float[Array, "..."],
        axis: int | None = None,
    ) -> Float[Array, "..."]:
        """Compute smooth maximum using logsumexp.

        Uses ``temperature * logsumexp(values / temperature)`` which
        is an upper bound on the true max. This property is essential
        for dynamic programming algorithms (Smith-Waterman, Viterbi).

        Args:
            values: Input array.
            axis: Axis along which to compute max.

        Returns:
            Smooth maximum value(s), always >= hard max.
        """
        temp = self._temperature
        return temp * jax.scipy.special.logsumexp(values / temp, axis=axis)

    def soft_argmax(
        self,
        logits: Float[Array, "..."],
        axis: int = -1,
    ) -> Float[Array, "..."]:
        """Compute soft argmax returning SoftIndex.

        Delegates to :func:`diffbio.core.soft_ops.sorting.argmax`.

        Args:
            logits: Input logits.
            axis: Axis along which to compute argmax.

        Returns:
            SoftIndex probability distribution.
        """
        return soft_sorting.argmax(
            logits,
            axis=axis,
            softness=self._temperature,
            mode="smooth",
            standardize=False,
        )

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Base apply method - should be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement apply()")


class SequenceOperator(OperatorModule):
    """Base class for operators processing biological sequences.

    Provides utilities for sequence validation and manipulation including:
    - Validation of one-hot encoded sequences
    - Sequence normalization to valid probability distributions
    - Alphabet handling

    Subclasses should define their config with 'alphabet_size' and
    optionally 'max_length' fields.
    """

    def __init__(
        self,
        config: OperatorConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize SequenceOperator.

        Args:
            config: Configuration with 'alphabet_size' field.
            rngs: Flax NNX random number generators.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        self.alphabet_size = getattr(config, "alphabet_size", 4)
        self.max_length = getattr(config, "max_length", None)

    def validate_sequence(
        self,
        sequence: Float[Array, "length alphabet"],
    ) -> bool:
        """Check if sequence has valid shape for this operator.

        Args:
            sequence: Sequence to validate.

        Returns:
            True if sequence is valid.
        """
        if sequence.ndim != 2:
            return False
        if sequence.shape[1] != self.alphabet_size:
            return False
        if self.max_length is not None and sequence.shape[0] > self.max_length:
            return False
        return True

    def normalize_sequence(
        self,
        sequence: Float[Array, "length alphabet"],
    ) -> Float[Array, "length alphabet"]:
        """Normalize sequence so each position sums to 1.

        Args:
            sequence: Possibly unnormalized sequence.

        Returns:
            Normalized sequence (valid probability distribution at each position).
        """
        return jax.nn.softmax(sequence, axis=-1)

    def mask_sequence(
        self,
        sequence: Float[Array, "length alphabet"],
        mask: Float[Array, "length"],
    ) -> Float[Array, "length alphabet"]:
        """Apply mask to sequence.

        Args:
            sequence: Input sequence.
            mask: Boolean or soft mask.

        Returns:
            Masked sequence (masked positions set to uniform).
        """
        uniform = jnp.ones(self.alphabet_size) / self.alphabet_size
        return jnp.where(mask[:, None], sequence, uniform)

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Base apply method - should be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement apply()")


class EncoderDecoderOperator(OperatorModule):
    """Base class for VAE-style encoder-decoder operators.

    Provides utilities for variational autoencoders:
    - Reparameterization trick for sampling
    - KL divergence computation
    - ELBO loss components

    Used by single-cell analysis, normalization, and generative models.

    Subclasses should define their config with 'latent_dim' and
    'hidden_dim' fields.
    """

    def __init__(
        self,
        config: OperatorConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize EncoderDecoderOperator.

        Args:
            config: Configuration with 'latent_dim' field.
            rngs: Flax NNX random number generators.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        self.latent_dim = getattr(config, "latent_dim", 10)
        self.hidden_dim = getattr(config, "hidden_dim", 64)
        self.rngs = ensure_rngs(rngs)

    def reparameterize(
        self,
        mean: Float[Array, "... latent_dim"],
        log_var: Float[Array, "... latent_dim"],
    ) -> Float[Array, "... latent_dim"]:
        """Sample from latent distribution using reparameterization trick.

        z = mean + std * epsilon, where epsilon ~ N(0, 1)

        This allows gradients to flow through the sampling operation.

        Args:
            mean: Mean of the latent distribution.
            log_var: Log variance of the latent distribution.

        Returns:
            Sampled latent representation.
        """
        key = get_rng_key(self.rngs, "sample", fallback_seed=0)
        std = jnp.exp(0.5 * log_var)
        epsilon = jax.random.normal(key, mean.shape)
        return mean + std * epsilon

    def kl_divergence(
        self,
        mean: Float[Array, "... latent_dim"],
        log_var: Float[Array, "... latent_dim"],
    ) -> Float[Array, ""]:
        """Compute KL divergence from standard normal.

        KL(q(z|x) || p(z)) where p(z) = N(0, I)

        Args:
            mean: Mean of approximate posterior.
            log_var: Log variance of approximate posterior.

        Returns:
            Scalar KL divergence.
        """
        # KL = -0.5 * sum(1 + log_var - mean^2 - exp(log_var))
        kl = -0.5 * jnp.sum(1 + log_var - mean**2 - jnp.exp(log_var))
        return kl

    def elbo_loss(
        self,
        recon_loss: Float[Array, ""],
        mean: Float[Array, "... latent_dim"],
        log_var: Float[Array, "... latent_dim"],
        beta: float = 1.0,
    ) -> Float[Array, ""]:
        """Compute negative ELBO loss.

        loss = recon_loss + beta * KL_divergence

        Args:
            recon_loss: Reconstruction loss (e.g., MSE or BCE).
            mean: Mean of latent distribution.
            log_var: Log variance of latent distribution.
            beta: Weight for KL term (default 1.0, >1 for beta-VAE).

        Returns:
            Negative ELBO loss.
        """
        kl = self.kl_divergence(mean, log_var)
        return recon_loss + beta * kl

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Base apply method - should be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement apply()")


class GraphOperator(OperatorModule):
    """Base class for graph neural network operators.

    Provides utilities for graph-structured data processing:
    - Scatter-based aggregation (sum, mean, max)
    - Edge handling utilities
    - Graph pooling operations

    Used by assembly graphs, molecular graphs, and phylogenetic trees.

    Subclasses should define their config with 'node_features',
    'edge_features', and 'num_heads' fields.
    """

    def __init__(
        self,
        config: OperatorConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize GraphOperator.

        Args:
            config: Configuration with graph-related fields.
            rngs: Flax NNX random number generators.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        self.node_features = getattr(config, "node_features", 32)
        self.edge_features = getattr(config, "edge_features", 8)
        self.num_heads = getattr(config, "num_heads", 1)

    def scatter_aggregate(
        self,
        messages: Float[Array, "num_messages features"],
        indices: Int[Array, "num_messages"],
        num_nodes: int,
        aggregation: Literal["sum", "mean", "max"] = "sum",
    ) -> Float[Array, "num_nodes features"]:
        """Aggregate messages at nodes using scatter operations.

        Args:
            messages: Message features to aggregate.
            indices: Target node indices for each message.
            num_nodes: Total number of nodes.
            aggregation: Aggregation method.

        Returns:
            Aggregated features for each node.
        """
        if aggregation == "sum":
            return jax.ops.segment_sum(messages, indices, num_segments=num_nodes)
        elif aggregation == "mean":
            sum_messages = jax.ops.segment_sum(messages, indices, num_segments=num_nodes)
            counts = jax.ops.segment_sum(
                jnp.ones(messages.shape[0]), indices, num_segments=num_nodes
            )
            return sum_messages / (counts[:, None] + EPSILON)
        elif aggregation == "max":
            result = jax.ops.segment_max(messages, indices, num_segments=num_nodes)
            # Replace -inf with 0
            return jnp.where(jnp.isinf(result), jnp.zeros_like(result), result)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

    def global_pool(
        self,
        node_features: Float[Array, "num_nodes features"],
        batch: Int[Array, "num_nodes"] | None = None,
        aggregation: Literal["sum", "mean", "max"] = "mean",
    ) -> Float[Array, "batch_size features"] | Float[Array, "features"]:
        """Pool node features to graph-level representation.

        Args:
            node_features: Node feature matrix.
            batch: Batch assignment for each node (None for single graph).
            aggregation: Pooling method.

        Returns:
            Graph-level features.
        """
        if batch is None:
            # Single graph
            if aggregation == "sum":
                return jnp.sum(node_features, axis=0)
            elif aggregation == "mean":
                return jnp.mean(node_features, axis=0)
            elif aggregation == "max":
                return jnp.max(node_features, axis=0)
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
        else:
            # Batched graphs
            num_graphs = int(jnp.max(batch)) + 1
            return self.scatter_aggregate(node_features, batch, num_graphs, aggregation)

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Base apply method - should be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement apply()")


class HMMOperator(OperatorModule):
    """Base class for Hidden Markov Model operators.

    Provides the core HMM algorithms:
    - Forward algorithm for likelihood computation
    - Forward-backward for posterior computation
    - Viterbi for MAP decoding (soft version)

    Used by variant calling, gene finding, chromatin state annotation.

    Subclasses should define their config with 'num_states',
    'num_emissions', and 'temperature' fields.
    """

    def __init__(
        self,
        config: OperatorConfig,
        *,
        rngs: nnx.Rngs | None = None,
        name: str | None = None,
    ):
        """Initialize HMMOperator.

        Args:
            config: Configuration with HMM-related fields.
            rngs: Flax NNX random number generators.
            name: Optional operator name.
        """
        super().__init__(config, rngs=rngs, name=name)

        self.num_states = getattr(config, "num_states", 3)
        self.num_emissions = getattr(config, "num_emissions", 4)
        self.temperature = getattr(config, "temperature", DEFAULT_TEMPERATURE)

        # Initialize HMM parameters
        rngs = ensure_rngs(rngs)

        # Transition logits (will be normalized via log_softmax)
        key = get_rng_key(rngs, "params", fallback_seed=0)
        init_trans = jax.random.normal(key, (self.num_states, self.num_states)) * 0.1
        self.log_transition_params = nnx.Param(init_trans)

        # Emission logits
        key = get_rng_key(rngs, "params", fallback_seed=1)
        init_emit = jax.random.normal(key, (self.num_states, self.num_emissions)) * 0.1
        self.log_emission_params = nnx.Param(init_emit)

        # Initial state logits
        key = get_rng_key(rngs, "params", fallback_seed=2)
        init_initial = jax.random.normal(key, (self.num_states,)) * 0.1
        self.log_initial_params = nnx.Param(init_initial)

    def get_log_transition_matrix(self) -> Float[Array, "num_states num_states"]:
        """Get normalized log transition matrix."""
        return jax.nn.log_softmax(self.log_transition_params[...] / self.temperature, axis=1)

    def get_log_emission_matrix(self) -> Float[Array, "num_states num_emissions"]:
        """Get normalized log emission matrix."""
        return jax.nn.log_softmax(self.log_emission_params[...] / self.temperature, axis=1)

    def get_log_initial_distribution(self) -> Float[Array, "num_states"]:
        """Get normalized log initial state distribution."""
        return jax.nn.log_softmax(self.log_initial_params[...] / self.temperature)

    def forward_pass(
        self,
        observations: Int[Array, "seq_len"],
    ) -> Float[Array, ""]:
        """Compute log probability using forward algorithm.

        Args:
            observations: Integer-encoded observations.

        Returns:
            Log probability of the observation sequence.
        """
        log_trans = self.get_log_transition_matrix()
        log_emit = self.get_log_emission_matrix()
        log_init = self.get_log_initial_distribution()

        # Initialize
        log_alpha = log_init + log_emit[:, observations[0]]

        # Forward pass
        def forward_step(log_alpha, obs):
            log_alpha_expanded = log_alpha[:, None]
            log_alpha_new = jax.scipy.special.logsumexp(log_alpha_expanded + log_trans, axis=0)
            log_alpha_new = log_alpha_new + log_emit[:, obs]
            return log_alpha_new, None

        log_alpha, _ = jax.lax.scan(forward_step, log_alpha, observations[1:])

        return jax.scipy.special.logsumexp(log_alpha)

    def forward_backward_posteriors(
        self,
        observations: Int[Array, "seq_len"],
    ) -> Float[Array, "seq_len num_states"]:
        """Compute state posteriors using forward-backward.

        Args:
            observations: Integer-encoded observations.

        Returns:
            State posteriors P(state | observations) at each position.
        """
        log_trans = self.get_log_transition_matrix()
        log_emit = self.get_log_emission_matrix()
        log_init = self.get_log_initial_distribution()

        # Forward pass - store all alpha values
        def forward_step(log_alpha, obs):
            log_alpha_expanded = log_alpha[:, None]
            log_alpha_new = jax.scipy.special.logsumexp(log_alpha_expanded + log_trans, axis=0)
            log_alpha_new = log_alpha_new + log_emit[:, obs]
            return log_alpha_new, log_alpha_new

        log_alpha_init = log_init + log_emit[:, observations[0]]
        _, log_alphas = jax.lax.scan(forward_step, log_alpha_init, observations[1:])
        log_alphas = jnp.concatenate([log_alpha_init[None, :], log_alphas], axis=0)

        # Backward pass
        def backward_step(log_beta, obs):
            log_beta_expanded = log_beta + log_emit[:, obs]
            log_beta_new = jax.scipy.special.logsumexp(log_trans + log_beta_expanded, axis=1)
            return log_beta_new, log_beta_new

        log_beta_init = jnp.zeros(self.num_states)
        _, log_betas_rev = jax.lax.scan(backward_step, log_beta_init, observations[1:][::-1])
        log_betas = jnp.concatenate([log_betas_rev[::-1], log_beta_init[None, :]], axis=0)

        # Compute posteriors
        log_posteriors = log_alphas + log_betas
        log_posteriors = log_posteriors - jax.scipy.special.logsumexp(
            log_posteriors, axis=1, keepdims=True
        )

        return jnp.exp(log_posteriors)

    def apply(
        self,
        data: PyTree,
        state: PyTree,
        metadata: dict[str, Any] | None,
        random_params: Any = None,
        stats: dict[str, Any] | None = None,
    ) -> tuple[PyTree, PyTree, dict[str, Any] | None]:
        """Base apply method - should be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement apply()")
