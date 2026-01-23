"""Tests for differentiable operations primitives.

Following TDD: These tests define the expected behavior for soft operations
that replace hard discrete operations with differentiable approximations.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx


class TestSoftArgmax:
    """Tests for soft_argmax operation."""

    def test_soft_argmax_basic(self):
        """Test soft_argmax returns expected shape and range."""
        from diffbio.core.differentiable_ops import soft_argmax

        logits = jnp.array([1.0, 5.0, 2.0, 0.5])
        result = soft_argmax(logits)

        # Should return a scalar position
        assert result.shape == ()
        # Should be close to index 1 (where max is)
        assert 0.0 <= float(result) < len(logits)

    def test_soft_argmax_temperature(self):
        """Test temperature controls sharpness."""
        from diffbio.core.differentiable_ops import soft_argmax

        logits = jnp.array([1.0, 5.0, 2.0, 0.5])

        # Low temperature -> closer to hard argmax (index 1)
        result_low = soft_argmax(logits, temperature=0.1)
        # High temperature -> more averaged
        result_high = soft_argmax(logits, temperature=10.0)

        # Low temp should be closer to index 1
        assert jnp.abs(result_low - 1.0) < jnp.abs(result_high - 1.0)

    def test_soft_argmax_differentiable(self):
        """Test gradients flow through soft_argmax."""
        from diffbio.core.differentiable_ops import soft_argmax

        def loss_fn(logits):
            return soft_argmax(logits)

        logits = jnp.array([1.0, 5.0, 2.0, 0.5])
        grads = jax.grad(loss_fn)(logits)

        assert grads is not None
        assert grads.shape == logits.shape
        assert jnp.all(jnp.isfinite(grads))

    def test_soft_argmax_2d(self):
        """Test soft_argmax works with 2D input along axis."""
        from diffbio.core.differentiable_ops import soft_argmax

        logits = jnp.array([[1.0, 5.0, 2.0], [3.0, 1.0, 4.0]])
        result = soft_argmax(logits, axis=-1)

        assert result.shape == (2,)
        # First row max at index 1, second at index 2
        assert result[0] < result[1]  # Position in first row < second


class TestSoftSort:
    """Tests for differentiable sorting."""

    def test_soft_sort_basic(self):
        """Test soft_sort returns sorted values."""
        from diffbio.core.differentiable_ops import soft_sort

        values = jnp.array([3.0, 1.0, 4.0, 1.0, 5.0])
        sorted_vals = soft_sort(values)

        assert sorted_vals.shape == values.shape
        # Should be approximately sorted
        for i in range(len(sorted_vals) - 1):
            assert sorted_vals[i] <= sorted_vals[i + 1] + 0.1  # Allow small tolerance

    def test_soft_sort_differentiable(self):
        """Test gradients flow through soft_sort."""
        from diffbio.core.differentiable_ops import soft_sort

        def loss_fn(values):
            return jnp.sum(soft_sort(values))

        values = jnp.array([3.0, 1.0, 4.0, 1.0, 5.0])
        grads = jax.grad(loss_fn)(values)

        assert grads is not None
        assert grads.shape == values.shape
        assert jnp.all(jnp.isfinite(grads))

    def test_soft_sort_temperature(self):
        """Test temperature affects sorting sharpness."""
        from diffbio.core.differentiable_ops import soft_sort

        values = jnp.array([3.0, 1.0, 4.0])

        # Low temperature -> closer to hard sort
        sorted_low = soft_sort(values, temperature=0.01)
        # High temperature -> more smoothed
        sorted_high = soft_sort(values, temperature=10.0)

        # Low temp should have larger differences between adjacent values
        diff_low = jnp.max(jnp.diff(sorted_low))
        diff_high = jnp.max(jnp.diff(sorted_high))
        assert diff_low > diff_high


class TestLogsumexpSmoothMax:
    """Tests for logsumexp-based smooth max."""

    def test_smooth_max_basic(self):
        """Test smooth max returns value close to max."""
        from diffbio.core.differentiable_ops import logsumexp_smooth_max

        values = jnp.array([1.0, 5.0, 2.0, 3.0])
        result = logsumexp_smooth_max(values)

        # Should be close to but slightly above max (5.0)
        assert float(result) >= 5.0
        assert float(result) < 6.0  # Not too far above

    def test_smooth_max_temperature(self):
        """Test temperature controls max sharpness."""
        from diffbio.core.differentiable_ops import logsumexp_smooth_max

        values = jnp.array([1.0, 5.0, 2.0, 3.0])

        # Low temperature -> closer to hard max
        result_low = logsumexp_smooth_max(values, temperature=0.1)
        # High temperature -> more average-like
        result_high = logsumexp_smooth_max(values, temperature=10.0)

        # Low temp should be closer to max (5.0)
        assert jnp.abs(result_low - 5.0) < jnp.abs(result_high - 5.0)

    def test_smooth_max_differentiable(self):
        """Test gradients flow through smooth max."""
        from diffbio.core.differentiable_ops import logsumexp_smooth_max

        def loss_fn(values):
            return logsumexp_smooth_max(values)

        values = jnp.array([1.0, 5.0, 2.0, 3.0])
        grads = jax.grad(loss_fn)(values)

        assert grads is not None
        assert grads.shape == values.shape
        # Largest input should have largest gradient
        assert jnp.argmax(grads) == jnp.argmax(values)

    def test_smooth_max_axis(self):
        """Test smooth max works along specified axis."""
        from diffbio.core.differentiable_ops import logsumexp_smooth_max

        values = jnp.array([[1.0, 5.0], [3.0, 2.0]])
        result = logsumexp_smooth_max(values, axis=-1)

        assert result.shape == (2,)
        # First row max is 5, second row max is 3
        assert result[0] > result[1]


class TestSegmentSoftmax:
    """Tests for segment-wise softmax."""

    def test_segment_softmax_basic(self):
        """Test segment_softmax normalizes within segments."""
        from diffbio.core.differentiable_ops import segment_softmax

        logits = jnp.array([1.0, 2.0, 3.0, 1.0, 2.0])
        segment_ids = jnp.array([0, 0, 0, 1, 1])
        result = segment_softmax(logits, segment_ids)

        assert result.shape == logits.shape

        # Check segment 0 sums to 1
        segment_0_sum = jnp.sum(result[:3])
        assert jnp.abs(segment_0_sum - 1.0) < 1e-5

        # Check segment 1 sums to 1
        segment_1_sum = jnp.sum(result[3:])
        assert jnp.abs(segment_1_sum - 1.0) < 1e-5

    def test_segment_softmax_differentiable(self):
        """Test gradients flow through segment_softmax."""
        from diffbio.core.differentiable_ops import segment_softmax

        def loss_fn(logits):
            segment_ids = jnp.array([0, 0, 0, 1, 1])
            return jnp.sum(segment_softmax(logits, segment_ids))

        logits = jnp.array([1.0, 2.0, 3.0, 1.0, 2.0])
        grads = jax.grad(loss_fn)(logits)

        assert grads is not None
        assert grads.shape == logits.shape


class TestGumbelSoftmax:
    """Tests for Gumbel-softmax sampling."""

    def test_gumbel_softmax_shape(self):
        """Test gumbel_softmax returns correct shape."""
        from diffbio.core.differentiable_ops import gumbel_softmax

        logits = jnp.array([1.0, 2.0, 3.0, 0.5])
        key = jax.random.PRNGKey(42)
        result = gumbel_softmax(logits, key)

        assert result.shape == logits.shape
        # Should sum to 1 (probabilities)
        assert jnp.abs(jnp.sum(result) - 1.0) < 1e-5

    def test_gumbel_softmax_hard(self):
        """Test hard gumbel_softmax is one-hot."""
        from diffbio.core.differentiable_ops import gumbel_softmax

        logits = jnp.array([1.0, 5.0, 2.0, 0.5])
        key = jax.random.PRNGKey(42)
        result = gumbel_softmax(logits, key, hard=True)

        # Should be approximately one-hot
        assert jnp.abs(jnp.sum(result) - 1.0) < 1e-5
        assert jnp.max(result) > 0.9  # One value dominates

    def test_gumbel_softmax_differentiable(self):
        """Test gradients flow through gumbel_softmax."""
        from diffbio.core.differentiable_ops import gumbel_softmax

        def loss_fn(logits):
            key = jax.random.PRNGKey(42)
            return jnp.sum(gumbel_softmax(logits, key))

        logits = jnp.array([1.0, 2.0, 3.0, 0.5])
        grads = jax.grad(loss_fn)(logits)

        assert grads is not None
        assert grads.shape == logits.shape

    def test_gumbel_softmax_temperature(self):
        """Test temperature affects sampling sharpness."""
        from diffbio.core.differentiable_ops import gumbel_softmax

        logits = jnp.array([1.0, 5.0, 2.0])
        key = jax.random.PRNGKey(42)

        # Low temperature -> more peaked
        result_low = gumbel_softmax(logits, key, temperature=0.1)
        # High temperature -> more uniform
        result_high = gumbel_softmax(logits, key, temperature=10.0)

        # Low temp should have higher max
        assert jnp.max(result_low) > jnp.max(result_high)


class TestDifferentiableScan:
    """Tests for differentiable scan operation."""

    def test_differentiable_scan_basic(self):
        """Test differentiable_scan for simple accumulation."""
        from diffbio.core.differentiable_ops import differentiable_scan

        def step_fn(carry, x):
            return carry + x, carry

        init = jnp.array(0.0)
        xs = jnp.array([1.0, 2.0, 3.0, 4.0])
        final, outputs = differentiable_scan(step_fn, init, xs)

        # Final should be sum
        assert jnp.abs(final - 10.0) < 1e-5
        # Outputs should be cumulative sums minus last element
        expected = jnp.array([0.0, 1.0, 3.0, 6.0])
        assert jnp.allclose(outputs, expected)

    def test_differentiable_scan_gradients(self):
        """Test gradients flow through differentiable_scan."""
        from diffbio.core.differentiable_ops import differentiable_scan

        def loss_fn(xs):
            def step_fn(carry, x):
                return carry + x, carry

            init = jnp.array(0.0)
            final, _ = differentiable_scan(step_fn, init, xs)
            return final

        xs = jnp.array([1.0, 2.0, 3.0, 4.0])
        grads = jax.grad(loss_fn)(xs)

        assert grads is not None
        assert grads.shape == xs.shape
        # All inputs contribute equally to sum, so gradients should be uniform
        assert jnp.allclose(grads, jnp.ones_like(xs))


class TestSoftThreshold:
    """Tests for soft_threshold (re-export from nn_utils)."""

    def test_soft_threshold_basic(self):
        """Test soft_threshold returns values in [0, 1]."""
        from diffbio.core.differentiable_ops import soft_threshold

        values = jnp.array([10.0, 20.0, 30.0, 40.0])
        threshold = 25.0
        result = soft_threshold(values, threshold)

        assert result.shape == values.shape
        assert jnp.all(result >= 0.0)
        assert jnp.all(result <= 1.0)

    def test_soft_threshold_behavior(self):
        """Test soft_threshold transitions around threshold."""
        from diffbio.core.differentiable_ops import soft_threshold

        values = jnp.array([10.0, 25.0, 40.0])
        threshold = 25.0
        result = soft_threshold(values, threshold, temperature=1.0)

        # Below threshold should be < 0.5
        assert float(result[0]) < 0.5
        # At threshold should be ~0.5
        assert jnp.abs(result[1] - 0.5) < 0.1
        # Above threshold should be > 0.5
        assert float(result[2]) > 0.5


class TestSoftOneHot:
    """Tests for soft_one_hot operation."""

    def test_soft_one_hot_basic(self):
        """Test soft_one_hot returns distribution over classes."""
        from diffbio.core.differentiable_ops import soft_one_hot

        index = jnp.array(1.0)  # Class 1
        num_classes = 4
        result = soft_one_hot(index, num_classes, temperature=0.1)

        assert result.shape == (num_classes,)
        # Should sum to 1 (valid distribution)
        assert jnp.abs(jnp.sum(result) - 1.0) < 1e-5
        # Index 1 should have highest probability
        assert jnp.argmax(result) == 1

    def test_soft_one_hot_between_classes(self):
        """Test soft_one_hot with index between two classes."""
        from diffbio.core.differentiable_ops import soft_one_hot

        index = jnp.array(1.5)  # Between class 1 and 2
        num_classes = 4
        result = soft_one_hot(index, num_classes, temperature=0.5)

        # Classes 1 and 2 should have highest probability
        assert result[1] > result[0]
        assert result[2] > result[3]
        # Classes 1 and 2 should be similar
        assert jnp.abs(result[1] - result[2]) < 0.5

    def test_soft_one_hot_differentiable(self):
        """Test gradients flow through soft_one_hot."""
        from diffbio.core.differentiable_ops import soft_one_hot

        def loss_fn(index):
            return jnp.sum(soft_one_hot(index, num_classes=4, temperature=1.0))

        index = jnp.array(1.5)
        grads = jax.grad(loss_fn)(index)
        assert jnp.isfinite(grads)


class TestSoftAttentionWeights:
    """Tests for soft_attention_weights operation."""

    def test_soft_attention_weights_basic(self):
        """Test soft_attention_weights returns valid attention."""
        from diffbio.core.differentiable_ops import soft_attention_weights

        query = jnp.array([1.0, 0.0, 0.0, 0.0])  # 4-dim query
        keys = jnp.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        )  # 3 keys
        result = soft_attention_weights(query, keys, temperature=1.0)

        assert result.shape == (3,)
        # Should sum to 1 (valid attention weights)
        assert jnp.abs(jnp.sum(result) - 1.0) < 1e-5
        # First key should have highest attention (most similar)
        assert jnp.argmax(result) == 0

    def test_soft_attention_weights_differentiable(self):
        """Test gradients flow through attention weights."""
        from diffbio.core.differentiable_ops import soft_attention_weights

        def loss_fn(query):
            keys = jnp.eye(4)[:3]
            return jnp.sum(soft_attention_weights(query, keys, temperature=1.0))

        query = jnp.array([1.0, 0.0, 0.0, 0.0])
        grads = jax.grad(loss_fn)(query)
        assert grads.shape == query.shape
        assert jnp.all(jnp.isfinite(grads))


class TestDifferentiableTopK:
    """Tests for differentiable_topk operation."""

    def test_differentiable_topk_basic(self):
        """Test differentiable_topk returns soft selection weights."""
        from diffbio.core.differentiable_ops import differentiable_topk

        values = jnp.array([1.0, 5.0, 2.0, 4.0, 3.0])
        k = 2
        result = differentiable_topk(values, k, temperature=0.1)

        assert result.shape == values.shape
        # All values should be in [0, 1]
        assert jnp.all(result >= 0.0)
        assert jnp.all(result <= 1.0)
        # Top 2 values (5.0, 4.0 at indices 1, 3) should have highest weights
        top_indices = jnp.argsort(values)[-k:]
        for idx in top_indices:
            assert result[idx] > 0.5

    def test_differentiable_topk_differentiable(self):
        """Test gradients flow through differentiable_topk."""
        from diffbio.core.differentiable_ops import differentiable_topk

        def loss_fn(values):
            return jnp.sum(differentiable_topk(values, k=2, temperature=1.0))

        values = jnp.array([1.0, 5.0, 2.0, 4.0, 3.0])
        grads = jax.grad(loss_fn)(values)
        assert grads.shape == values.shape
        assert jnp.all(jnp.isfinite(grads))


class TestEdgeCases:
    """Test edge cases for differentiable operations."""

    def test_empty_arrays(self):
        """Test operations handle empty arrays gracefully."""
        from diffbio.core.differentiable_ops import logsumexp_smooth_max

        # Empty array
        empty = jnp.array([])
        result = logsumexp_smooth_max(empty)
        # Should return -inf for empty
        assert jnp.isinf(result)

    def test_single_element(self):
        """Test operations work with single element."""
        from diffbio.core.differentiable_ops import (
            logsumexp_smooth_max,
            soft_argmax,
            soft_sort,
        )

        single = jnp.array([5.0])

        # Smooth max of single element is that element
        max_result = logsumexp_smooth_max(single)
        assert jnp.abs(max_result - 5.0) < 0.1

        # Soft argmax of single element is 0
        argmax_result = soft_argmax(single)
        assert jnp.abs(argmax_result - 0.0) < 0.1

        # Soft sort of single element is that element
        sort_result = soft_sort(single)
        assert jnp.abs(sort_result[0] - 5.0) < 0.1

    def test_numerical_stability(self):
        """Test operations are numerically stable with extreme values."""
        from diffbio.core.differentiable_ops import gumbel_softmax, logsumexp_smooth_max

        # Large values
        large = jnp.array([1000.0, 1001.0, 1002.0])
        result = logsumexp_smooth_max(large)
        assert jnp.isfinite(result)

        # Very different scales
        varied = jnp.array([-100.0, 0.0, 100.0])
        key = jax.random.PRNGKey(42)
        result = gumbel_softmax(varied, key)
        assert jnp.all(jnp.isfinite(result))

    def test_jit_compatibility(self):
        """Test all operations work with JIT."""
        from diffbio.core.differentiable_ops import (
            gumbel_softmax,
            logsumexp_smooth_max,
            soft_argmax,
            soft_sort,
        )

        # JIT compile each function
        smooth_max_jit = jax.jit(logsumexp_smooth_max)
        soft_argmax_jit = jax.jit(soft_argmax)
        soft_sort_jit = jax.jit(soft_sort)
        gumbel_jit = jax.jit(lambda x, k: gumbel_softmax(x, k))

        values = jnp.array([1.0, 5.0, 2.0])
        key = jax.random.PRNGKey(42)

        # All should work without error
        smooth_max_jit(values)
        soft_argmax_jit(values)
        soft_sort_jit(values)
        gumbel_jit(values, key)


# =============================================================================
# Mathematical Verification Tests
# =============================================================================


class TestMathematicalVerification:
    """Tests verifying mathematical correctness of implementations."""

    def test_logsumexp_formula(self):
        """Verify logsumexp_smooth_max matches the mathematical formula."""
        from diffbio.core.differentiable_ops import logsumexp_smooth_max

        values = jnp.array([1.0, 2.0, 3.0])
        temperature = 0.5

        # Manual computation: T * log(sum(exp(x/T)))
        expected = temperature * jnp.log(jnp.sum(jnp.exp(values / temperature)))
        result = logsumexp_smooth_max(values, temperature=temperature)

        assert jnp.allclose(result, expected, atol=1e-5)

    def test_logsumexp_limit_approaches_max(self):
        """Verify logsumexp approaches hard max as temperature -> 0."""
        from diffbio.core.differentiable_ops import logsumexp_smooth_max

        values = jnp.array([1.0, 5.0, 3.0, 2.0])
        hard_max = jnp.max(values)

        # Very low temperature should approach hard max
        result = logsumexp_smooth_max(values, temperature=0.001)
        assert jnp.abs(result - hard_max) < 0.01

    def test_soft_argmax_weighted_average_formula(self):
        """Verify soft_argmax is weighted average of positions."""
        from diffbio.core.differentiable_ops import soft_argmax

        logits = jnp.array([1.0, 2.0, 3.0, 4.0])
        temperature = 1.0

        # Manual computation: sum(softmax(x/T) * positions)
        weights = jax.nn.softmax(logits / temperature)
        positions = jnp.arange(len(logits), dtype=logits.dtype)
        expected = jnp.sum(weights * positions)

        result = soft_argmax(logits, temperature=temperature)
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_gumbel_softmax_distribution(self):
        """Verify Gumbel-softmax produces expected distribution over many samples."""
        from diffbio.core.differentiable_ops import gumbel_softmax

        logits = jnp.array([0.0, 1.0, 2.0])  # Probabilities should be ~[0.09, 0.24, 0.67]
        expected_probs = jax.nn.softmax(logits)

        # Generate many samples and estimate distribution
        num_samples = 1000
        keys = jax.random.split(jax.random.PRNGKey(42), num_samples)
        samples = jax.vmap(lambda k: gumbel_softmax(logits, k, temperature=0.1))(keys)
        estimated_probs = jnp.mean(samples, axis=0)

        # Should approximately match softmax distribution
        assert jnp.allclose(estimated_probs, expected_probs, atol=0.1)

    def test_soft_sort_preserves_sum(self):
        """Verify soft_sort preserves the sum of values (permutation property)."""
        from diffbio.core.differentiable_ops import soft_sort

        values = jnp.array([3.0, 1.0, 4.0, 1.0, 5.0])
        sorted_vals = soft_sort(values, temperature=0.01)

        # Sum should be preserved (approximately, due to soft nature)
        assert jnp.abs(jnp.sum(sorted_vals) - jnp.sum(values)) < 0.5

    def test_soft_sort_ordering(self):
        """Verify soft_sort produces correctly ordered output at low temperature."""
        from diffbio.core.differentiable_ops import soft_sort

        values = jnp.array([5.0, 2.0, 8.0, 1.0, 6.0])
        sorted_vals = soft_sort(values, temperature=0.001)
        hard_sorted = jnp.sort(values)

        # Should match hard sort closely at very low temperature
        assert jnp.allclose(sorted_vals, hard_sorted, atol=0.1)

    def test_segment_softmax_independence(self):
        """Verify segment_softmax normalizes segments independently."""
        from diffbio.core.differentiable_ops import segment_softmax

        logits = jnp.array([100.0, 0.0, 0.0, 1.0, 2.0])
        segment_ids = jnp.array([0, 0, 0, 1, 1])
        result = segment_softmax(logits, segment_ids, temperature=1.0)

        # Segment 0 should be dominated by first element
        assert result[0] > 0.9
        # Segments should sum to 1 independently
        assert jnp.allclose(jnp.sum(result[:3]), 1.0, atol=1e-5)
        assert jnp.allclose(jnp.sum(result[3:]), 1.0, atol=1e-5)

    def test_kl_divergence_zero_for_standard_normal(self):
        """Verify KL divergence is zero when q(z) = p(z) = N(0,1)."""
        from diffbio.core.base_operators import EncoderDecoderOperator
        from dataclasses import dataclass
        from datarax.core.config import OperatorConfig
        from flax import nnx

        @dataclass
        class Config(OperatorConfig):
            latent_dim: int = 10

        rngs = nnx.Rngs(42)
        op = EncoderDecoderOperator(Config(), rngs=rngs)

        # Standard normal: mean=0, log_var=0 (var=1)
        mean = jnp.zeros((5, 10))
        log_var = jnp.zeros((5, 10))

        kl = op.kl_divergence(mean, log_var)
        assert jnp.abs(kl) < 1e-5  # Should be ~0


# =============================================================================
# Flax NNX Compatibility Tests
# =============================================================================


class TestFlaxNNXCompatibility:
    """Tests verifying Flax NNX compatibility."""

    def test_nnx_jit_with_module(self, rngs):
        """Test operations work inside nnx.jit decorated functions."""
        from diffbio.core.neural_components import GumbelSoftmaxModule

        module = GumbelSoftmaxModule(temperature=1.0, rngs=rngs)

        @nnx.jit
        def forward(m, logits):
            return m(logits)

        logits = jnp.array([[1.0, 2.0, 3.0]])
        result = forward(module, logits)
        assert result.shape == logits.shape
        assert jnp.allclose(jnp.sum(result, axis=-1), 1.0, atol=1e-5)

    def test_nnx_grad_with_module(self, rngs):
        """Test nnx.grad works with module parameters."""
        from diffbio.core.neural_components import GraphMessagePassing

        layer = GraphMessagePassing(node_features=4, edge_features=2, hidden_dim=8, rngs=rngs)

        def loss_fn(model, node_feat, edge_feat, edge_index):
            output = model(node_feat, edge_feat, edge_index)
            return jnp.sum(output)

        node_feat = jnp.ones((3, 4))
        edge_feat = jnp.ones((3, 2))
        edge_index = jnp.array([[0, 1, 2], [1, 2, 0]])

        grads = nnx.grad(loss_fn)(layer, node_feat, edge_feat, edge_index)
        assert grads is not None

    def test_nnx_value_and_grad(self, rngs):
        """Test nnx.value_and_grad for training loops."""
        from diffbio.core.neural_components import GumbelSoftmaxModule

        module = GumbelSoftmaxModule(temperature=1.0, rngs=rngs)

        def loss_fn(m, logits):
            output = m(logits)
            return jnp.mean(output**2)

        logits = jnp.array([[1.0, 2.0, 3.0]])
        loss, grads = nnx.value_and_grad(loss_fn)(module, logits)

        assert isinstance(loss, jax.Array)
        assert jnp.isfinite(loss)
        assert grads is not None

    def test_module_state_separation(self, rngs):
        """Test nnx.split and nnx.merge for state management."""
        from diffbio.core.neural_components import GraphMessagePassing

        layer = GraphMessagePassing(node_features=4, edge_features=2, hidden_dim=8, rngs=rngs)

        # Split into graph def and state
        graphdef, state = nnx.split(layer)

        # Merge back
        restored = nnx.merge(graphdef, state)

        # Should produce same output
        node_feat = jnp.ones((3, 4))
        edge_feat = jnp.ones((3, 2))
        edge_index = jnp.array([[0, 1, 2], [1, 2, 0]])

        output1 = layer(node_feat, edge_feat, edge_index)
        output2 = restored(node_feat, edge_feat, edge_index)

        assert jnp.allclose(output1, output2)

    def test_vmap_with_operations(self):
        """Test operations work with jax.vmap."""
        from diffbio.core.differentiable_ops import logsumexp_smooth_max, soft_argmax

        # Batch of inputs
        batch_logits = jnp.array([[1.0, 5.0, 2.0], [3.0, 1.0, 4.0], [2.0, 2.0, 2.0]])

        # vmap over batch
        batch_smooth_max = jax.vmap(logsumexp_smooth_max)(batch_logits)
        batch_soft_argmax = jax.vmap(soft_argmax)(batch_logits)

        assert batch_smooth_max.shape == (3,)
        assert batch_soft_argmax.shape == (3,)

    def test_module_eval_mode(self, rngs):
        """Test modules work correctly in eval mode (no randomness needed)."""
        from diffbio.core.neural_components import GraphMessagePassing

        layer = GraphMessagePassing(node_features=4, edge_features=2, hidden_dim=8, rngs=rngs)

        # Deterministic forward pass
        node_feat = jnp.ones((3, 4))
        edge_feat = jnp.ones((3, 2))
        edge_index = jnp.array([[0, 1, 2], [1, 2, 0]])

        output1 = layer(node_feat, edge_feat, edge_index)
        output2 = layer(node_feat, edge_feat, edge_index)

        # GraphMessagePassing should be deterministic
        assert jnp.allclose(output1, output2)


# =============================================================================
# Scalability Tests
# =============================================================================


class TestScalability:
    """Tests verifying scalability with larger inputs."""

    @pytest.mark.parametrize("size", [100, 500, 1000])
    def test_soft_sort_scales(self, size):
        """Test soft_sort handles various input sizes."""
        from diffbio.core.differentiable_ops import soft_sort

        values = jax.random.normal(jax.random.PRNGKey(42), (size,))
        sorted_vals = soft_sort(values, temperature=0.1)

        assert sorted_vals.shape == (size,)
        assert jnp.all(jnp.isfinite(sorted_vals))

    @pytest.mark.parametrize("size", [100, 500, 1000])
    def test_segment_softmax_scales(self, size):
        """Test segment_softmax handles various input sizes."""
        from diffbio.core.differentiable_ops import segment_softmax

        logits = jax.random.normal(jax.random.PRNGKey(42), (size,))
        # Create random segments
        num_segments = 10
        segment_ids = jax.random.randint(jax.random.PRNGKey(43), (size,), 0, num_segments)
        result = segment_softmax(logits, segment_ids)

        assert result.shape == (size,)
        assert jnp.all(jnp.isfinite(result))
        assert jnp.all(result >= 0.0)
        assert jnp.all(result <= 1.0)

    @pytest.mark.parametrize("batch_size", [10, 50, 100])
    def test_gumbel_softmax_batched(self, batch_size):
        """Test gumbel_softmax handles batched inputs efficiently."""
        from diffbio.core.differentiable_ops import gumbel_softmax

        num_classes = 20
        logits = jax.random.normal(jax.random.PRNGKey(42), (batch_size, num_classes))
        keys = jax.random.split(jax.random.PRNGKey(43), batch_size)

        # Batched application
        result = jax.vmap(lambda l, k: gumbel_softmax(l, k, temperature=1.0))(logits, keys)

        assert result.shape == (batch_size, num_classes)
        # Each row should sum to 1
        row_sums = jnp.sum(result, axis=-1)
        assert jnp.allclose(row_sums, 1.0, atol=1e-5)

    def test_graph_message_passing_scales(self, rngs):
        """Test GraphMessagePassing handles larger graphs."""
        from diffbio.core.neural_components import GraphMessagePassing

        num_nodes = 100
        num_edges = 500
        node_feat_dim = 32
        edge_feat_dim = 8
        hidden_dim = 64

        layer = GraphMessagePassing(
            node_features=node_feat_dim,
            edge_features=edge_feat_dim,
            hidden_dim=hidden_dim,
            rngs=rngs,
        )

        node_feat = jax.random.normal(jax.random.PRNGKey(42), (num_nodes, node_feat_dim))
        edge_feat = jax.random.normal(jax.random.PRNGKey(43), (num_edges, edge_feat_dim))
        # Random edges
        edge_index = jax.random.randint(jax.random.PRNGKey(44), (2, num_edges), 0, num_nodes)

        output = layer(node_feat, edge_feat, edge_index)

        assert output.shape == (num_nodes, hidden_dim)
        assert jnp.all(jnp.isfinite(output))

    @pytest.mark.parametrize("seq_len", [50, 100, 200])
    def test_differentiable_scan_scales(self, seq_len):
        """Test differentiable_scan handles various sequence lengths."""
        from diffbio.core.differentiable_ops import differentiable_scan

        def step_fn(carry, x):
            # Simple RNN-like step
            new_carry = 0.9 * carry + 0.1 * x
            return new_carry, new_carry

        init = jnp.zeros(10)
        xs = jax.random.normal(jax.random.PRNGKey(42), (seq_len, 10))

        final, outputs = differentiable_scan(step_fn, init, xs)

        assert final.shape == (10,)
        assert outputs.shape == (seq_len, 10)
        assert jnp.all(jnp.isfinite(outputs))

    def test_memory_efficiency_with_jit(self):
        """Test JIT compilation provides memory efficiency."""
        from diffbio.core.differentiable_ops import logsumexp_smooth_max

        @jax.jit
        def compute_many_smooth_max(values):
            # Apply smooth max multiple times
            result = logsumexp_smooth_max(values)
            for _ in range(10):
                result = result + logsumexp_smooth_max(values)
            return result

        values = jax.random.normal(jax.random.PRNGKey(42), (1000,))
        result = compute_many_smooth_max(values)

        assert jnp.isfinite(result)

    def test_hmm_operator_scales(self, rngs):
        """Test HMMOperator handles longer sequences."""
        from diffbio.core.base_operators import HMMOperator
        from dataclasses import dataclass
        from datarax.core.config import OperatorConfig

        @dataclass
        class Config(OperatorConfig):
            num_states: int = 5
            num_emissions: int = 10
            temperature: float = 1.0

        op = HMMOperator(Config(), rngs=rngs)

        # Longer sequence
        seq_len = 100
        observations = jax.random.randint(jax.random.PRNGKey(42), (seq_len,), 0, 10)

        log_prob = op.forward_pass(observations)
        posteriors = op.forward_backward_posteriors(observations)

        assert jnp.isfinite(log_prob)
        assert posteriors.shape == (seq_len, 5)
        # Posteriors should sum to 1 at each position
        assert jnp.allclose(jnp.sum(posteriors, axis=-1), 1.0, atol=1e-5)
