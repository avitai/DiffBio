"""Tests for graph utility functions.

Following TDD: These tests define the expected behavior for k-NN graph
construction, pairwise distance computation, fuzzy membership, and graph
symmetrization functions extracted from UMAP.
"""

import jax
import jax.numpy as jnp


class TestComputePairwiseDistances:
    """Tests for compute_pairwise_distances."""

    def test_euclidean_basic_shape(self) -> None:
        """Test Euclidean distances return correct shape."""
        from diffbio.core.graph_utils import compute_pairwise_distances

        features = jax.random.normal(jax.random.key(0), (20, 10))
        distances = compute_pairwise_distances(features, metric="euclidean")

        assert distances.shape == (20, 20)

    def test_euclidean_self_distance_zero(self) -> None:
        """Test diagonal (self-distance) is approximately zero for Euclidean."""
        from diffbio.core.graph_utils import compute_pairwise_distances

        features = jax.random.normal(jax.random.key(0), (10, 5))
        distances = compute_pairwise_distances(features, metric="euclidean")

        # Diagonal should be near zero (within numerical epsilon tolerance)
        diag = jnp.diag(distances)
        assert jnp.allclose(diag, 0.0, atol=1e-3)

    def test_euclidean_symmetry(self) -> None:
        """Test Euclidean distance matrix is symmetric."""
        from diffbio.core.graph_utils import compute_pairwise_distances

        features = jax.random.normal(jax.random.key(1), (15, 8))
        distances = compute_pairwise_distances(features, metric="euclidean")

        assert jnp.allclose(distances, distances.T, atol=1e-6)

    def test_euclidean_non_negative(self) -> None:
        """Test all Euclidean distances are non-negative."""
        from diffbio.core.graph_utils import compute_pairwise_distances

        features = jax.random.normal(jax.random.key(2), (12, 6))
        distances = compute_pairwise_distances(features, metric="euclidean")

        assert jnp.all(distances >= 0.0)

    def test_cosine_basic_shape(self) -> None:
        """Test cosine distances return correct shape."""
        from diffbio.core.graph_utils import compute_pairwise_distances

        features = jax.random.normal(jax.random.key(0), (20, 10))
        distances = compute_pairwise_distances(features, metric="cosine")

        assert distances.shape == (20, 20)

    def test_cosine_self_distance_zero(self) -> None:
        """Test diagonal is approximately zero for cosine distance."""
        from diffbio.core.graph_utils import compute_pairwise_distances

        features = jax.random.normal(jax.random.key(3), (10, 5))
        distances = compute_pairwise_distances(features, metric="cosine")

        diag = jnp.diag(distances)
        assert jnp.allclose(diag, 0.0, atol=1e-3)

    def test_cosine_symmetry(self) -> None:
        """Test cosine distance matrix is symmetric."""
        from diffbio.core.graph_utils import compute_pairwise_distances

        features = jax.random.normal(jax.random.key(4), (15, 8))
        distances = compute_pairwise_distances(features, metric="cosine")

        assert jnp.allclose(distances, distances.T, atol=1e-6)

    def test_cosine_non_negative(self) -> None:
        """Test all cosine distances are non-negative."""
        from diffbio.core.graph_utils import compute_pairwise_distances

        features = jax.random.normal(jax.random.key(5), (12, 6))
        distances = compute_pairwise_distances(features, metric="cosine")

        assert jnp.all(distances >= 0.0)

    def test_known_euclidean_values(self) -> None:
        """Test Euclidean distances against known values."""
        from diffbio.core.graph_utils import compute_pairwise_distances

        features = jnp.array([[0.0, 0.0], [3.0, 4.0]])
        distances = compute_pairwise_distances(features, metric="euclidean")

        # Distance from [0,0] to [3,4] should be 5.0
        assert jnp.isclose(distances[0, 1], 5.0, atol=1e-3)
        assert jnp.isclose(distances[1, 0], 5.0, atol=1e-3)

    def test_identical_points_cosine(self) -> None:
        """Test cosine distance between identical directions is near zero."""
        from diffbio.core.graph_utils import compute_pairwise_distances

        features = jnp.array([[1.0, 2.0], [2.0, 4.0]])  # Same direction
        distances = compute_pairwise_distances(features, metric="cosine")

        assert jnp.isclose(distances[0, 1], 0.0, atol=1e-3)

    def test_gradient_flow_euclidean(self) -> None:
        """Test gradients flow through Euclidean distance computation."""
        from diffbio.core.graph_utils import compute_pairwise_distances

        def loss_fn(features: jax.Array) -> jax.Array:
            distances = compute_pairwise_distances(features, metric="euclidean")
            return jnp.sum(distances)

        features = jax.random.normal(jax.random.key(6), (8, 4))
        grads = jax.grad(loss_fn)(features)

        assert grads.shape == features.shape
        assert jnp.all(jnp.isfinite(grads))

    def test_gradient_flow_cosine(self) -> None:
        """Test gradients flow through cosine distance computation."""
        from diffbio.core.graph_utils import compute_pairwise_distances

        def loss_fn(features: jax.Array) -> jax.Array:
            distances = compute_pairwise_distances(features, metric="cosine")
            return jnp.sum(distances)

        features = jax.random.normal(jax.random.key(7), (8, 4))
        grads = jax.grad(loss_fn)(features)

        assert grads.shape == features.shape
        assert jnp.all(jnp.isfinite(grads))

    def test_all_values_finite(self) -> None:
        """Test output is fully finite for random inputs."""
        from diffbio.core.graph_utils import compute_pairwise_distances

        features = jax.random.normal(jax.random.key(8), (25, 12))

        for metric in ("euclidean", "cosine"):
            distances = compute_pairwise_distances(features, metric=metric)
            assert jnp.all(jnp.isfinite(distances))

    def test_single_sample(self) -> None:
        """Test with a single sample returns 1x1 matrix."""
        from diffbio.core.graph_utils import compute_pairwise_distances

        features = jnp.array([[1.0, 2.0, 3.0]])
        distances = compute_pairwise_distances(features, metric="euclidean")

        assert distances.shape == (1, 1)
        assert jnp.isclose(distances[0, 0], 0.0, atol=1e-3)


class TestComputeKnnGraph:
    """Tests for compute_knn_graph."""

    def test_basic_shape(self) -> None:
        """Test k-NN graph returns correct number of edges."""
        from diffbio.core.graph_utils import compute_knn_graph

        n = 10
        k = 3
        distances = jax.random.uniform(jax.random.key(0), (n, n))
        # Make symmetric and zero diagonal
        distances = (distances + distances.T) / 2
        distances = distances + jnp.eye(n) * 1e10

        edge_indices, edge_weights = compute_knn_graph(distances, k=k)

        # Each node has k neighbors -> n * k edges total
        assert edge_indices.shape == (n * k, 2)
        assert edge_weights.shape == (n * k,)

    def test_correct_neighbors(self) -> None:
        """Test k-NN selects the correct nearest neighbors."""
        from diffbio.core.graph_utils import compute_knn_graph

        # Construct distances where neighbors are known
        # Point 0: closest to 1, then 2
        # Point 1: closest to 0, then 2
        # Point 2: closest to 1, then 0
        distances = jnp.array(
            [
                [1e10, 1.0, 2.0],
                [1.0, 1e10, 1.5],
                [2.0, 1.5, 1e10],
            ]
        )

        edge_indices, edge_weights = compute_knn_graph(distances, k=2)

        # For node 0: neighbors should be 1 and 2
        node0_edges = edge_indices[edge_indices[:, 0] == 0]
        node0_neighbors = set(node0_edges[:, 1].tolist())
        assert node0_neighbors == {1, 2}

    def test_weights_are_distances(self) -> None:
        """Test edge weights correspond to actual distances."""
        from diffbio.core.graph_utils import compute_knn_graph

        distances = jnp.array(
            [
                [1e10, 1.0, 3.0, 2.0],
                [1.0, 1e10, 2.0, 4.0],
                [3.0, 2.0, 1e10, 1.0],
                [2.0, 4.0, 1.0, 1e10],
            ]
        )

        edge_indices, edge_weights = compute_knn_graph(distances, k=2)

        # All weights should be finite and positive
        assert jnp.all(jnp.isfinite(edge_weights))
        assert jnp.all(edge_weights > 0.0)

    def test_k_equals_n_minus_one(self) -> None:
        """Test edge case where k equals n - 1 (all neighbors)."""
        from diffbio.core.graph_utils import compute_knn_graph

        n = 5
        k = n - 1
        distances = jax.random.uniform(jax.random.key(1), (n, n))
        distances = (distances + distances.T) / 2
        distances = distances + jnp.eye(n) * 1e10

        edge_indices, edge_weights = compute_knn_graph(distances, k=k)

        assert edge_indices.shape == (n * k, 2)
        assert edge_weights.shape == (n * k,)

    def test_k_larger_than_n_minus_one_clips(self) -> None:
        """Test k is clipped to n - 1 when larger."""
        from diffbio.core.graph_utils import compute_knn_graph

        n = 4
        k = 10  # Larger than n - 1
        distances = jax.random.uniform(jax.random.key(2), (n, n))
        distances = (distances + distances.T) / 2
        distances = distances + jnp.eye(n) * 1e10

        edge_indices, edge_weights = compute_knn_graph(distances, k=k)

        effective_k = n - 1
        assert edge_indices.shape == (n * effective_k, 2)
        assert edge_weights.shape == (n * effective_k,)

    def test_no_self_loops(self) -> None:
        """Test k-NN graph has no self-loops."""
        from diffbio.core.graph_utils import compute_knn_graph

        n = 8
        k = 3
        distances = jax.random.uniform(jax.random.key(3), (n, n))
        distances = distances + jnp.eye(n) * 1e10

        edge_indices, _ = compute_knn_graph(distances, k=k)

        # No edge (i, i)
        assert jnp.all(edge_indices[:, 0] != edge_indices[:, 1])

    def test_sorted_edge_weights_per_node(self) -> None:
        """Test edge weights for each node are the k smallest distances."""
        from diffbio.core.graph_utils import compute_knn_graph

        n = 6
        k = 3
        distances = jax.random.uniform(jax.random.key(4), (n, n)) + 0.1
        distances = (distances + distances.T) / 2
        distances = distances + jnp.eye(n) * 1e10

        edge_indices, edge_weights = compute_knn_graph(distances, k=k)

        for i in range(n):
            mask = edge_indices[:, 0] == i
            node_weights = edge_weights[mask]
            # These should be the k smallest distances from node i
            sorted_row = jnp.sort(distances[i])
            expected_smallest_k = sorted_row[:k]
            assert jnp.allclose(jnp.sort(node_weights), expected_smallest_k, atol=1e-5)


class TestComputeFuzzyMembership:
    """Tests for compute_fuzzy_membership."""

    def test_basic_shape(self) -> None:
        """Test fuzzy membership returns correct shape."""
        from diffbio.core.graph_utils import compute_fuzzy_membership

        n = 10
        k = 5
        distances = jax.random.uniform(jax.random.key(0), (n, n)) + 0.1
        distances = (distances + distances.T) / 2
        distances = distances + jnp.eye(n) * 1e10

        membership = compute_fuzzy_membership(distances, k=k)

        assert membership.shape == (n, n)

    def test_values_in_range(self) -> None:
        """Test fuzzy membership values are in [0, 1]."""
        from diffbio.core.graph_utils import compute_fuzzy_membership

        n = 15
        k = 5
        distances = jax.random.uniform(jax.random.key(1), (n, n)) + 0.1
        distances = (distances + distances.T) / 2
        distances = distances + jnp.eye(n) * 1e10

        membership = compute_fuzzy_membership(distances, k=k)

        assert jnp.all(membership >= 0.0)
        assert jnp.all(membership <= 1.0)

    def test_diagonal_is_zero(self) -> None:
        """Test diagonal of fuzzy membership is zero (no self-similarity)."""
        from diffbio.core.graph_utils import compute_fuzzy_membership

        n = 10
        k = 4
        distances = jax.random.uniform(jax.random.key(2), (n, n)) + 0.1
        distances = (distances + distances.T) / 2
        distances = distances + jnp.eye(n) * 1e10

        membership = compute_fuzzy_membership(distances, k=k)

        diag = jnp.diag(membership)
        assert jnp.allclose(diag, 0.0, atol=1e-6)

    def test_closer_points_higher_membership(self) -> None:
        """Test that closer points have higher fuzzy membership."""
        from diffbio.core.graph_utils import compute_fuzzy_membership

        # 3 points: 0 is close to 1, far from 2
        distances = jnp.array(
            [
                [1e10, 0.5, 5.0],
                [0.5, 1e10, 4.5],
                [5.0, 4.5, 1e10],
            ]
        )

        membership = compute_fuzzy_membership(distances, k=2)

        # Point 0's membership to point 1 should be higher than to point 2
        assert membership[0, 1] > membership[0, 2]

    def test_local_bandwidth(self) -> None:
        """Test that bandwidth adapts to local density (k-th neighbor distance)."""
        from diffbio.core.graph_utils import compute_fuzzy_membership

        # Two points in a dense region, one far away
        distances = jnp.array(
            [
                [1e10, 0.1, 10.0],
                [0.1, 1e10, 10.0],
                [10.0, 10.0, 1e10],
            ]
        )

        # With k=1, sigma is distance to nearest neighbor
        membership = compute_fuzzy_membership(distances, k=1)

        # Point 2's sigma is 10.0 (far from everything)
        # Points 0,1 have sigma=0.1 (close to each other)
        # So point 2's membership to 0 should be higher than 0's membership to 2
        # (because point 2 has a larger sigma)
        assert membership[2, 0] > membership[0, 2]

    def test_gradient_flow(self) -> None:
        """Test gradients flow through fuzzy membership computation."""
        from diffbio.core.graph_utils import compute_fuzzy_membership

        def loss_fn(distances: jax.Array) -> jax.Array:
            membership = compute_fuzzy_membership(distances, k=3)
            return jnp.sum(membership)

        n = 8
        distances = jax.random.uniform(jax.random.key(3), (n, n)) + 0.1
        distances = (distances + distances.T) / 2
        distances = distances + jnp.eye(n) * 1e10

        grads = jax.grad(loss_fn)(distances)

        assert grads.shape == distances.shape
        assert jnp.all(jnp.isfinite(grads))

    def test_all_values_finite(self) -> None:
        """Test output is fully finite."""
        from diffbio.core.graph_utils import compute_fuzzy_membership

        n = 20
        k = 5
        distances = jax.random.uniform(jax.random.key(4), (n, n)) + 0.1
        distances = (distances + distances.T) / 2
        distances = distances + jnp.eye(n) * 1e10

        membership = compute_fuzzy_membership(distances, k=k)

        assert jnp.all(jnp.isfinite(membership))

    def test_k_clipped_to_n_minus_one(self) -> None:
        """Test k is clipped when larger than n - 1."""
        from diffbio.core.graph_utils import compute_fuzzy_membership

        n = 4
        k = 10
        distances = jax.random.uniform(jax.random.key(5), (n, n)) + 0.1
        distances = (distances + distances.T) / 2
        distances = distances + jnp.eye(n) * 1e10

        # Should not raise, k is clipped internally
        membership = compute_fuzzy_membership(distances, k=k)
        assert membership.shape == (n, n)


class TestSymmetrizeGraph:
    """Tests for symmetrize_graph."""

    def test_result_is_symmetric(self) -> None:
        """Test that output is symmetric."""
        from diffbio.core.graph_utils import symmetrize_graph

        # Create asymmetric adjacency
        adjacency = jnp.array(
            [
                [0.0, 0.8, 0.3],
                [0.2, 0.0, 0.6],
                [0.5, 0.1, 0.0],
            ]
        )

        symmetric = symmetrize_graph(adjacency)

        assert jnp.allclose(symmetric, symmetric.T, atol=1e-6)

    def test_values_in_range(self) -> None:
        """Test output values are in [0, 1] for inputs in [0, 1]."""
        from diffbio.core.graph_utils import symmetrize_graph

        adjacency = jax.random.uniform(jax.random.key(0), (10, 10))
        adjacency = adjacency * (1 - jnp.eye(10))  # Zero diagonal

        symmetric = symmetrize_graph(adjacency)

        assert jnp.all(symmetric >= 0.0)
        assert jnp.all(symmetric <= 1.0)

    def test_fuzzy_union_formula(self) -> None:
        """Test the fuzzy union formula: p + p^T - p * p^T."""
        from diffbio.core.graph_utils import symmetrize_graph

        adjacency = jnp.array(
            [
                [0.0, 0.8, 0.0],
                [0.0, 0.0, 0.5],
                [0.3, 0.0, 0.0],
            ]
        )

        symmetric = symmetrize_graph(adjacency)

        # Manual computation for (0, 1): p=0.8, p^T=0.0 -> 0.8 + 0.0 - 0.8*0.0 = 0.8
        assert jnp.isclose(symmetric[0, 1], 0.8, atol=1e-6)
        # Manual computation for (0, 2): p=0.0, p^T=0.3 -> 0.0 + 0.3 - 0.0*0.3 = 0.3
        assert jnp.isclose(symmetric[0, 2], 0.3, atol=1e-6)
        # Manual computation for (1, 2): p=0.5, p^T=0.0 -> 0.5 + 0.0 - 0.5*0.0 = 0.5
        assert jnp.isclose(symmetric[1, 2], 0.5, atol=1e-6)

    def test_already_symmetric_unchanged(self) -> None:
        """Test that already symmetric input is unchanged."""
        from diffbio.core.graph_utils import symmetrize_graph

        adjacency = jnp.array(
            [
                [0.0, 0.5, 0.3],
                [0.5, 0.0, 0.7],
                [0.3, 0.7, 0.0],
            ]
        )

        symmetric = symmetrize_graph(adjacency)

        # p + p^T - p*p^T with p = p^T gives 2p - p^2 = p(2 - p)
        # For p=0.5: 0.5*(2-0.5) = 0.75
        expected_01 = 0.5 + 0.5 - 0.5 * 0.5  # 0.75
        assert jnp.isclose(symmetric[0, 1], expected_01, atol=1e-6)

    def test_gradient_flow(self) -> None:
        """Test gradients flow through symmetrization."""
        from diffbio.core.graph_utils import symmetrize_graph

        def loss_fn(adjacency: jax.Array) -> jax.Array:
            symmetric = symmetrize_graph(adjacency)
            return jnp.sum(symmetric)

        adjacency = jax.random.uniform(jax.random.key(1), (6, 6))
        adjacency = adjacency * (1 - jnp.eye(6))

        grads = jax.grad(loss_fn)(adjacency)

        assert grads.shape == adjacency.shape
        assert jnp.all(jnp.isfinite(grads))

    def test_zero_adjacency(self) -> None:
        """Test with zero adjacency matrix."""
        from diffbio.core.graph_utils import symmetrize_graph

        adjacency = jnp.zeros((5, 5))
        symmetric = symmetrize_graph(adjacency)

        assert jnp.allclose(symmetric, 0.0)

    def test_diagonal_preserved(self) -> None:
        """Test that diagonal remains zero when input diagonal is zero."""
        from diffbio.core.graph_utils import symmetrize_graph

        adjacency = jax.random.uniform(jax.random.key(2), (8, 8))
        adjacency = adjacency * (1 - jnp.eye(8))

        symmetric = symmetrize_graph(adjacency)

        diag = jnp.diag(symmetric)
        assert jnp.allclose(diag, 0.0, atol=1e-6)


class TestGradientFlow:
    """Tests for end-to-end gradient flow through the full pipeline."""

    def test_full_pipeline_gradient(self) -> None:
        """Test gradient flows through full pipeline (distances, membership, symmetrize)."""
        from diffbio.core.graph_utils import (
            compute_fuzzy_membership,
            compute_pairwise_distances,
            symmetrize_graph,
        )

        def pipeline(features: jax.Array) -> jax.Array:
            distances = compute_pairwise_distances(features, metric="euclidean")
            n_samples = features.shape[0]
            distances = distances + jnp.eye(n_samples) * 1e10
            membership = compute_fuzzy_membership(distances, k=3)
            symmetric = symmetrize_graph(membership)
            return jnp.sum(symmetric)

        features = jax.random.normal(jax.random.key(0), (10, 5))
        grads = jax.grad(pipeline)(features)

        assert grads.shape == features.shape
        assert jnp.all(jnp.isfinite(grads))

    def test_full_pipeline_gradient_cosine(self) -> None:
        """Test gradient flows through full pipeline with cosine metric."""
        from diffbio.core.graph_utils import (
            compute_fuzzy_membership,
            compute_pairwise_distances,
            symmetrize_graph,
        )

        def pipeline(features: jax.Array) -> jax.Array:
            distances = compute_pairwise_distances(features, metric="cosine")
            n_samples = features.shape[0]
            distances = distances + jnp.eye(n_samples) * 1e10
            membership = compute_fuzzy_membership(distances, k=3)
            symmetric = symmetrize_graph(membership)
            return jnp.sum(symmetric)

        features = jax.random.normal(jax.random.key(1), (10, 5))
        grads = jax.grad(pipeline)(features)

        assert grads.shape == features.shape
        assert jnp.all(jnp.isfinite(grads))

    def test_gradient_nonzero(self) -> None:
        """Test that gradients are nonzero (information propagates)."""
        from diffbio.core.graph_utils import (
            compute_fuzzy_membership,
            compute_pairwise_distances,
            symmetrize_graph,
        )

        def pipeline(features: jax.Array) -> jax.Array:
            distances = compute_pairwise_distances(features, metric="euclidean")
            n_samples = features.shape[0]
            distances = distances + jnp.eye(n_samples) * 1e10
            membership = compute_fuzzy_membership(distances, k=3)
            symmetric = symmetrize_graph(membership)
            return jnp.sum(symmetric)

        features = jax.random.normal(jax.random.key(2), (8, 4))
        grads = jax.grad(pipeline)(features)

        # At least some gradients should be nonzero
        assert jnp.any(jnp.abs(grads) > 1e-10)


class TestJITCompatibility:
    """Tests for JIT compilation of all graph utility functions."""

    def test_jit_compute_pairwise_distances_euclidean(self) -> None:
        """Test JIT compilation of compute_pairwise_distances with Euclidean."""
        from diffbio.core.graph_utils import compute_pairwise_distances

        @jax.jit
        def jit_fn(features: jax.Array) -> jax.Array:
            return compute_pairwise_distances(features, metric="euclidean")

        features = jax.random.normal(jax.random.key(0), (10, 5))
        result = jit_fn(features)

        assert result.shape == (10, 10)
        assert jnp.all(jnp.isfinite(result))

    def test_jit_compute_pairwise_distances_cosine(self) -> None:
        """Test JIT compilation of compute_pairwise_distances with cosine."""
        from diffbio.core.graph_utils import compute_pairwise_distances

        @jax.jit
        def jit_fn(features: jax.Array) -> jax.Array:
            return compute_pairwise_distances(features, metric="cosine")

        features = jax.random.normal(jax.random.key(1), (10, 5))
        result = jit_fn(features)

        assert result.shape == (10, 10)
        assert jnp.all(jnp.isfinite(result))

    def test_jit_compute_knn_graph(self) -> None:
        """Test JIT compilation of compute_knn_graph."""
        from diffbio.core.graph_utils import compute_knn_graph

        @jax.jit
        def jit_fn(distances: jax.Array) -> tuple[jax.Array, jax.Array]:
            return compute_knn_graph(distances, k=3)

        n = 8
        distances = jax.random.uniform(jax.random.key(2), (n, n)) + 0.1
        distances = (distances + distances.T) / 2
        distances = distances + jnp.eye(n) * 1e10

        edge_indices, edge_weights = jit_fn(distances)

        assert edge_indices.shape == (n * 3, 2)
        assert jnp.all(jnp.isfinite(edge_weights))

    def test_jit_compute_fuzzy_membership(self) -> None:
        """Test JIT compilation of compute_fuzzy_membership."""
        from diffbio.core.graph_utils import compute_fuzzy_membership

        @jax.jit
        def jit_fn(distances: jax.Array) -> jax.Array:
            return compute_fuzzy_membership(distances, k=3)

        n = 8
        distances = jax.random.uniform(jax.random.key(3), (n, n)) + 0.1
        distances = (distances + distances.T) / 2
        distances = distances + jnp.eye(n) * 1e10

        result = jit_fn(distances)

        assert result.shape == (n, n)
        assert jnp.all(jnp.isfinite(result))

    def test_jit_symmetrize_graph(self) -> None:
        """Test JIT compilation of symmetrize_graph."""
        from diffbio.core.graph_utils import symmetrize_graph

        @jax.jit
        def jit_fn(adjacency: jax.Array) -> jax.Array:
            return symmetrize_graph(adjacency)

        adjacency = jax.random.uniform(jax.random.key(4), (6, 6))
        adjacency = adjacency * (1 - jnp.eye(6))

        result = jit_fn(adjacency)

        assert result.shape == (6, 6)
        assert jnp.allclose(result, result.T, atol=1e-6)

    def test_jit_full_pipeline(self) -> None:
        """Test JIT compilation of the full pipeline."""
        from diffbio.core.graph_utils import (
            compute_fuzzy_membership,
            compute_pairwise_distances,
            symmetrize_graph,
        )

        @jax.jit
        def pipeline(features: jax.Array) -> jax.Array:
            distances = compute_pairwise_distances(features, metric="euclidean")
            n_samples = features.shape[0]
            distances = distances + jnp.eye(n_samples) * 1e10
            membership = compute_fuzzy_membership(distances, k=3)
            return symmetrize_graph(membership)

        features = jax.random.normal(jax.random.key(5), (10, 5))
        result = pipeline(features)

        assert result.shape == (10, 10)
        assert jnp.all(jnp.isfinite(result))
        assert jnp.allclose(result, result.T, atol=1e-6)
