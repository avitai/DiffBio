"""Integration tests for DiffBio operators with Datarax pipeline composition.

These tests verify that DiffBio operators can:
1. Process Datarax Batch objects via apply_batch() for all operators
2. Use the apply() method for single-element processing
3. Be composed with other operators via Datarax Batch pipelines
4. Maintain gradient flow through composed pipelines

Note on apply_batch() and dynamic output structure:
    Datarax's apply_batch() now supports operators that add new keys to output,
    using get_output_structure() with jax.eval_shape for automatic discovery.

    All operators can use apply_batch():
    - QualityFilter: {sequence, quality_scores} -> {sequence, quality_scores}
    - SmithWaterman: {seq1, seq2} -> {seq1, seq2, score, alignment_matrix, soft_alignment}
    - Pileup: {reads, positions, quality} -> {reads, positions, quality, pileup}
    - Classifier: {pileup_window} -> {pileup_window, logits, probabilities}

Note on gradient tests:
    The __call__ method increments _iteration_count which can't be mutated inside
    JAX gradient tracing. Use apply() directly for gradient tests.
"""

import jax
import jax.numpy as jnp
import pytest
from datarax.typing import Batch, Element

from diffbio.operators.alignment import SmoothSmithWaterman, SmithWatermanConfig
from diffbio.operators.alignment.scoring import create_dna_scoring_matrix
from diffbio.operators.quality_filter import (
    DifferentiableQualityFilter,
    QualityFilterConfig,
)
from diffbio.operators.variant import (
    DifferentiablePileup,
    PileupConfig,
    VariantClassifier,
    VariantClassifierConfig,
)
from diffbio.sequences.dna import encode_dna_string


class TestQualityFilterBatchProcessing:
    """Tests for QualityFilter with Datarax Batch objects."""

    @pytest.fixture
    def quality_filter(self, rngs):
        config = QualityFilterConfig(initial_threshold=20.0)
        return DifferentiableQualityFilter(config, rngs=rngs)

    @pytest.fixture
    def sample_batch(self):
        """Create a batch of DNA sequences with quality scores."""
        elements = []
        for i in range(4):
            seq = encode_dna_string("ACGTACGT")
            quality = jnp.array([30.0 - i * 5, 25.0, 20.0, 15.0, 10.0, 5.0, 0.0, 40.0])
            data = {"sequence": seq, "quality_scores": quality}
            state = {"processed": jnp.array(0)}
            elements.append(Element(data=data, state=state))
        return Batch(elements)

    def test_apply_batch_processes_all_elements(self, quality_filter, sample_batch):
        """Test apply_batch processes entire batch."""
        result_batch = quality_filter.apply_batch(sample_batch)

        # Verify batch structure preserved
        assert result_batch.batch_size == sample_batch.batch_size

        # Verify data transformed
        result_data = result_batch.data.get_value()
        assert "sequence" in result_data
        assert result_data["sequence"].shape[0] == 4  # Batch size

    def test_callable_interface(self, quality_filter, sample_batch):
        """Test __call__ interface works with Batch."""
        result_batch = quality_filter(sample_batch)

        assert result_batch.batch_size == sample_batch.batch_size
        result_data = result_batch.data.get_value()
        assert "sequence" in result_data


class TestSmithWatermanBatchProcessing:
    """Tests for SmoothSmithWaterman with Datarax Batch objects.

    SmithWaterman is a structure-changing operator (adds score, alignment_matrix, etc.),
    which now works with apply_batch() thanks to Datarax's dynamic output structure support.
    """

    @pytest.fixture
    def aligner(self, rngs):
        config = SmithWatermanConfig(temperature=1.0)
        scoring = create_dna_scoring_matrix(match=2.0, mismatch=-1.0)
        return SmoothSmithWaterman(config, scoring_matrix=scoring, rngs=rngs)

    @pytest.fixture
    def sample_alignment_batch(self):
        """Create a batch of sequence pairs for alignment."""
        sequences = ["ACGT", "ACGG", "TGCA", "AAAA"]
        reference = "ACGT"
        elements = []
        for seq in sequences:
            seq1 = encode_dna_string(seq)
            seq2 = encode_dna_string(reference)
            data = {"seq1": seq1, "seq2": seq2}
            state = {"aligned": jnp.array(0)}
            elements.append(Element(data=data, state=state))
        return Batch(elements)

    def test_apply_single_element(self, aligner):
        """Test apply() processes a single element correctly."""
        seq1 = encode_dna_string("ACGT")
        seq2 = encode_dna_string("ACGG")
        data = {"seq1": seq1, "seq2": seq2}
        state = {"aligned": jnp.array(0)}

        result_data, result_state, _ = aligner.apply(data, state, None)

        assert "score" in result_data
        assert "alignment_matrix" in result_data
        assert "soft_alignment" in result_data
        # Input keys preserved
        assert "seq1" in result_data
        assert "seq2" in result_data

    def test_apply_batch_alignment(self, aligner, sample_alignment_batch):
        """Test apply_batch() processes entire batch with dynamic output structure."""
        result_batch = aligner.apply_batch(sample_alignment_batch)

        # Verify batch structure
        assert result_batch.batch_size == 4

        # Verify output contains new keys
        result_data = result_batch.data.get_value()
        assert "score" in result_data
        assert "alignment_matrix" in result_data
        assert "soft_alignment" in result_data
        # Input keys preserved
        assert "seq1" in result_data
        assert "seq2" in result_data

        # Verify batched shapes
        assert result_data["score"].shape == (4,)
        assert result_data["alignment_matrix"].shape[0] == 4

        # First sequence (exact match) should have highest score
        assert result_data["score"][0] >= result_data["score"][1]

    def test_callable_interface(self, aligner, sample_alignment_batch):
        """Test __call__ interface works with structure-changing batch."""
        result_batch = aligner(sample_alignment_batch)

        assert result_batch.batch_size == 4
        result_data = result_batch.data.get_value()
        assert "score" in result_data


class TestPileupBatchProcessing:
    """Tests for DifferentiablePileup with Datarax Batch objects.

    Pileup is a structure-changing operator (adds 'pileup' key to output),
    which now works with apply_batch() thanks to Datarax's dynamic output structure support.
    """

    @pytest.fixture
    def pileup(self, rngs):
        # reference_length=30 must match test data
        config = PileupConfig(use_quality_weights=True, reference_length=30)
        return DifferentiablePileup(config, rngs=rngs)

    @pytest.fixture
    def sample_reads_batch(self):
        """Create batch of read data for pileup generation."""
        key = jax.random.PRNGKey(42)
        batch_size = 3
        num_reads = 5
        read_length = 10
        reference_length = 30  # Must match config.reference_length

        elements = []
        keys = jax.random.split(key, batch_size * 3)

        for i in range(batch_size):
            k1, k2, k3 = keys[i * 3 : (i + 1) * 3]
            indices = jax.random.randint(k1, (num_reads, read_length), 0, 4)
            reads = jax.nn.one_hot(indices, 4)
            positions = jax.random.randint(k2, (num_reads,), 0, reference_length - read_length)
            quality = jax.random.uniform(k3, (num_reads, read_length), minval=10.0, maxval=40.0)

            data = {"reads": reads, "positions": positions, "quality": quality}
            state = {"coverage": jnp.array(0.0)}
            elements.append(Element(data=data, state=state))

        return Batch(elements)

    def test_apply_single_element(self, pileup):
        """Test apply() processes a single element correctly."""
        key = jax.random.PRNGKey(42)
        k1, k2, k3 = jax.random.split(key, 3)

        indices = jax.random.randint(k1, (5, 10), 0, 4)
        reads = jax.nn.one_hot(indices, 4)
        positions = jax.random.randint(k2, (5,), 0, 20)
        quality = jax.random.uniform(k3, (5, 10), minval=10.0, maxval=40.0)

        data = {"reads": reads, "positions": positions, "quality": quality}
        state = {"coverage": jnp.array(0.0)}

        result_data, result_state, _ = pileup.apply(data, state, None)

        assert "pileup" in result_data
        assert "reads" in result_data  # Input preserved
        assert "positions" in result_data
        assert "quality" in result_data
        assert result_data["pileup"].shape == (30, 4)  # reference_length, 4

    def test_apply_batch_pileup(self, pileup, sample_reads_batch):
        """Test apply_batch() processes entire batch with dynamic output structure."""
        result_batch = pileup.apply_batch(sample_reads_batch)

        # Verify batch structure
        assert result_batch.batch_size == 3

        # Verify output contains new keys
        result_data = result_batch.data.get_value()
        assert "pileup" in result_data
        # Input keys preserved
        assert "reads" in result_data
        assert "positions" in result_data
        assert "quality" in result_data

        # Verify batched shapes (batch_size, reference_length, 4)
        assert result_data["pileup"].shape == (3, 30, 4)

    def test_callable_interface(self, pileup, sample_reads_batch):
        """Test __call__ interface works with structure-changing batch."""
        result_batch = pileup(sample_reads_batch)

        assert result_batch.batch_size == 3
        result_data = result_batch.data.get_value()
        assert "pileup" in result_data

    def test_pileup_output_is_valid_distribution(self, pileup, sample_reads_batch):
        """Test pileup outputs are valid probability distributions."""
        result_batch = pileup.apply_batch(sample_reads_batch)
        result_data = result_batch.data.get_value()
        batched_pileup = result_data["pileup"]

        # Each position should sum to 1 (valid probability distribution)
        sums = batched_pileup.sum(axis=-1)
        assert jnp.allclose(sums, 1.0, atol=1e-5)


class TestClassifierBatchProcessing:
    """Tests for VariantClassifier with Datarax Batch objects.

    Classifier is a structure-changing operator (adds 'logits', 'probabilities'),
    which now works with apply_batch() thanks to Datarax's dynamic output structure support.
    """

    @pytest.fixture
    def classifier(self, rngs):
        config = VariantClassifierConfig(num_classes=3, hidden_dim=32, input_window=11)
        classifier = VariantClassifier(config, rngs=rngs)
        classifier.eval()  # Disable dropout for deterministic testing
        return classifier

    @pytest.fixture
    def sample_pileup_batch(self):
        """Create batch of pileup windows for classification."""
        key = jax.random.PRNGKey(42)
        batch_size = 4
        window_size = 11

        elements = []
        keys = jax.random.split(key, batch_size)
        for k in keys:
            window = jax.random.uniform(k, (window_size, 4))
            window = window / window.sum(axis=-1, keepdims=True)
            data = {"pileup_window": window}
            state = {"variant_called": jnp.array(0)}
            elements.append(Element(data=data, state=state))

        return Batch(elements)

    def test_apply_single_element(self, classifier):
        """Test apply() processes a single element correctly."""
        key = jax.random.PRNGKey(42)
        window = jax.random.uniform(key, (11, 4))
        window = window / window.sum(axis=-1, keepdims=True)

        data = {"pileup_window": window}
        state = {"variant_called": jnp.array(0)}

        result_data, _, _ = classifier.apply(data, state, None)

        assert "logits" in result_data
        assert "probabilities" in result_data
        assert "pileup_window" in result_data  # Input preserved
        assert result_data["logits"].shape == (3,)  # num_classes

    def test_apply_batch_classification(self, classifier, sample_pileup_batch):
        """Test apply_batch() processes entire batch with dynamic output structure."""
        result_batch = classifier.apply_batch(sample_pileup_batch)

        # Verify batch structure
        assert result_batch.batch_size == 4

        # Verify output contains new keys
        result_data = result_batch.data.get_value()
        assert "logits" in result_data
        assert "probabilities" in result_data
        # Input key preserved
        assert "pileup_window" in result_data

        # Verify batched shapes (batch_size, num_classes)
        assert result_data["logits"].shape == (4, 3)
        assert result_data["probabilities"].shape == (4, 3)

    def test_callable_interface(self, classifier, sample_pileup_batch):
        """Test __call__ interface works with structure-changing batch."""
        result_batch = classifier(sample_pileup_batch)

        assert result_batch.batch_size == 4
        result_data = result_batch.data.get_value()
        assert "logits" in result_data

    def test_probabilities_are_valid(self, classifier, sample_pileup_batch):
        """Test classifier outputs valid probability distributions."""
        result_batch = classifier.apply_batch(sample_pileup_batch)
        result_data = result_batch.data.get_value()
        batched_probs = result_data["probabilities"]

        # Each prediction should sum to 1
        sums = batched_probs.sum(axis=-1)
        assert jnp.allclose(sums, 1.0, atol=1e-5)

        # All probabilities should be non-negative
        assert jnp.all(batched_probs >= 0)


class TestOperatorComposition:
    """Tests for composing multiple DiffBio operators."""

    def test_quality_filter_chain(self, rngs):
        """Test chaining multiple quality filters."""
        filter1 = DifferentiableQualityFilter(
            QualityFilterConfig(initial_threshold=20.0), rngs=rngs
        )
        filter2 = DifferentiableQualityFilter(
            QualityFilterConfig(initial_threshold=25.0), rngs=rngs
        )

        # Create batch
        seq = encode_dna_string("ACGTACGT")
        quality = jnp.array([30.0, 28.0, 26.0, 24.0, 22.0, 20.0, 18.0, 40.0])
        data = {"sequence": seq, "quality_scores": quality}
        state = {"processed": jnp.array(0)}
        batch = Batch([Element(data=data, state=state)])

        # Apply sequentially
        result1 = filter1(batch)
        result2 = filter2(result1)

        # Verify double filtering reduces signal more
        original_sum = jnp.sum(seq)
        result1_data = result1.data.get_value()
        result2_data = result2.data.get_value()
        result1_sum = jnp.sum(result1_data["sequence"])
        result2_sum = jnp.sum(result2_data["sequence"])

        # Second filter should reduce further (higher threshold)
        assert result2_sum <= result1_sum <= original_sum

    def test_aligner_composition(self, rngs):
        """Test composing aligners with different temperatures.

        Using apply() directly for single-element comparison of temperature effects.
        """
        scoring = create_dna_scoring_matrix(match=2.0, mismatch=-1.0)

        aligner_sharp = SmoothSmithWaterman(
            SmithWatermanConfig(temperature=0.5), scoring_matrix=scoring, rngs=rngs
        )
        aligner_smooth = SmoothSmithWaterman(
            SmithWatermanConfig(temperature=2.0), scoring_matrix=scoring, rngs=rngs
        )

        seq1 = encode_dna_string("ACGT")
        seq2 = encode_dna_string("ACGG")
        data = {"seq1": seq1, "seq2": seq2}
        state = {"aligned": jnp.array(0)}

        # Apply both aligners using apply() directly
        result_sharp, _, _ = aligner_sharp.apply(data, state, None)
        result_smooth, _, _ = aligner_smooth.apply(data, state, None)

        # Different temperatures should give different soft alignments
        sharp_align = result_sharp["soft_alignment"]
        smooth_align = result_smooth["soft_alignment"]

        # Sharp should have more peaked distribution
        sharp_max = jnp.max(sharp_align)
        smooth_max = jnp.max(smooth_align)
        assert sharp_max >= smooth_max


class TestGradientFlowThroughPipelines:
    """Tests for gradient flow through composed operators.

    Note: These tests use apply() directly instead of __call__() because
    __call__ mutates _iteration_count which can't be done inside JAX tracing.
    """

    def test_gradient_through_quality_filter(self, rngs):
        """Test gradients flow through quality filter."""
        filter_op = DifferentiableQualityFilter(
            QualityFilterConfig(initial_threshold=20.0), rngs=rngs
        )

        seq = encode_dna_string("ACGT")
        quality = jnp.array([25.0, 15.0, 25.0, 15.0])

        def loss_fn(sequence):
            data = {"sequence": sequence, "quality_scores": quality}
            result_data, _, _ = filter_op.apply(data, {}, None)
            return jnp.sum(result_data["sequence"])

        grad = jax.grad(loss_fn)(seq)

        assert grad is not None
        assert grad.shape == seq.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_gradient_through_aligner(self, rngs):
        """Test gradients flow through aligner."""
        scoring = create_dna_scoring_matrix(match=2.0, mismatch=-1.0)
        aligner = SmoothSmithWaterman(
            SmithWatermanConfig(temperature=1.0), scoring_matrix=scoring, rngs=rngs
        )

        seq1 = encode_dna_string("ACGT")
        seq2 = encode_dna_string("ACGG")

        def loss_fn(s1):
            data = {"seq1": s1, "seq2": seq2}
            result_data, _, _ = aligner.apply(data, {}, None)
            return result_data["score"]

        grad = jax.grad(loss_fn)(seq1)

        assert grad is not None
        assert grad.shape == seq1.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_gradient_through_classifier(self, rngs):
        """Test gradients flow through classifier."""
        config = VariantClassifierConfig(num_classes=3, hidden_dim=32, input_window=5)
        classifier = VariantClassifier(config, rngs=rngs)
        classifier.eval()

        key = jax.random.PRNGKey(42)
        pileup_window = jax.random.uniform(key, (5, 4))
        pileup_window = pileup_window / pileup_window.sum(axis=-1, keepdims=True)

        def loss_fn(window):
            data = {"pileup_window": window}
            result_data, _, _ = classifier.apply(data, {}, None)
            return jnp.sum(result_data["logits"])

        grad = jax.grad(loss_fn)(pileup_window)

        assert grad is not None
        assert grad.shape == pileup_window.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_gradient_through_pileup(self, rngs):
        """Test gradients flow through pileup generation."""
        config = PileupConfig(reference_length=20)
        pileup = DifferentiablePileup(config, rngs=rngs)

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        indices = jax.random.randint(k1, (5, 10), 0, 4)
        reads = jax.nn.one_hot(indices, 4).astype(jnp.float32)
        positions = jax.random.randint(k2, (5,), 0, 10)
        quality = jnp.ones((5, 10)) * 30.0

        def loss_fn(r):
            data = {"reads": r, "positions": positions, "quality": quality}
            result_data, _, _ = pileup.apply(data, {}, None)
            return jnp.sum(result_data["pileup"])

        grad = jax.grad(loss_fn)(reads)

        assert grad is not None
        assert grad.shape == reads.shape
        assert jnp.all(jnp.isfinite(grad))

    def test_gradient_through_composed_pipeline(self, rngs):
        """Test gradients flow through a composed pileup -> classifier pipeline."""
        pileup_config = PileupConfig(reference_length=11)
        pileup_op = DifferentiablePileup(pileup_config, rngs=rngs)

        classifier_config = VariantClassifierConfig(num_classes=3, hidden_dim=32, input_window=11)
        classifier = VariantClassifier(classifier_config, rngs=rngs)
        classifier.eval()

        key = jax.random.PRNGKey(42)
        k1, k2 = jax.random.split(key)
        indices = jax.random.randint(k1, (3, 6), 0, 4)
        reads = jax.nn.one_hot(indices, 4).astype(jnp.float32)
        positions = jax.random.randint(k2, (3,), 0, 5)
        quality = jnp.ones((3, 6)) * 30.0

        def loss_fn(r):
            # Step 1: Generate pileup
            pileup_data = {"reads": r, "positions": positions, "quality": quality}
            pileup_result, _, _ = pileup_op.apply(pileup_data, {}, None)

            # Step 2: Classify (pileup is the window)
            classifier_data = {"pileup_window": pileup_result["pileup"]}
            classifier_result, _, _ = classifier.apply(classifier_data, {}, None)

            return jnp.sum(classifier_result["logits"])

        grad = jax.grad(loss_fn)(reads)

        assert grad is not None
        assert grad.shape == reads.shape
        assert jnp.all(jnp.isfinite(grad))


class TestOperatorModuleInterface:
    """Tests verifying OperatorModule interface compliance."""

    def test_quality_filter_has_apply_method(self, rngs):
        """Test QualityFilter has proper apply() signature."""
        op = DifferentiableQualityFilter(QualityFilterConfig(), rngs=rngs)

        # Should have apply method
        assert hasattr(op, "apply")
        assert callable(op.apply)

        # Test apply with correct signature
        data = {"sequence": jnp.ones((4, 4)), "quality_scores": jnp.ones(4) * 30}
        state = {}
        result_data, result_state, result_metadata = op.apply(data, state, None)

        assert "sequence" in result_data

    def test_aligner_has_apply_method(self, rngs):
        """Test SmoothSmithWaterman has proper apply() signature."""
        scoring = create_dna_scoring_matrix(match=2.0, mismatch=-1.0)
        op = SmoothSmithWaterman(SmithWatermanConfig(), scoring_matrix=scoring, rngs=rngs)

        assert hasattr(op, "apply")
        assert callable(op.apply)

        data = {"seq1": jnp.ones((4, 4)), "seq2": jnp.ones((4, 4))}
        state = {}
        result_data, result_state, result_metadata = op.apply(data, state, None)

        assert "score" in result_data
        assert "alignment_matrix" in result_data

    def test_pileup_has_apply_method(self, rngs):
        """Test DifferentiablePileup has proper apply() signature."""
        # reference_length is set in config, not data
        op = DifferentiablePileup(PileupConfig(reference_length=20), rngs=rngs)

        assert hasattr(op, "apply")
        assert callable(op.apply)

        key = jax.random.PRNGKey(42)
        indices = jax.random.randint(key, (5, 10), 0, 4)
        reads = jax.nn.one_hot(indices, 4)
        # reference_length is NOT in data - it's in config
        data = {
            "reads": reads,
            "positions": jnp.array([0, 2, 4, 6, 8]),
            "quality": jnp.ones((5, 10)) * 30,
        }
        state = {}
        result_data, _, _ = op.apply(data, state, None)

        assert "pileup" in result_data

    def test_classifier_has_apply_method(self, rngs):
        """Test VariantClassifier has proper apply() signature."""
        op = VariantClassifier(VariantClassifierConfig(input_window=5), rngs=rngs)
        op.eval()

        assert hasattr(op, "apply")
        assert callable(op.apply)

        data = {"pileup_window": jnp.ones((5, 4)) / 4}
        state = {}
        result_data, result_state, result_metadata = op.apply(data, state, None)

        assert "logits" in result_data
        assert "probabilities" in result_data

    def test_operators_have_config_attribute(self, rngs):
        """Test all operators store their config."""
        scoring = create_dna_scoring_matrix(match=2.0, mismatch=-1.0)

        qf = DifferentiableQualityFilter(QualityFilterConfig(), rngs=rngs)
        sw = SmoothSmithWaterman(SmithWatermanConfig(), scoring_matrix=scoring, rngs=rngs)
        pileup = DifferentiablePileup(PileupConfig(), rngs=rngs)
        classifier = VariantClassifier(VariantClassifierConfig(), rngs=rngs)

        assert hasattr(qf, "config")
        assert hasattr(sw, "config")
        assert hasattr(pileup, "config")
        assert hasattr(classifier, "config")

    def test_operators_have_stochastic_flag(self, rngs):
        """Test all operators have stochastic flag from OperatorConfig."""
        scoring = create_dna_scoring_matrix(match=2.0, mismatch=-1.0)

        qf = DifferentiableQualityFilter(QualityFilterConfig(), rngs=rngs)
        sw = SmoothSmithWaterman(SmithWatermanConfig(), scoring_matrix=scoring, rngs=rngs)
        pileup = DifferentiablePileup(PileupConfig(), rngs=rngs)
        classifier = VariantClassifier(VariantClassifierConfig(), rngs=rngs)

        # All should be deterministic (stochastic=False)
        assert qf.stochastic is False
        assert sw.stochastic is False
        assert pileup.stochastic is False
        assert classifier.stochastic is False
