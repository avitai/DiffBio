"""Tests for diffbio.sequences.dna module."""

import jax
import jax.numpy as jnp
import pytest

from diffbio.sequences.dna import (
    DNA_ALPHABET,
    DNA_ALPHABET_SIZE,
    complement_dna,
    create_dna_element_data,
    decode_dna_onehot,
    encode_dna_string,
    gc_content,
    phred_to_probability,
    probability_to_phred,
    reverse_complement_dna,
    soft_encode_dna,
)


class TestDNAConstants:
    """Tests for DNA constants."""

    def test_alphabet(self):
        """Verify DNA alphabet is ACGT."""
        assert DNA_ALPHABET == "ACGT"

    def test_alphabet_size(self):
        """Verify alphabet size is 4."""
        assert DNA_ALPHABET_SIZE == 4


class TestEncodeDNAString:
    """Tests for encode_dna_string function."""

    def test_single_nucleotide_a(self):
        """Test encoding single A nucleotide."""
        encoded = encode_dna_string("A")
        expected = jnp.array([[1, 0, 0, 0]], dtype=jnp.float32)
        assert jnp.allclose(encoded, expected)

    def test_single_nucleotide_c(self):
        """Test encoding single C nucleotide."""
        encoded = encode_dna_string("C")
        expected = jnp.array([[0, 1, 0, 0]], dtype=jnp.float32)
        assert jnp.allclose(encoded, expected)

    def test_single_nucleotide_g(self):
        """Test encoding single G nucleotide."""
        encoded = encode_dna_string("G")
        expected = jnp.array([[0, 0, 1, 0]], dtype=jnp.float32)
        assert jnp.allclose(encoded, expected)

    def test_single_nucleotide_t(self):
        """Test encoding single T nucleotide."""
        encoded = encode_dna_string("T")
        expected = jnp.array([[0, 0, 0, 1]], dtype=jnp.float32)
        assert jnp.allclose(encoded, expected)

    def test_sequence_acgt(self):
        """Test encoding ACGT sequence."""
        encoded = encode_dna_string("ACGT")
        expected = jnp.eye(4, dtype=jnp.float32)
        assert jnp.allclose(encoded, expected)

    def test_lowercase_input(self):
        """Test that lowercase input is handled correctly."""
        encoded_lower = encode_dna_string("acgt")
        encoded_upper = encode_dna_string("ACGT")
        assert jnp.allclose(encoded_lower, encoded_upper)

    def test_n_uniform_encoding(self):
        """Test N nucleotide encoded as uniform distribution."""
        encoded = encode_dna_string("N", handle_n="uniform")
        expected = jnp.array([[0.25, 0.25, 0.25, 0.25]], dtype=jnp.float32)
        assert jnp.allclose(encoded, expected)

    def test_n_zero_encoding(self):
        """Test N nucleotide encoded as zeros."""
        encoded = encode_dna_string("N", handle_n="zero")
        expected = jnp.array([[0, 0, 0, 0]], dtype=jnp.float32)
        assert jnp.allclose(encoded, expected)

    def test_mixed_sequence_with_n(self):
        """Test sequence containing N nucleotide."""
        encoded = encode_dna_string("ANA", handle_n="uniform")
        assert encoded.shape == (3, 4)
        # First and last should be A
        assert jnp.allclose(encoded[0], jnp.array([1, 0, 0, 0]))
        assert jnp.allclose(encoded[2], jnp.array([1, 0, 0, 0]))
        # Middle should be uniform
        assert jnp.allclose(encoded[1], jnp.array([0.25, 0.25, 0.25, 0.25]))

    def test_invalid_nucleotide_raises(self):
        """Test that invalid nucleotide raises ValueError."""
        with pytest.raises(ValueError, match="Invalid nucleotide"):
            encode_dna_string("X")

    def test_output_shape(self):
        """Test output shape is (length, 4)."""
        sequence = "ACGTACGT"
        encoded = encode_dna_string(sequence)
        assert encoded.shape == (len(sequence), DNA_ALPHABET_SIZE)

    def test_output_sums_to_one(self):
        """Test each row sums to 1 (except for N with zero encoding)."""
        encoded = encode_dna_string("ACGT")
        row_sums = jnp.sum(encoded, axis=1)
        assert jnp.allclose(row_sums, jnp.ones(4))


class TestDecodeDNAOnehot:
    """Tests for decode_dna_onehot function."""

    def test_roundtrip_simple(self):
        """Test encoding then decoding returns original sequence."""
        original = "ACGT"
        encoded = encode_dna_string(original)
        decoded = decode_dna_onehot(encoded)
        assert decoded == original

    def test_roundtrip_long_sequence(self):
        """Test roundtrip with longer sequence."""
        original = "ACGTACGTACGTACGT"
        encoded = encode_dna_string(original)
        decoded = decode_dna_onehot(encoded)
        assert decoded == original

    def test_low_confidence_returns_n(self):
        """Test that low confidence values return N."""
        # Create low confidence encoding
        low_conf = jnp.array([[0.3, 0.3, 0.2, 0.2]])
        decoded = decode_dna_onehot(low_conf, threshold=0.5)
        assert decoded == "N"

    def test_threshold_boundary(self):
        """Test threshold boundary behavior."""
        # Exactly at threshold should pass
        at_threshold = jnp.array([[0.5, 0.0, 0.0, 0.5]])
        decoded = decode_dna_onehot(at_threshold, threshold=0.5)
        assert decoded in ["A", "T"]  # Either A or T depending on argmax tie-breaking


class TestPhredConversion:
    """Tests for Phred score conversion functions."""

    def test_phred_to_probability_q0(self):
        """Test Q0 gives probability 1.0."""
        prob = phred_to_probability(jnp.array([0.0]))
        assert jnp.allclose(prob, jnp.array([1.0]))

    def test_phred_to_probability_q10(self):
        """Test Q10 gives probability 0.1."""
        prob = phred_to_probability(jnp.array([10.0]))
        assert jnp.allclose(prob, jnp.array([0.1]))

    def test_phred_to_probability_q20(self):
        """Test Q20 gives probability 0.01."""
        prob = phred_to_probability(jnp.array([20.0]))
        assert jnp.allclose(prob, jnp.array([0.01]))

    def test_phred_to_probability_q30(self):
        """Test Q30 gives probability 0.001."""
        prob = phred_to_probability(jnp.array([30.0]))
        assert jnp.allclose(prob, jnp.array([0.001]))

    def test_probability_to_phred_roundtrip(self):
        """Test roundtrip conversion."""
        original_phred = jnp.array([10.0, 20.0, 30.0])
        probs = phred_to_probability(original_phred)
        recovered = probability_to_phred(probs)
        assert jnp.allclose(original_phred, recovered, atol=1e-5)

    def test_probability_to_phred_clips_max(self):
        """Test that very small probabilities are clipped."""
        tiny_prob = jnp.array([1e-15])
        phred = probability_to_phred(tiny_prob, max_phred=60.0)
        assert float(phred[0]) == 60.0


class TestSoftEncodeDNA:
    """Tests for soft_encode_dna function."""

    def test_high_quality_stays_onehot(self):
        """Test high quality sequences stay close to one-hot."""
        seq = encode_dna_string("A")
        quality = jnp.array([40.0])  # High quality
        soft = soft_encode_dna(seq, quality)
        # Should be close to one-hot
        assert jnp.argmax(soft[0]) == 0  # Still A
        assert float(soft[0, 0]) > 0.9  # High confidence

    def test_low_quality_becomes_uniform(self):
        """Test low quality sequences become more uniform."""
        seq = encode_dna_string("A")
        quality = jnp.array([0.0])  # Very low quality (Q0 = 100% error)
        soft = soft_encode_dna(seq, quality)
        # Should be more uniform
        assert jnp.max(soft) < 0.5

    def test_output_sums_to_one(self):
        """Test soft encoding maintains probability distribution."""
        seq = encode_dna_string("ACGT")
        quality = jnp.array([10.0, 20.0, 30.0, 40.0])
        soft = soft_encode_dna(seq, quality)
        row_sums = jnp.sum(soft, axis=1)
        assert jnp.allclose(row_sums, jnp.ones(4))

    def test_temperature_effect(self):
        """Test temperature parameter affects softness."""
        seq = encode_dna_string("A")
        quality = jnp.array([20.0])
        soft_low_temp = soft_encode_dna(seq, quality, temperature=0.1)
        soft_high_temp = soft_encode_dna(seq, quality, temperature=10.0)
        # Lower temperature should be sharper (higher max)
        assert jnp.max(soft_low_temp) > jnp.max(soft_high_temp)


class TestComplementDNA:
    """Tests for DNA complement functions."""

    def test_complement_a_to_t(self):
        """Test A complements to T."""
        a = encode_dna_string("A")
        comp = complement_dna(a)
        decoded = decode_dna_onehot(comp)
        assert decoded == "T"

    def test_complement_c_to_g(self):
        """Test C complements to G."""
        c = encode_dna_string("C")
        comp = complement_dna(c)
        decoded = decode_dna_onehot(comp)
        assert decoded == "G"

    def test_complement_g_to_c(self):
        """Test G complements to C."""
        g = encode_dna_string("G")
        comp = complement_dna(g)
        decoded = decode_dna_onehot(comp)
        assert decoded == "C"

    def test_complement_t_to_a(self):
        """Test T complements to A."""
        t = encode_dna_string("T")
        comp = complement_dna(t)
        decoded = decode_dna_onehot(comp)
        assert decoded == "A"

    def test_double_complement_is_identity(self):
        """Test complement(complement(x)) = x."""
        seq = encode_dna_string("ACGT")
        double_comp = complement_dna(complement_dna(seq))
        assert jnp.allclose(seq, double_comp)


class TestReverseComplementDNA:
    """Tests for reverse complement function."""

    def test_reverse_complement_simple(self):
        """Test reverse complement of ACGT."""
        seq = encode_dna_string("ACGT")
        rc = reverse_complement_dna(seq)
        decoded = decode_dna_onehot(rc)
        # ACGT -> complement: TGCA -> reverse: ACGT
        # Wait, that's wrong. Let me recalculate:
        # ACGT -> reverse: TGCA -> complement: ACGT
        # Actually: ACGT -> complement: TGCA -> reverse: ACGT
        # No: complement(A)=T, complement(C)=G, complement(G)=C, complement(T)=A
        # ACGT -> TGCA (complement) -> ACGT (reverse)
        # Hmm, let me think again:
        # reverse_complement = reverse(complement(seq))
        # So: ACGT -> complement -> TGCA -> reverse -> ACGT
        # Actually our implementation does: complement(seq[::-1])
        # ACGT[::-1] = TGCA -> complement(TGCA) = ACGT
        # This is the same result!
        assert decoded == "ACGT"

    def test_reverse_complement_asymmetric(self):
        """Test reverse complement of asymmetric sequence."""
        seq = encode_dna_string("AACGT")
        rc = reverse_complement_dna(seq)
        decoded = decode_dna_onehot(rc)
        # AACGT -> reverse -> TGCAA -> complement -> ACGTT
        assert decoded == "ACGTT"

    def test_double_reverse_complement_is_identity(self):
        """Test reverse_complement(reverse_complement(x)) = x."""
        seq = encode_dna_string("ACGTACGT")
        double_rc = reverse_complement_dna(reverse_complement_dna(seq))
        assert jnp.allclose(seq, double_rc)


class TestGCContent:
    """Tests for GC content calculation."""

    def test_all_gc(self):
        """Test sequence of all G/C has GC content 1.0."""
        seq = encode_dna_string("GCGCGC")
        gc = gc_content(seq)
        assert jnp.allclose(gc, 1.0)

    def test_no_gc(self):
        """Test sequence of all A/T has GC content 0.0."""
        seq = encode_dna_string("ATATAT")
        gc = gc_content(seq)
        assert jnp.allclose(gc, 0.0)

    def test_half_gc(self):
        """Test balanced sequence has GC content 0.5."""
        seq = encode_dna_string("ACGT")
        gc = gc_content(seq)
        assert jnp.allclose(gc, 0.5)

    def test_gc_content_range(self):
        """Test GC content is always in [0, 1]."""
        seq = encode_dna_string("GGCCAT")
        gc = gc_content(seq)
        assert 0.0 <= float(gc) <= 1.0


class TestCreateDNAElementData:
    """Tests for create_dna_element_data function."""

    def test_string_input(self):
        """Test creating element data from string."""
        data = create_dna_element_data("ACGT")
        assert "sequence" in data
        assert data["sequence"].shape == (4, 4)

    def test_array_input(self):
        """Test creating element data from array."""
        encoded = encode_dna_string("ACGT")
        data = create_dna_element_data(encoded)
        assert "sequence" in data
        assert jnp.allclose(data["sequence"], encoded)

    def test_with_quality_scores(self):
        """Test creating element data with quality scores."""
        quality = jnp.array([30.0, 30.0, 30.0, 30.0])
        data = create_dna_element_data("ACGT", quality_scores=quality)
        assert "sequence" in data
        assert "quality_scores" in data
        assert jnp.allclose(data["quality_scores"], quality)

    def test_without_quality_scores(self):
        """Test creating element data without quality scores."""
        data = create_dna_element_data("ACGT")
        assert "sequence" in data
        assert "quality_scores" not in data


class TestGradientFlow:
    """Tests for gradient flow through DNA operations."""

    def test_soft_encode_gradient(self):
        """Test gradients flow through soft encoding."""
        seq = encode_dna_string("ACGT")
        quality = jnp.array([20.0, 20.0, 20.0, 20.0])

        def loss_fn(q):
            soft = soft_encode_dna(seq, q)
            return jnp.sum(soft)

        grad = jax.grad(loss_fn)(quality)
        assert grad is not None
        assert grad.shape == quality.shape

    def test_gc_content_gradient(self):
        """Test gradients flow through GC content."""

        def loss_fn(encoded):
            return gc_content(encoded)

        seq = encode_dna_string("ACGT").astype(jnp.float32)
        grad = jax.grad(loss_fn)(seq)
        assert grad is not None
        assert grad.shape == seq.shape

    def test_complement_gradient(self):
        """Test gradients flow through complement."""

        def loss_fn(encoded):
            return jnp.sum(complement_dna(encoded))

        seq = encode_dna_string("ACGT").astype(jnp.float32)
        grad = jax.grad(loss_fn)(seq)
        assert grad is not None
        assert grad.shape == seq.shape


class TestJITCompatibility:
    """Tests for JAX JIT compilation compatibility."""

    def test_encode_jit(self):
        """Test encode_dna_string works with JIT (indirectly via array ops)."""
        # encode_dna_string itself can't be JIT'd due to string input
        # But the encoded result should work in JIT functions
        encoded = encode_dna_string("ACGT")

        @jax.jit
        def process(x):
            return x * 2

        result = process(encoded)
        assert jnp.allclose(result, encoded * 2)

    def test_soft_encode_jit(self):
        """Test soft_encode_dna is JIT compatible."""
        seq = encode_dna_string("ACGT")
        quality = jnp.array([20.0, 20.0, 20.0, 20.0])

        jit_soft_encode = jax.jit(soft_encode_dna, static_argnums=(2,))
        result = jit_soft_encode(seq, quality, 1.0)
        assert result.shape == seq.shape

    def test_gc_content_jit(self):
        """Test gc_content is JIT compatible."""
        seq = encode_dna_string("GCGCAT")

        @jax.jit
        def compute_gc(x):
            return gc_content(x)

        result = compute_gc(seq)
        assert 0.0 <= float(result) <= 1.0

    def test_complement_jit(self):
        """Test complement_dna is JIT compatible."""
        seq = encode_dna_string("ACGT")

        @jax.jit
        def compute_complement(x):
            return complement_dna(x)

        result = compute_complement(seq)
        assert result.shape == seq.shape
