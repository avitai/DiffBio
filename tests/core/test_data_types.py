"""Tests for data types and protocols.

Following TDD: These tests define the expected type protocols and aliases
that provide type safety across the DiffBio codebase.
"""

import jax.numpy as jnp
import pytest
from jaxtyping import Array


class TestSequenceProtocol:
    """Tests for sequence data protocol."""

    def test_sequence_data_structure(self):
        """Test SequenceData TypedDict structure."""
        from diffbio.core.data_types import SequenceData

        # Should be able to create valid SequenceData
        data: SequenceData = {
            "sequence": jnp.zeros((10, 4)),
        }
        assert "sequence" in data

    def test_sequence_data_with_quality(self):
        """Test SequenceData with optional quality scores."""
        from diffbio.core.data_types import SequenceData

        data: SequenceData = {
            "sequence": jnp.zeros((10, 4)),
            "quality_scores": jnp.zeros((10,)),
        }
        assert "quality_scores" in data


class TestAlignmentResult:
    """Tests for alignment result types."""

    def test_alignment_result_structure(self):
        """Test AlignmentResult TypedDict structure."""
        from diffbio.core.data_types import AlignmentResultData

        data: AlignmentResultData = {
            "score": jnp.array(0.0),
            "alignment_matrix": jnp.zeros((5, 5)),
            "soft_alignment": jnp.zeros((4, 4)),
        }
        assert "score" in data
        assert "alignment_matrix" in data


class TestOperatorOutputs:
    """Tests for operator output type aliases."""

    def test_operator_output_tuple(self):
        """Test OperatorOutput type structure."""
        from diffbio.core.data_types import OperatorOutput

        # OperatorOutput is a tuple of (data, state, metadata)
        output: OperatorOutput = ({"key": jnp.array(1.0)}, {}, None)
        assert len(output) == 3

    def test_state_dict_type(self):
        """Test StateDict type alias."""
        from diffbio.core.data_types import StateDict

        state: StateDict = {"hidden": jnp.zeros((10,)), "count": 5}
        assert isinstance(state, dict)


class TestArrayTypeAliases:
    """Tests for array type aliases."""

    def test_batch_array(self):
        """Test BatchArray type alias for batched data."""
        from diffbio.core.data_types import BatchArray

        # BatchArray is Array with batch dimension first
        batch: BatchArray = jnp.zeros((32, 100, 4))  # (batch, length, features)
        assert batch.ndim == 3

    def test_sequence_array(self):
        """Test SequenceArray type alias."""
        from diffbio.core.data_types import SequenceArray

        # SequenceArray is Array of shape (length, alphabet_size)
        seq: SequenceArray = jnp.zeros((100, 4))
        assert seq.ndim == 2

    def test_probability_array(self):
        """Test ProbabilityArray type alias."""
        from diffbio.core.data_types import ProbabilityArray

        # ProbabilityArray should be in [0, 1]
        probs: ProbabilityArray = jnp.array([0.1, 0.2, 0.7])
        assert jnp.all(probs >= 0) and jnp.all(probs <= 1)


class TestProtocols:
    """Tests for protocol interfaces."""

    def test_differentiable_operator_protocol(self):
        """Test DifferentiableOperator protocol."""
        from diffbio.core.data_types import DifferentiableOperator

        # Protocol defines apply method
        assert hasattr(DifferentiableOperator, "apply")

    def test_sequence_encoder_protocol(self):
        """Test SequenceEncoder protocol."""
        from diffbio.core.data_types import SequenceEncoder

        # Protocol defines encode and decode methods
        assert hasattr(SequenceEncoder, "encode")
        assert hasattr(SequenceEncoder, "decode")


class TestConfigTypes:
    """Tests for configuration type aliases."""

    def test_temperature_value(self):
        """Test Temperature type alias."""
        from diffbio.core.data_types import Temperature

        # Temperature is a float > 0
        temp: Temperature = 1.0
        assert temp > 0

    def test_probability_value(self):
        """Test Probability type alias."""
        from diffbio.core.data_types import Probability

        # Probability is a float in [0, 1]
        prob: Probability = 0.5
        assert 0 <= prob <= 1
