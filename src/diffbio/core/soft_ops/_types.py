"""Type aliases for soft differentiable operations.

SoftBool represents a probability in [0, 1] -- a soft relaxation of a
boolean value. SoftIndex represents a probability distribution over
discrete indices -- a soft relaxation of an integer index.
"""

from jax import Array
from jaxtyping import Float

SoftBool = Float[Array, "..."]
"""Soft boolean: probability in [0, 1]."""

SoftIndex = Float[Array, "..."]
"""Soft index: probabilities summing to 1 along the last axis."""
