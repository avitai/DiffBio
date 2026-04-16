"""Shared validation for transformer-style operator configs."""


def validate_transformer_encoder_shape(
    *,
    hidden_dim: int,
    num_layers: int,
    num_heads: int,
    intermediate_dim: int,
    max_length: int,
    dropout_rate: float,
) -> None:
    """Validate common transformer encoder hyperparameters."""
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive.")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive.")
    if num_heads <= 0:
        raise ValueError("num_heads must be positive.")
    if hidden_dim % num_heads != 0:
        raise ValueError("hidden_dim must be divisible by num_heads.")
    if intermediate_dim <= 0:
        raise ValueError("intermediate_dim must be positive.")
    if max_length <= 0:
        raise ValueError("max_length must be positive.")
    if not 0.0 <= dropout_rate < 1.0:
        raise ValueError("dropout_rate must be in [0.0, 1.0).")


class TransformerEncoderShapeValidationMixin:
    """Mixin that validates shared transformer encoder dimensions."""

    hidden_dim: int
    num_layers: int
    num_heads: int
    intermediate_dim: int
    max_length: int
    dropout_rate: float

    def __post_init__(self) -> None:
        """Run base config validation and shared transformer checks."""
        super().__post_init__()
        validate_transformer_encoder_shape(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            intermediate_dim=self.intermediate_dim,
            max_length=self.max_length,
            dropout_rate=self.dropout_rate,
        )
