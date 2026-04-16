"""Tests for the shared count-VAE operator substrate."""

from typing import Any

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

import diffbio.operators._count_vae as count_vae


class _ReconstructionHarness(count_vae.CountReconstructionMixin):
    """Minimal harness exposing the shared reconstruction helpers."""


class TestCountVAEBackbone:
    """Tests for the shared encoder/decoder backbone."""

    def test_encode_returns_latent_statistics(self) -> None:
        """Encoding produces mean/logvar vectors with the requested latent size."""
        backbone = count_vae.CountVAEBackbone(
            n_inputs=6,
            latent_dim=3,
            hidden_dims=[5, 4],
            n_outputs=6,
            rngs=nnx.Rngs(0),
        )

        mean, logvar = backbone.encode(jnp.ones((6,)))

        assert mean.shape == (3,)
        assert logvar.shape == (3,)

    def test_encode_clips_logvar_for_numerical_stability(self) -> None:
        """Latent log-variance is clamped into the stable range."""
        backbone = count_vae.CountVAEBackbone(
            n_inputs=4,
            latent_dim=2,
            hidden_dims=[3],
            n_outputs=4,
            rngs=nnx.Rngs(0),
        )
        backbone.fc_logvar.bias = nnx.Param(jnp.array([50.0, -50.0]))

        _, logvar = backbone.encode(jnp.ones((4,)))

        assert jnp.all(logvar <= 10.0)
        assert jnp.all(logvar >= -10.0)

    def test_decode_hidden_uses_first_decoder_width(self) -> None:
        """Decoder hidden activations use the first hidden width after reversal."""
        backbone = count_vae.CountVAEBackbone(
            n_inputs=7,
            latent_dim=3,
            hidden_dims=[6, 5],
            n_outputs=7,
            rngs=nnx.Rngs(0),
        )

        hidden = backbone.decode_hidden(jnp.ones((3,)))

        assert hidden.shape == (6,)

    def test_decode_rates_returns_output_dimensionality(self) -> None:
        """Rate decoding returns one value per output feature."""
        backbone = count_vae.CountVAEBackbone(
            n_inputs=5,
            latent_dim=2,
            hidden_dims=[4],
            n_outputs=8,
            rngs=nnx.Rngs(0),
        )

        log_rate = backbone.decode_rates(jnp.ones((2,)))

        assert log_rate.shape == (8,)


class TestCountReconstructionMixin:
    """Tests for the shared count reconstruction helpers."""

    def test_poisson_reconstruction_path(self) -> None:
        """Poisson reconstruction uses only the decoded rate."""
        harness = _ReconstructionHarness()

        loss = harness.reconstruction_loss(
            counts=jnp.array([1.0, 2.0]),
            decode_output={"log_rate": jnp.array([0.0, 0.1])},
        )

        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_zinb_reconstruction_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ZINB reconstruction delegates to the shared ZINB loss."""
        harness = _ReconstructionHarness()
        calls: dict[str, Any] = {}

        def _fake_zinb_nll(
            counts: jax.Array,
            log_rate: jax.Array,
            log_theta: jax.Array,
            pi_logit: jax.Array,
        ) -> jax.Array:
            calls["counts"] = counts
            calls["log_rate"] = log_rate
            calls["log_theta"] = log_theta
            calls["pi_logit"] = pi_logit
            return jnp.array(5.0)

        monkeypatch.setattr(harness, "_zinb_nll", _fake_zinb_nll)

        loss = harness.reconstruction_loss(
            counts=jnp.array([1.0, 0.0]),
            decode_output={
                "log_rate": jnp.array([0.0, 0.1]),
                "log_theta": jnp.array([0.2, 0.3]),
                "pi_logit": jnp.array([0.4, 0.5]),
            },
        )

        assert jnp.allclose(loss, 5.0)
        assert set(calls) == {"counts", "log_rate", "log_theta", "pi_logit"}
