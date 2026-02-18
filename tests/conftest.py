"""Test configuration for DiffBio."""

import logging
import os

# Configure JAX before imports
os.environ.setdefault("JAX_ENABLE_X64", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")

import jax
import pytest
from flax import nnx

logger = logging.getLogger(__name__)


# Configure beartype for runtime type checking
try:
    from beartype import BeartypeConf, BeartypeStrategy
    import beartype

    try:
        beartype.beartype(conf=BeartypeConf(strategy=BeartypeStrategy.On))
    except (TypeError, AttributeError) as exc:
        logger.warning("Could not apply beartype configuration: %s", exc)
except ImportError:
    pass


def pytest_configure(config):
    """Register custom markers for pytest."""
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "e2e: mark test as an end-to-end test")
    config.addinivalue_line("markers", "benchmark: mark test as a performance benchmark")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "slow: mark test as slow running")


def pytest_addoption(parser):
    """Add command-line options to pytest."""
    parser.addoption(
        "--integration",
        action="store_true",
        default=False,
        help="run integration tests",
    )
    parser.addoption(
        "--device",
        choices=["cpu", "gpu", "all"],
        default="all",
        help="select device type for tests (cpu, gpu, or all)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests based on command-line options and GPU availability."""
    device_option = config.getoption("--device")

    # Check GPU availability
    try:
        gpu_available = len(jax.devices("gpu")) > 0
    except RuntimeError:
        gpu_available = False

    skip_gpu = pytest.mark.skip(reason="GPU not available")
    skip_cpu_only = pytest.mark.skip(reason="running GPU tests only")

    for item in items:
        # Skip GPU tests if GPU not available or CPU-only mode
        if "gpu" in item.keywords:
            if not gpu_available or device_option == "cpu":
                item.add_marker(skip_gpu)

        # Skip CPU tests if GPU-only mode
        if device_option == "gpu" and "cpu" in item.keywords:
            item.add_marker(skip_cpu_only)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def rngs() -> nnx.Rngs:
    """Provide Flax NNX random number generators with fixed seed for reproducibility."""
    return nnx.Rngs(42)


@pytest.fixture
def jax_key() -> jax.Array:
    """Provide a JAX PRNG key with fixed seed for reproducibility."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def device():
    """Provide the default JAX device (CPU for consistent testing)."""
    return jax.devices("cpu")[0]


@pytest.fixture
def sample_dna_sequence() -> jax.Array:
    """Provide a sample one-hot encoded DNA sequence.

    Returns:
        Array of shape (100, 4) representing a 100bp DNA sequence.
        Columns represent A, C, G, T.
    """
    key = jax.random.PRNGKey(42)
    # Generate random one-hot vectors (100 nucleotides, 4 possible bases)
    indices = jax.random.randint(key, (100,), 0, 4)
    return jax.nn.one_hot(indices, 4)


@pytest.fixture
def sample_quality_scores() -> jax.Array:
    """Provide sample Phred quality scores.

    Returns:
        Array of shape (100,) with quality scores in range [0, 40].
    """
    key = jax.random.PRNGKey(43)
    return jax.random.uniform(key, (100,), minval=0.0, maxval=40.0)


@pytest.fixture
def sample_protein_sequence() -> jax.Array:
    """Provide a sample one-hot encoded protein sequence.

    Returns:
        Array of shape (50, 20) representing a 50aa protein sequence.
        Columns represent the 20 standard amino acids.
    """
    key = jax.random.PRNGKey(44)
    indices = jax.random.randint(key, (50,), 0, 20)
    return jax.nn.one_hot(indices, 20)


@pytest.fixture
def batch_size() -> int:
    """Provide standard batch size for tests."""
    return 32


@pytest.fixture
def sequence_length() -> int:
    """Provide standard sequence length for tests."""
    return 100
