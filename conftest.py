# conftest.py - Pytest configuration for DiffBio

import logging
import os
import warnings
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)


def setup_jax_environment():
    """Set up JAX environment variables with safe CPU fallback."""
    cuda_home = os.environ.get("CUDA_HOME", "/usr/local/cuda")
    cuda_lib_path = str(Path(cuda_home) / "lib64")
    current_ld_path = os.environ.get("LD_LIBRARY_PATH", "")

    if Path(cuda_lib_path).exists() and cuda_lib_path not in current_ld_path:
        if current_ld_path:
            new_ld_path = f"{cuda_lib_path}:{current_ld_path}"
        else:
            new_ld_path = cuda_lib_path
        os.environ["LD_LIBRARY_PATH"] = new_ld_path

    # Set additional CUDA environment variables
    os.environ["CUDA_ROOT"] = cuda_home
    os.environ["CUDA_HOME"] = cuda_home

    # Respect explicit platform settings (e.g., from activate.sh/.env).
    # Default to CPU-only so tests can run on machines without CUDA setup.
    if "JAX_PLATFORMS" not in os.environ:
        os.environ["JAX_PLATFORMS"] = "cpu"

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"

    # Disable CUDA plugin validation to bypass cuSPARSE check
    os.environ["JAX_CUDA_PLUGIN_VERIFY"] = "false"
    os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false"


def pytest_configure(config):
    """Configure pytest environment with proper JAX/CUDA handling."""
    # Configure environment before any JAX imports
    setup_jax_environment()

    # Suppress CUDA warnings and errors in test output
    warnings.filterwarnings("ignore", category=UserWarning, module="jax._src.xla_bridge")
    warnings.filterwarnings("ignore", message=".*cuSPARSE.*")
    warnings.filterwarnings("ignore", message=".*CUDA-enabled jaxlib.*")

    # Import JAX and configure after environment setup
    try:
        import jax

        # Force JAX to initialize with current environment
        # Respect the JAX_PLATFORMS environment variable
        current_platforms = os.environ.get("JAX_PLATFORMS", "cpu")
        jax.config.update("jax_platforms", current_platforms)

        # Check if CUDA is available
        try:
            devices = jax.devices()
            gpu_devices = [d for d in devices if d.platform == "gpu"]
            cpu_devices = [d for d in devices if d.platform == "cpu"]

            config.addinivalue_line(
                "markers", f"gpu_available: GPU devices available: {len(gpu_devices) > 0}"
            )
            config.addinivalue_line("markers", f"devices: Available devices: {devices}")

            logger.info("GPU available for testing: %s", len(gpu_devices) > 0)
            if gpu_devices:
                logger.info("GPU devices: %s", gpu_devices)
            logger.info("CPU devices: %s", cpu_devices)

        except RuntimeError as e:
            logger.warning("Device detection failed: %s", e)
            config.addinivalue_line("markers", "gpu_available: GPU devices available: False")

    except ImportError as e:
        logger.warning("JAX import failed: %s", e)


@pytest.fixture
def random_key():
    """Provide a JAX random key."""
    import jax

    return jax.random.PRNGKey(42)


@pytest.fixture
def sample_sequences(random_key):
    """Generate sample one-hot encoded sequences for testing."""
    import jax

    k1, k2 = jax.random.split(random_key)

    # Create two sample sequences (one-hot encoded, 4 bases: A, C, G, T)
    seq1_indices = jax.random.randint(k1, (50,), 0, 4)
    seq2_indices = jax.random.randint(k2, (60,), 0, 4)

    return {
        "seq1": jax.nn.one_hot(seq1_indices, 4),
        "seq2": jax.nn.one_hot(seq2_indices, 4),
    }


@pytest.fixture
def sample_reads(random_key):
    """Generate sample read data for pileup testing."""
    import jax

    num_reads = 10
    read_length = 30
    k1, k2, k3 = jax.random.split(random_key, 3)

    return {
        "reads": jax.nn.one_hot(jax.random.randint(k1, (num_reads, read_length), 0, 4), 4),
        "positions": jax.random.randint(k2, (num_reads,), 0, 70),
        "quality": jax.random.uniform(k3, (num_reads, read_length), minval=10.0, maxval=40.0),
    }


def pytest_runtest_setup(item):
    """Set up for each test run - ensure clean JAX state."""
    # Clear any JAX compilation cache to avoid issues
    try:
        import jax

        jax.clear_caches()

        # If this is a GPU test, ensure GPU memory is available
        if hasattr(item, "get_closest_marker"):
            gpu_marker = item.get_closest_marker("gpu")
            cuda_marker = item.get_closest_marker("cuda")

            if gpu_marker or cuda_marker:
                # Force garbage collection before GPU tests
                import gc

                gc.collect()

                # Verify GPU is available for GPU-marked tests
                try:
                    gpu_devices = jax.devices("gpu")
                    if not gpu_devices:
                        pytest.skip("GPU test skipped: No GPU devices available")
                except RuntimeError:
                    pytest.skip("GPU test skipped: GPU not accessible")

    except ImportError:
        pass


def pytest_runtest_teardown(item, nextitem):  # noqa: ARG001
    """Teardown after each test run - clean up GPU memory."""
    try:
        import jax

        # Clear JAX caches after each test
        jax.clear_caches()

        # Force garbage collection to free GPU memory
        import gc

        gc.collect()

        # If this was a GPU test, ensure memory cleanup
        if hasattr(item, "get_closest_marker"):
            gpu_marker = item.get_closest_marker("gpu")
            cuda_marker = item.get_closest_marker("cuda")

            if gpu_marker or cuda_marker:
                try:
                    # Additional GPU memory cleanup
                    import jax.numpy as jnp

                    # Force any pending operations to complete
                    dummy = jnp.array([1.0])
                    dummy.block_until_ready()
                    del dummy
                except RuntimeError:
                    pass

    except ImportError:
        pass
