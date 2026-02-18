#!/bin/bash
# DiffBio Development Environment Setup
# Consolidated script for creating, building, and activating venv for both CPU and GPU development

set -e  # Exit on any error

# Default values
DEEP_CLEAN=false
CPU_ONLY=false
HELP=false
VERBOSE=false
FORCE_REINSTALL=false

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --deep-clean)
            DEEP_CLEAN=true
            shift
            ;;
        --cpu-only)
            CPU_ONLY=true
            shift
            ;;
        --force)
            FORCE_REINSTALL=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --help|-h)
            HELP=true
            shift
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Show help if requested
if [ "$HELP" = true ]; then
    cat << 'EOF'
DiffBio Development Environment Setup
======================================

Creates, builds, and prepares the virtual environment for DiffBio development
with automatic GPU/CPU detection and optimal configuration.

USAGE:
    ./setup.sh [OPTIONS]

OPTIONS:
    --deep-clean     Perform comprehensive cleaning (JAX cache, pip cache, etc.)
    --cpu-only       Force CPU-only setup (skip GPU detection)
    --force          Force reinstallation even if environment exists
    --verbose, -v    Show detailed output during setup
    --help, -h       Show this help message

EXAMPLES:
    ./setup.sh                    # Standard setup with auto GPU detection
    ./setup.sh --deep-clean       # Clean setup with cache clearing
    ./setup.sh --cpu-only         # Force CPU-only development setup
    ./setup.sh --force --verbose  # Verbose forced reinstallation

ACTIVATION:
    After setup, activate the environment with:
    source ./activate.sh

FILES CREATED:
    .venv/           Virtual environment directory
    .env             Environment variables and CUDA configuration
    activate.sh      Unified activation script
    uv.lock          Dependency lock file

REQUIREMENTS:
    - uv package manager (installed automatically if missing)
    - Python 3.11+ (handled by uv)
    - NVIDIA drivers (for GPU support)

EOF
    exit 0
fi

# Utility functions
log_info() {
    echo -e "${BLUE}$1${NC}"
}

log_success() {
    echo -e "${GREEN}$1${NC}"
}

log_warning() {
    echo -e "${YELLOW}$1${NC}"
}

log_error() {
    echo -e "${RED}$1${NC}"
}

log_step() {
    echo -e "${PURPLE}$1${NC}"
}

verbose_log() {
    if [ "$VERBOSE" = true ]; then
        echo -e "${CYAN}   -> $1${NC}"
    fi
}

# Function to check and install uv if needed
ensure_uv_installed() {
    if ! command -v uv &> /dev/null; then
        log_warning "uv not found. Installing uv package manager..."
        curl -LsSf https://astral.sh/uv/install.sh | sh

        # Add uv to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"

        if ! command -v uv &> /dev/null; then
            log_error "Failed to install uv. Please install manually:"
            echo "curl -LsSf https://astral.sh/uv/install.sh | sh"
            exit 1
        fi
        log_success "uv installed successfully"
    else
        verbose_log "uv already installed: $(uv --version)"
    fi
}

# Function to detect CUDA availability
detect_cuda_support() {
    if [ "$CPU_ONLY" = true ]; then
        log_info "CPU-only mode requested, skipping GPU detection"
        return 1
    fi

    if command -v nvidia-smi &> /dev/null; then
        local gpu_info
        if gpu_info=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1) && [ -n "$gpu_info" ]; then
            log_success "NVIDIA GPU detected: $gpu_info"

            # Check CUDA installation
            if [ -d "/usr/local/cuda" ] || [ -n "$CUDA_HOME" ]; then
                verbose_log "CUDA installation found"
                return 0
            else
                log_warning "GPU detected but CUDA not found in standard locations"
                log_info "Will attempt GPU setup anyway"
                return 0
            fi
        fi
    fi

    log_info "No NVIDIA GPU detected - setting up CPU-only environment"
    return 1
}

# Function to perform cleaning
perform_cleaning() {
    log_step "Cleaning existing environment..."

    # Remove virtual environment
    if [ -d ".venv" ]; then
        verbose_log "Removing virtual environment (.venv)"
        rm -rf .venv
    fi

    # Remove lock files if force reinstall
    if [ "$FORCE_REINSTALL" = true ] && [ -f "uv.lock" ]; then
        verbose_log "Removing lock file (uv.lock)"
        rm -f uv.lock
    fi

    # Clean uv cache to avoid package conflicts
    verbose_log "Cleaning uv cache"
    uv cache clean 2>/dev/null || true

    # Remove existing environment files
    if [ -f ".env" ]; then
        verbose_log "Removing existing .env file"
        rm -f .env
    fi

    # Clean Python cache files
    verbose_log "Cleaning Python cache files"
    find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
    find . -name "*.pyc" -delete 2>/dev/null || true
    find . -name "*.pyo" -delete 2>/dev/null || true

    # Deep cleaning if requested
    if [ "$DEEP_CLEAN" = true ]; then
        log_step "Performing deep cleaning..."

        # Clean JAX compilation cache
        if [ -d "$HOME/.cache/jax" ]; then
            verbose_log "Removing JAX compilation cache"
            rm -rf "$HOME/.cache/jax"
        fi

        # Clean pip cache
        verbose_log "Cleaning pip cache"
        python -m pip cache purge 2>/dev/null || pip cache purge 2>/dev/null || true

        # Clean pytest cache
        if [ -d ".pytest_cache" ]; then
            verbose_log "Removing pytest cache"
            rm -rf .pytest_cache
        fi

        # Clean coverage files
        for file in .coverage .coverage.*; do
            if [ -f "$file" ]; then
                verbose_log "Removing coverage file: $file"
                rm -f "$file"
            fi
        done
        if [ -d "htmlcov" ]; then
            verbose_log "Removing HTML coverage directory"
            rm -rf htmlcov
        fi

        # Clean temporary files
        verbose_log "Cleaning temporary files"
        find . -name "tmp*" -type d -exec rm -rf {} + 2>/dev/null || true
        find . -name ".tmp*" -type f -delete 2>/dev/null || true

        # Clean temp directory
        if [ -d "temp" ]; then
            verbose_log "Cleaning temp directory"
            rm -rf temp/*
        fi
    fi

    log_success "Environment cleaned"
}

# Function to create .env file
create_env_file() {
    local has_cuda=$1

    log_step "Creating environment configuration..."

    if [ "$has_cuda" = true ]; then
        # Check if .env.example exists and use it
        if [ -f ".env.example" ]; then
            # Copy template and replace $(pwd) with actual absolute path
            sed "s|PROJECT_DIR=\"\$(pwd)\"|PROJECT_DIR=\"$(pwd)\"|g" .env.example > .env
            verbose_log "Created GPU-enabled .env configuration from template"
        else
            # Fallback to embedded template if .env.example is missing
            cat > .env << 'EOF'
# DiffBio Environment Configuration - GPU Enabled
# Auto-generated by setup script

# CUDA Library Configuration - Use local venv CUDA installation
PROJECT_DIR="$(pwd)"

# Dynamically detect Python version
if [ -f "${PROJECT_DIR}/.venv/bin/python" ]; then
    PYTHON_VERSION=$("${PROJECT_DIR}/.venv/bin/python" -c "import sys; print(f'python{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
elif [ -d "${PROJECT_DIR}/.venv/lib/python3.12" ]; then
    PYTHON_VERSION="python3.12"
elif [ -d "${PROJECT_DIR}/.venv/lib/python3.11" ]; then
    PYTHON_VERSION="python3.11"
elif [ -d "${PROJECT_DIR}/.venv/lib/python3.10" ]; then
    PYTHON_VERSION="python3.10"
else
    PYTHON_VERSION=$(ls -d "${PROJECT_DIR}/.venv/lib/python3."* 2>/dev/null | head -1 | xargs basename 2>/dev/null || echo "python3.11")
fi
VENV_CUDA_BASE="${PROJECT_DIR}/.venv/lib/${PYTHON_VERSION}/site-packages/nvidia"

# Filter out old CUDA paths from existing LD_LIBRARY_PATH
if [ -n "$LD_LIBRARY_PATH" ]; then
    FILTERED_LD_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v -E '(nvidia|cuda|cudnn|nccl|cublas|cusolver|cusparse|cufft|curand|nvjitlink)' | tr '\n' ':' | sed 's/:$//')
else
    FILTERED_LD_PATH=""
fi

# Set new CUDA paths
NEW_CUDA_PATHS="${VENV_CUDA_BASE}/cublas/lib:${VENV_CUDA_BASE}/cusolver/lib:${VENV_CUDA_BASE}/cusparse/lib:${VENV_CUDA_BASE}/cusparselt/lib:${VENV_CUDA_BASE}/cudnn/lib:${VENV_CUDA_BASE}/cufft/lib:${VENV_CUDA_BASE}/curand/lib:${VENV_CUDA_BASE}/cufile/lib:${VENV_CUDA_BASE}/nccl/lib:${VENV_CUDA_BASE}/nvjitlink/lib:${VENV_CUDA_BASE}/cuda_runtime/lib:${VENV_CUDA_BASE}/cuda_nvrtc/lib:${VENV_CUDA_BASE}/cuda_cupti/lib:${VENV_CUDA_BASE}/nvtx/lib:${VENV_CUDA_BASE}/cuda_nvcc/bin"

if [ -n "$FILTERED_LD_PATH" ]; then
    export LD_LIBRARY_PATH="${NEW_CUDA_PATHS}:${FILTERED_LD_PATH}"
else
    export LD_LIBRARY_PATH="${NEW_CUDA_PATHS}"
fi

# Set CUDA_HOME
export CUDA_HOME="${VENV_CUDA_BASE}"
export CUDA_PATH="${VENV_CUDA_BASE}"
export PATH="${VENV_CUDA_BASE}/cuda_nvcc/bin:${PATH}"

# Force CUDA to use venv libraries only
export CUDA_CACHE_DISABLE="1"
export CUDA_MODULE_LOADING="LAZY"

# === CUDA Performance Settings ===
export CUDA_MODULE_LOADING="LAZY"       # Faster startup, reduced memory
export CUDA_CACHE_DISABLE="1"           # Use JAX cache instead

# === JAX Platform Configuration ===
export JAX_PLATFORMS="cuda,cpu"
export JAX_ENABLE_X64="0"               # Keep 32-bit for performance

# === JAX Memory Management ===
export XLA_PYTHON_CLIENT_PREALLOCATE="false"
export XLA_PYTHON_CLIENT_MEM_FRACTION="0.85"  # Best practice: room for lazy-loaded CUDA kernels

# === JAX Compilation Cache (LOCAL - optimal for single-machine dev) ===
export JAX_COMPILATION_CACHE_DIR="${PROJECT_DIR}/.cache/jax"
export XLA_CACHE_DIR="${PROJECT_DIR}/.cache/xla"

# === XLA Performance Flags (AMD/NVIDIA compatible - no Triton flags) ===
# Note: Async collectives are enabled by default in JAX 0.9+
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_enable_latency_hiding_scheduler=true"

# === JAX CUDA Plugin ===
export JAX_CUDA_PLUGIN_VERIFY="false"

# === Logging (reduce noise) ===
export TF_CPP_MIN_LOG_LEVEL="1"

# === Development Settings ===
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${PROJECT_DIR}"
export PYTEST_CUDA_ENABLED="true"
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"
EOF
            verbose_log "Created GPU-enabled .env configuration from embedded template"
        fi
    else
        cat > .env << 'EOF'
# DiffBio Environment Configuration - CPU Only
# Auto-generated by setup script

PROJECT_DIR="$(pwd)"

# JAX Configuration for CPU
export JAX_PLATFORMS="cpu"
export JAX_ENABLE_X64="0"

# Development settings
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${PROJECT_DIR}"

# Testing configuration
export PYTEST_CUDA_ENABLED="false"

# Performance settings
export TF_CPP_MIN_LOG_LEVEL="1"

# Protobuf Configuration
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"
EOF
        verbose_log "Created CPU-only .env configuration"
    fi

    log_success "Environment configuration created"
}

# Function to create unified activation script
create_activation_script() {
    log_step "Creating unified activation script..."

    cat > activate.sh << 'EOF'
#!/bin/bash
# DiffBio Environment Activation Script
# Created by setup script

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${BLUE}Activating DiffBio Development Environment${NC}"
echo "============================================"

# Check if already activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo -e "${YELLOW}Virtual environment already active: $VIRTUAL_ENV${NC}"
    echo "Deactivating current environment..."

    # Check for processes using the virtual environment
    VENV_PROCESSES=$(pgrep -f "$VIRTUAL_ENV" | xargs -I {} ps -p {} -o pid,etime,args --no-headers 2>/dev/null || true)

    if [[ -n "$VENV_PROCESSES" ]]; then
        echo -e "${YELLOW}Checking for processes using the virtual environment...${NC}"
        PROCESS_COUNT=$(echo "$VENV_PROCESSES" | wc -l)
        if [[ $PROCESS_COUNT -gt 0 ]]; then
            echo -e "${YELLOW}Found $PROCESS_COUNT process(es) using the virtual environment:${NC}"
            echo ""
            echo "$VENV_PROCESSES" | while IFS= read -r line; do
                if [[ -n "$line" ]]; then
                    PID=$(echo "$line" | awk '{print $1}')
                    ETIME=$(echo "$line" | awk '{print $2}')
                    CMD=$(echo "$line" | awk '{for(i=3;i<=NF;i++) printf "%s ", $i; print ""}' | sed 's/[[:space:]]*$//')
                    echo -e "${CYAN}   PID $PID (running for $ETIME): ${NC}$CMD"
                fi
            done
            echo ""
            echo -e "${YELLOW}Options:${NC}"
            echo -e "${CYAN}   1. Wait for processes to complete naturally${NC}"
            echo -e "${CYAN}   2. Press Ctrl+C to cancel activation${NC}"
            echo -e "${CYAN}   3. In another terminal, stop processes manually:${NC}"
            echo -e "${CYAN}      pkill -f pytest  # Stop test processes${NC}"
            echo ""
        fi
    fi

    # Attempt deactivation
    deactivate 2>/dev/null || true
fi

# Activate virtual environment
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
    echo -e "${GREEN}Virtual environment activated${NC}"
else
    echo -e "${RED}Virtual environment not found!${NC}"
    echo "Run './setup.sh' to create the environment first."
    exit 1
fi

# Load environment variables
if [ -f .env ]; then
    source .env
    echo -e "${GREEN}Environment configuration loaded${NC}"

    # Show configuration based on JAX_PLATFORMS
    if [[ "$JAX_PLATFORMS" == *"cuda"* ]]; then
        echo -e "${CYAN}   GPU Mode: CUDA enabled${NC}"
        echo -e "${CYAN}   CUDA_HOME: ${CUDA_HOME:-not set}${NC}"
    else
        echo -e "${CYAN}   CPU Mode: CPU-only configuration${NC}"
    fi
else
    echo -e "${YELLOW}.env file not found - using minimal setup${NC}"
    export JAX_PLATFORMS="cpu"
fi

# Display system information
echo ""
echo -e "${BLUE}Environment Status:${NC}"
echo -e "${CYAN}   Python: $(python --version)${NC}"
echo -e "${CYAN}   Working Directory: $(pwd)${NC}"
echo -e "${CYAN}   Virtual Environment: $VIRTUAL_ENV${NC}"

# Check JAX installation
echo ""
echo -e "${BLUE}JAX Configuration:${NC}"

python << 'PYTHON_EOF'
try:
    import jax
    import jax.numpy as jnp

    print(f"   JAX version: {jax.__version__}")
    print(f"   Default backend: {jax.default_backend()}")

    devices = jax.devices()
    print(f"   Available devices: {len(devices)} total")

    gpu_devices = [d for d in devices if d.platform == 'gpu']
    cpu_devices = [d for d in devices if d.platform == 'cpu']

    if gpu_devices:
        print(f"   GPU devices: {len(gpu_devices)} ({[str(d) for d in gpu_devices]})")
        print("   CUDA acceleration ready!")

        try:
            x = jnp.array([1., 2., 3.])
            y = jnp.sum(x**2)
            print(f"   GPU test successful: {float(y)}")
        except Exception as e:
            print(f"   GPU test warning: {e}")
    else:
        print(f"   CPU devices: {len(cpu_devices)} ({[str(d) for d in cpu_devices]})")
        print("   Running in CPU-only mode")

    try:
        x = jnp.linspace(0, 1, 100)
        y = jnp.sin(2 * jnp.pi * x)
        print(f"   JAX functionality verified")
    except Exception as e:
        print(f"   JAX functionality test failed: {e}")

except ImportError as e:
    print(f"   JAX not installed properly: {e}")
    print("   Run './setup.sh' to reinstall dependencies")
except Exception as e:
    print(f"   JAX configuration issue: {e}")
PYTHON_EOF

# Display usage information
echo ""
echo -e "${BLUE}Ready for Development!${NC}"
echo "======================"
echo ""
echo -e "${GREEN}Common Commands:${NC}"
echo -e "${CYAN}   uv run pytest tests/ -v                    ${NC}# Run all tests"
echo -e "${CYAN}   uv run pytest tests/operators/ -v          ${NC}# Run operator tests"
echo -e "${CYAN}   uv run pytest tests/pipelines/ -v          ${NC}# Run pipeline tests"
echo -e "${CYAN}   uv run python your_script.py               ${NC}# Run your code"
echo ""
echo -e "${GREEN}Development Tools:${NC}"
echo -e "${CYAN}   uv add package_name                        ${NC}# Add new dependency"
echo -e "${CYAN}   uv run pre-commit run --all-files          ${NC}# Run code quality checks"
echo -e "${CYAN}   uv run pytest --cov=src/diffbio tests/     ${NC}# Run tests with coverage"
echo ""
echo -e "${GREEN}Documentation:${NC}"
echo -e "${CYAN}   uv run mkdocs serve                        ${NC}# Serve docs locally"
echo -e "${CYAN}   uv run mkdocs build                        ${NC}# Build documentation"
echo ""
echo -e "${YELLOW}To deactivate: ${NC}deactivate"
EOF

    chmod +x activate.sh
    log_success "Unified activation script created: ./activate.sh"
}

# Function to create virtual environment and install dependencies
setup_environment() {
    local has_cuda=$1

    log_step "Creating virtual environment..."
    uv venv

    # Create cache directories for JAX compilation
    verbose_log "Creating JAX compilation cache directories"
    mkdir -p .cache/jax .cache/xla

    # Activate the environment for installation
    source .venv/bin/activate
    source .env

    log_step "Installing dependencies..."

    if [ "$has_cuda" = true ]; then
        log_info "Installing with CUDA support..."
        verbose_log "Installing complete package with all dependencies (dev, docs, gpu, test)"

        # Install complete package with all dependency groups
        if ! uv sync --extra all 2>/dev/null; then
            log_warning "Full installation with GPU failed, falling back to CPU-only"
            uv sync --extra dev
            has_cuda=false
            # Update .env file for CPU-only
            create_env_file false
        else
            log_success "Complete installation with GPU support successful"

            # Ensure matching JAX CUDA plugin versions
            log_info "Ensuring correct JAX CUDA plugin versions..."
            JAX_VERSION=$(python -c "import jax; print(jax.__version__)" 2>/dev/null || echo "0.6.1")

            # Install matching CUDA plugins for JAX
            verbose_log "Installing JAX CUDA plugins version $JAX_VERSION"
            uv pip install --force-reinstall "jax-cuda12-pjrt==$JAX_VERSION" "jax-cuda12-plugin==$JAX_VERSION" 2>/dev/null || true
        fi
    else
        log_info "Installing CPU-only version with dev dependencies..."
        uv sync --extra dev
    fi

    log_success "Dependencies installed successfully"
    return 0
}

# Function to verify installation
verify_installation() {
    local has_cuda=$1

    log_step "Verifying installation..."

    # Test JAX installation
    python << PYTHON_EOF
import sys
import traceback

try:
    import jax
    import jax.numpy as jnp
    import flax
    import optax

    print(f"Core dependencies verified:")
    print(f"   JAX: {jax.__version__}")
    print(f"   Flax: {flax.__version__}")
    print(f"   Optax: {optax.__version__}")

    # Test basic functionality
    x = jnp.array([1.0, 2.0, 3.0])
    y = jnp.sum(x**2)
    print(f"Basic computation test: {float(y)}")

    # Test devices
    devices = jax.devices()
    gpu_devices = [d for d in devices if d.platform == 'gpu']

    print(f"Available devices: {len(devices)} total")
    if gpu_devices:
        print(f"GPU devices detected: {len(gpu_devices)}")
        try:
            with jax.default_device(gpu_devices[0]):
                z = jnp.array([1., 2., 3.])
                w = jnp.dot(z, z)
            print(f"GPU computation test: {float(w)}")
        except Exception as e:
            print(f"GPU test warning: {e}")
    else:
        print("No GPU devices (CPU-only mode)")

    # Test DiffBio import
    try:
        import diffbio
        print(f"DiffBio import successful")
    except ImportError as e:
        print(f"DiffBio import warning: {e}")

    print("Installation verification complete!")

except ImportError as e:
    print(f"Import error: {e}")
    print("Installation may be incomplete")
    sys.exit(1)
except Exception as e:
    print(f"Verification error: {e}")
    traceback.print_exc()
    sys.exit(1)
PYTHON_EOF

    local verify_status=$?
    if [ $verify_status -eq 0 ]; then
        log_success "Installation verified successfully"
        return 0
    else
        log_error "Installation verification failed"
        return 1
    fi
}

# Function to display setup summary
display_summary() {
    local has_cuda=$1

    echo ""
    echo -e "${GREEN}DiffBio Development Environment Setup Complete!${NC}"
    echo "================================================"
    echo ""
    echo -e "${BLUE}Files Created:${NC}"
    echo -e "${CYAN}   .venv/                 Virtual environment${NC}"
    echo -e "${CYAN}   .env                   Environment configuration${NC}"
    echo -e "${CYAN}   activate.sh            Unified activation script${NC}"
    echo -e "${CYAN}   uv.lock                Dependency lock file${NC}"
    echo ""
    echo -e "${BLUE}Quick Start:${NC}"
    echo -e "${YELLOW}   source ./activate.sh   ${NC}# Activate environment (use 'source'!)"
    echo -e "${CYAN}   uv run pytest tests/   ${NC}# Run tests to verify setup"
    echo ""

    if [ "$has_cuda" = true ]; then
        echo -e "${GREEN}GPU Support: CUDA Enabled${NC}"
        echo "   Your environment is ready for GPU-accelerated development!"
    else
        echo -e "${BLUE}GPU Support: CPU-Only Mode${NC}"
        echo "   For GPU support, ensure NVIDIA drivers and CUDA are installed,"
        echo "   then re-run with: ./setup.sh --force"
    fi
    echo ""
    echo -e "${PURPLE}For more information, see README.md${NC}"
}

# Main execution function
main() {
    echo -e "${PURPLE}DiffBio Development Environment Setup${NC}"
    echo "======================================"
    echo ""

    # Pre-flight checks
    ensure_uv_installed

    # Detect GPU capability
    HAS_CUDA=false
    if detect_cuda_support; then
        HAS_CUDA=true
    fi

    # Check if environment already exists and handle appropriately
    if [ -d ".venv" ] && [ "$FORCE_REINSTALL" != true ]; then
        log_warning "Virtual environment already exists"
        echo "Use --force to reinstall or source ./activate.sh to use existing environment"
        exit 1
    fi

    # Perform cleanup
    perform_cleaning

    # Create configuration files
    create_env_file "$HAS_CUDA"
    create_activation_script

    # Setup environment and install dependencies
    if ! setup_environment "$HAS_CUDA"; then
        log_error "Failed to setup environment"
        exit 1
    fi

    # Verify installation works
    if ! verify_installation "$HAS_CUDA"; then
        log_error "Installation verification failed"
        exit 1
    fi

    # Show summary
    display_summary "$HAS_CUDA"
}

# Run main function with all arguments
main "$@"
