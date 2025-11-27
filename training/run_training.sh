#!/bin/bash
# =============================================================================
# DeepSeek OCR Training Pipeline for Turkish E-Invoice Extraction
# =============================================================================
# This script runs the complete training pipeline from scratch:
# 1. Prepares the dataset from JSON annotations and images
# 2. Fine-tunes the DeepSeek OCR model
# 3. Runs inference on test images
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Configuration
# =============================================================================

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_DIR}/output"
IMAGES_DIR="${PROJECT_DIR}/images"
DATASET_DIR="${PROJECT_DIR}/dataset"
CHECKPOINTS_DIR="${PROJECT_DIR}/checkpoints"
MODEL_DIR="${PROJECT_DIR}/deepseek_ocr_code"
TRAINING_DIR="${PROJECT_DIR}/training"
LOGS_DIR="${PROJECT_DIR}/logs"

# Training configuration
CONFIG_FILE="${TRAINING_DIR}/config.json"

# =============================================================================
# Functions
# =============================================================================

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_requirements() {
    log_info "Checking requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check CUDA
    if python3 -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
        log_success "CUDA is available"
        DEVICE="cuda"
    else
        log_warning "CUDA is not available, using CPU (training will be slow)"
        DEVICE="cpu"
    fi
    
    # Check for model files
    if [ ! -d "$MODEL_DIR" ]; then
        log_error "Model directory not found: $MODEL_DIR"
        exit 1
    fi
    
    log_success "Requirements check passed"
}

install_dependencies() {
    log_info "Installing dependencies..."
    
    cd "$PROJECT_DIR"
    
    # Install requirements
    pip install -r "${TRAINING_DIR}/requirements.txt"
    
    log_success "Dependencies installed"
}

prepare_dataset() {
    log_info "Preparing dataset..."
    
    # Create dataset directory
    mkdir -p "$DATASET_DIR"
    
    # Run dataset preparation
    python3 "${TRAINING_DIR}/prepare_dataset.py" \
        --output-dir "$OUTPUT_DIR" \
        --images-dir "$IMAGES_DIR" \
        --dataset-dir "$DATASET_DIR" \
        --train-ratio 0.85 \
        --val-ratio 0.10 \
        --test-ratio 0.05 \
        --seed 42
    
    log_success "Dataset prepared at $DATASET_DIR"
}

run_training() {
    log_info "Starting training..."
    
    # Create directories
    mkdir -p "$CHECKPOINTS_DIR"
    mkdir -p "$LOGS_DIR"
    
    # Set environment variables
    export TOKENIZERS_PARALLELISM=false
    export TRANSFORMERS_CACHE="${PROJECT_DIR}/.cache/transformers"
    
    # Run training
    python3 "${TRAINING_DIR}/train.py" \
        --config "$CONFIG_FILE" \
        2>&1 | tee "${LOGS_DIR}/training_$(date +%Y%m%d_%H%M%S).log"
    
    log_success "Training completed. Checkpoints saved at $CHECKPOINTS_DIR"
}

run_inference() {
    log_info "Running inference on test images..."
    
    INFERENCE_OUTPUT="${PROJECT_DIR}/inference_results"
    mkdir -p "$INFERENCE_OUTPUT"
    
    # Find the best checkpoint
    BEST_CHECKPOINT=$(ls -td "${CHECKPOINTS_DIR}"/checkpoint-* 2>/dev/null | head -1)
    
    if [ -z "$BEST_CHECKPOINT" ]; then
        log_warning "No checkpoint found, using original model"
        BEST_CHECKPOINT="$MODEL_DIR"
    fi
    
    log_info "Using checkpoint: $BEST_CHECKPOINT"
    
    # Run inference on test images
    python3 "${TRAINING_DIR}/inference.py" \
        --model-path "$BEST_CHECKPOINT" \
        --image-dir "$IMAGES_DIR" \
        --output-dir "$INFERENCE_OUTPUT" \
        --device "$DEVICE"
    
    log_success "Inference completed. Results saved at $INFERENCE_OUTPUT"
}

show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --install-deps    Install Python dependencies"
    echo "  --prepare-data    Prepare dataset from annotations"
    echo "  --train           Run model training"
    echo "  --infer           Run inference on test images"
    echo "  --all             Run complete pipeline"
    echo "  --help            Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 --all              # Run complete pipeline"
    echo "  $0 --prepare-data     # Only prepare dataset"
    echo "  $0 --train            # Only run training"
}

# =============================================================================
# Main
# =============================================================================

main() {
    echo "=============================================="
    echo "DeepSeek OCR Training Pipeline"
    echo "Turkish E-Invoice Extraction"
    echo "=============================================="
    echo ""
    
    # Parse arguments
    if [ $# -eq 0 ]; then
        show_usage
        exit 0
    fi
    
    # Check requirements first
    check_requirements
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --install-deps)
                install_dependencies
                shift
                ;;
            --prepare-data)
                prepare_dataset
                shift
                ;;
            --train)
                run_training
                shift
                ;;
            --infer)
                run_inference
                shift
                ;;
            --all)
                install_dependencies
                prepare_dataset
                run_training
                run_inference
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    echo ""
    log_success "Pipeline completed!"
}

main "$@"
