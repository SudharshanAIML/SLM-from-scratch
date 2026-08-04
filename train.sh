#!/bin/bash
# SLM Training Helper Script
# Convenient commands for common training tasks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function print_header() {
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
}

function print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

function print_error() {
    echo -e "${RED}✗ $1${NC}"
}

function print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

function print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

# Check if virtual environment is activated
function check_venv() {
    if [[ "$VIRTUAL_ENV" == "" ]]; then
        print_warning "Virtual environment not activated"
        print_info "Run: source venv/bin/activate"
        return 1
    fi
    print_success "Virtual environment activated: $VIRTUAL_ENV"
    return 0
}

# Check GPU availability
function check_gpu() {
    print_header "GPU Status"
    
    if ! command -v nvidia-smi &> /dev/null; then
        print_error "nvidia-smi not found - NVIDIA driver may not be installed"
        return 1
    fi
    
    nvidia-smi --query-gpu=index,name,driver_version,memory.total \
        --format=csv,noheader
    
    python3 << 'EOF'
import torch
print(f"\nPyTorch CUDA capability:")
print(f"  Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print(f"  Compute Capability: {torch.cuda.get_device_capability(0)}")
EOF
    
    print_success "GPU check complete"
}

# Validate complete setup
function validate_setup() {
    print_header "Environment Validation"
    python3 scripts/validate_setup.py
}

# Show Python and package versions
function show_versions() {
    print_header "Package Versions"
    
    python3 << 'EOF'
import sys
import torch
import numpy as np
import yaml

try:
    import datasets
    datasets_version = datasets.__version__
except:
    datasets_version = "not installed"

try:
    import sentencepiece
    sp_version = sentencepiece.__version__
except:
    sp_version = "not installed"

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"NumPy: {np.__version__}")
print(f"PyYAML: {yaml.__version__}")
print(f"Datasets: {datasets_version}")
print(f"SentencePiece: {sp_version}")

if torch.cuda.is_available():
    print(f"CUDA: {torch.version.cuda}")
    print(f"cuDNN: {torch.backends.cudnn.version()}")
EOF
}

# List GPU processes
function gpu_processes() {
    print_header "GPU Processes"
    nvidia-smi | grep -E "PID|python|train" || print_info "No GPU processes found"
}

# Monitor GPU
function monitor_gpu() {
    print_header "GPU Monitor (Press Ctrl+C to stop)"
    watch -n 1 "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature_gpu --format=csv,noheader"
}

# Monitor training
function monitor_training() {
    print_header "Training Monitor"
    
    if [ ! -f "checkpoints/logs/history.jsonl" ]; then
        print_error "No training history found. Start training first."
        return 1
    fi
    
    print_info "Latest 10 training steps:"
    tail -10 checkpoints/logs/history.jsonl | python3 << 'EOF'
import sys
import json

for i, line in enumerate(sys.stdin, 1):
    data = json.loads(line)
    print(f"  Step {data['step']:6d}: loss={data['loss']:7.4f}, lr={data.get('lr', 0):8.5f}")
EOF
}

# Preprocess dataset
function preprocess_data() {
    print_header "FineWeb-Edu Preprocessing"
    
    local limit=${1:-100000}
    local vocab=${2:-50000}
    
    print_info "Configuration:"
    print_info "  Documents: $limit"
    print_info "  Vocab size: $vocab"
    
    check_venv || return 1
    
    python3 scripts/preprocess_fineweb.py \
        --limit "$limit" \
        --vocab-size "$vocab" \
        --context-length 2048 \
        --tokenizer simple
    
    print_success "Preprocessing complete"
}

# Train small model (recommended)
function train_small() {
    print_header "Training Small Model (90M params)"
    
    check_venv || return 1
    
    print_info "Configuration:"
    print_info "  Model: Small (~90M params)"
    print_info "  Batch size: 4"
    print_info "  Gradient accumulation: 4"
    print_info "  Max steps: 100000"
    print_info "  Context length: 2048"
    
    python3 scripts/train_real.py \
        --data-dir datasets/fineweb_edu/processed/train \
        --model-size small \
        --batch-size 4 \
        --gradient-accumulation-steps 4 \
        --max-steps 100000 \
        --learning-rate 5e-4 \
        --checkpoint-dir checkpoints
}

# Train medium model
function train_medium() {
    print_header "Training Medium Model (251M params)"
    
    check_venv || return 1
    
    print_warning "Medium model requires gradient checkpointing"
    
    print_info "Configuration:"
    print_info "  Model: Medium (~251M params)"
    print_info "  Batch size: 4"
    print_info "  Gradient accumulation: 8"
    print_info "  Gradient checkpointing: enabled"
    print_info "  Max steps: 50000"
    
    python3 scripts/train_real.py \
        --data-dir datasets/fineweb_edu/processed/train \
        --model-size medium \
        --batch-size 4 \
        --gradient-accumulation-steps 8 \
        --gradient-checkpointing \
        --max-steps 50000 \
        --checkpoint-dir checkpoints
}

# Resume training
function resume_training() {
    print_header "Resume Training"
    
    check_venv || return 1
    
    # Find latest checkpoint
    latest_checkpoint=$(ls -t checkpoints/model_step_*.pt 2>/dev/null | head -1)
    
    if [ -z "$latest_checkpoint" ]; then
        print_error "No checkpoint found"
        return 1
    fi
    
    print_success "Found checkpoint: $latest_checkpoint"
    
    python3 scripts/train_real.py \
        --data-dir datasets/fineweb_edu/processed/train \
        --checkpoint-dir checkpoints
}

# Show training statistics
function training_stats() {
    print_header "Training Statistics"
    
    if [ ! -f "checkpoints/logs/history.jsonl" ]; then
        print_error "No training history found"
        return 1
    fi
    
    python3 << 'EOF'
import json
from pathlib import Path
import statistics

history_file = Path("checkpoints/logs/history.jsonl")
with open(history_file) as f:
    losses = []
    for line in f:
        data = json.loads(line)
        losses.append(data['loss'])

if losses:
    print(f"Steps completed: {len(losses)}")
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Latest loss: {losses[-1]:.4f}")
    print(f"Improvement: {(1 - losses[-1]/losses[0])*100:.1f}%")
    
    if len(losses) > 100:
        last_100 = losses[-100:]
        print(f"\nLast 100 steps:")
        print(f"  Average loss: {statistics.mean(last_100):.4f}")
        print(f"  Std dev: {statistics.stdev(last_100):.4f}")
        print(f"  Min: {min(last_100):.4f}")
        print(f"  Max: {max(last_100):.4f}")
    
    # Check for NaN
    if any(str(x) == 'nan' for x in losses):
        print("\n⚠️ WARNING: Loss diverged to NaN!")
EOF
}

# Setup help
function show_help() {
    print_header "SLM Training Helper - Available Commands"
    
    cat << 'EOF'
Setup Commands:
  ./train.sh check-gpu         - Check GPU availability and status
  ./train.sh validate          - Validate complete environment setup
  ./train.sh versions          - Show package versions
  ./train.sh gpu-processes     - List GPU processes

Data Commands:
  ./train.sh preprocess [N]    - Preprocess FineWeb-Edu (N docs, default 100000)
  ./train.sh preprocess 10000  - Preprocess 10k docs (quick test)

Training Commands:
  ./train.sh train-small       - Train small model (90M, RECOMMENDED)
  ./train.sh train-medium      - Train medium model (251M, needs checkpointing)
  ./train.sh resume            - Resume from latest checkpoint

Monitoring Commands:
  ./train.sh monitor-gpu       - Watch GPU usage (Ctrl+C to exit)
  ./train.sh monitor           - Show latest training steps
  ./train.sh stats             - Show training statistics
  ./train.sh processes         - List GPU processes

Example Workflow:
  1. ./train.sh validate           # Verify setup
  2. ./train.sh preprocess 100000  # Prepare data (~5 min)
  3. ./train.sh train-small        # Start training
  4. (in another terminal)
  5. ./train.sh monitor-gpu        # Watch GPU
  6. ./train.sh monitor            # Watch loss
  7. ./train.sh resume             # Resume after interrupt

For detailed documentation, see:
  - SETUP.md              - Complete installation guide
  - QUICK_START.md        - Fast path for experts
  - TRAINING_MONITORING.md - Real-time diagnostics
  - TROUBLESHOOTING.md    - Error solutions
  - ARCHITECTURE.md       - System design
EOF
}

# Main command dispatch
case "${1:-help}" in
    check-gpu|gpu)
        check_gpu
        ;;
    validate)
        validate_setup
        ;;
    versions)
        show_versions
        ;;
    processes|gpu-processes)
        gpu_processes
        ;;
    preprocess)
        preprocess_data "${2:-100000}" "${3:-50000}"
        ;;
    train|train-small)
        train_small
        ;;
    train-medium)
        train_medium
        ;;
    resume)
        resume_training
        ;;
    monitor-gpu)
        monitor_gpu
        ;;
    monitor)
        monitor_training
        ;;
    stats)
        training_stats
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
