#!/usr/bin/env python3
"""
Validation script to verify all components are ready for training.
Run this before starting real training to catch configuration issues early.
"""

import sys
import subprocess
from pathlib import Path


def check_module(name, import_name=None):
    """Check if a Python module is installed."""
    if import_name is None:
        import_name = name
    try:
        __import__(import_name)
        print(f"✓ {name}")
        return True
    except ImportError as e:
        print(f"✗ {name}: {e}")
        return False


def check_command(name, cmd):
    """Check if a command is available."""
    try:
        subprocess.run([cmd, "--version"], capture_output=True, timeout=5)
        print(f"✓ {name}")
        return True
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print(f"✗ {name} not found")
        return False


def check_directory(path, should_exist=True):
    """Check if directory exists."""
    path = Path(path)
    if should_exist:
        if path.exists():
            print(f"✓ Directory: {path}")
            return True
        else:
            print(f"✗ Directory not found: {path}")
            return False
    else:
        if not path.exists():
            print(f"✓ Directory available (will be created): {path}")
            return True
        else:
            print(f"⚠ Directory already exists: {path}")
            return True


def check_slm_package():
    """Check if slm package is properly structured."""
    slm_dir = Path("slm")
    required_modules = [
        "slm/__init__.py",
        "slm/model/transformer.py",
        "slm/configs/model_config.py",
        "slm/configs/train_config.py",
        "slm/training/trainer.py",
        "slm/data/binary_dataset.py",
        "slm/preprocessing/fineweb_preprocessor.py",
    ]
    
    all_exist = True
    for module in required_modules:
        module_path = Path(module)
        if module_path.exists():
            print(f"✓ {module}")
        else:
            print(f"✗ Missing: {module}")
            all_exist = False
    
    return all_exist


def check_scripts():
    """Check if required scripts exist."""
    scripts_dir = Path("scripts")
    required_scripts = [
        "scripts/preprocess_fineweb.py",
        "scripts/train_real.py",
    ]
    
    all_exist = True
    for script in required_scripts:
        script_path = Path(script)
        if script_path.exists():
            print(f"✓ {script}")
        else:
            print(f"✗ Missing: {script}")
            all_exist = False
    
    return all_exist


def check_gpu():
    """Check GPU availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
            print(f"  - CUDA Version: {torch.version.cuda}")
            print(f"  - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            return True
        else:
            print("✗ No GPU found")
            return False
    except Exception as e:
        print(f"✗ GPU check failed: {e}")
        return False


def check_pytorch_cuda():
    """Check PyTorch CUDA integration."""
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        print(f"✓ cuDNN available: {torch.backends.cudnn.is_available()}")
        return torch.cuda.is_available()
    except Exception as e:
        print(f"✗ PyTorch/CUDA check failed: {e}")
        return False


def check_dataset():
    """Check if dataset shards exist."""
    dataset_dir = Path("datasets/fineweb_edu/processed/train")
    if not dataset_dir.exists():
        print(f"⚠ Dataset directory not found: {dataset_dir}")
        print("  (This is OK - run preprocess_fineweb.py to create it)")
        return True
    
    shards = list(dataset_dir.glob("shard_*.bin"))
    metadata = dataset_dir / "metadata.json"
    
    if shards:
        print(f"✓ Found {len(shards)} dataset shards")
    else:
        print(f"⚠ No dataset shards found in {dataset_dir}")
    
    if metadata.exists():
        print(f"✓ Metadata file exists")
        return True
    else:
        print(f"⚠ No metadata.json found")
        return True


def main():
    """Run all validation checks."""
    print("=" * 60)
    print("SLM Training Environment Validation")
    print("=" * 60)
    
    print("\n1. Python Packages")
    print("-" * 60)
    packages = [
        ("PyTorch", "torch"),
        ("NumPy", "numpy"),
        ("SentencePiece", "sentencepiece"),
        ("Hugging Face Datasets", "datasets"),
        ("PyYAML", "yaml"),
        ("tqdm", "tqdm"),
    ]
    packages_ok = all(check_module(name, import_name) for name, import_name in packages)
    
    print("\n2. System Commands")
    print("-" * 60)
    commands = [
        ("NVIDIA GPU Utils", "nvidia-smi"),
    ]
    commands_ok = all(check_command(name, cmd) for name, cmd in commands)
    
    print("\n3. GPU Status")
    print("-" * 60)
    gpu_ok = check_gpu()
    pytorch_ok = check_pytorch_cuda()
    
    print("\n4. Project Structure")
    print("-" * 60)
    slm_ok = check_slm_package()
    scripts_ok = check_scripts()
    
    print("\n5. Directories")
    print("-" * 60)
    dirs_ok = all([
        check_directory("checkpoints", should_exist=False),
        check_directory("datasets", should_exist=False),
    ])
    
    print("\n6. Dataset")
    print("-" * 60)
    dataset_ok = check_dataset()
    
    print("\n" + "=" * 60)
    
    # Summary
    all_critical = packages_ok and pytorch_ok and gpu_ok and slm_ok and scripts_ok
    
    if all_critical:
        print("✓ All critical checks passed!")
        print("\nNext steps:")
        print("1. Preprocess dataset:")
        print("   python3 scripts/preprocess_fineweb.py --limit 100000")
        print("\n2. Start training:")
        print("   python3 scripts/train_real.py --data-dir datasets/fineweb_edu/processed/train")
        return 0
    else:
        print("✗ Some critical checks failed!")
        print("\nPlease fix the issues above before proceeding.")
        if not packages_ok:
            print("\nTo install missing packages:")
            print("  pip install torch sentencepiece datasets pyyaml tqdm numpy")
        if not gpu_ok or not pytorch_ok:
            print("\nTo fix GPU issues:")
            print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        return 1


if __name__ == "__main__":
    sys.exit(main())
