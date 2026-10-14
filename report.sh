
cd "$HOME/Desktop/Paper/Vehicle Tracking Paper" || exit 1

REPORT="/tmp/vehicle_tracking_repo_report.txt"

{
    echo "============================================================"
    echo "VEHICLE TRACKING + OCCLUSIONS RESEARCH PROJECT"
    echo "REPOSITORY AND ENVIRONMENT REPORT"
    echo "============================================================"
    echo
    echo "DATE"
    date
    echo

    echo "============================================================"
    echo "1. REPOSITORY LOCATION AND TOP-LEVEL CONTENTS"
    echo "============================================================"
    pwd
    ls -lah
    echo

    echo "============================================================"
    echo "2. DIRECTORY TREE"
    echo "============================================================"
    if command -v tree >/dev/null 2>&1; then
        tree -a -I '.git|__pycache__|.pytest_cache|.mypy_cache|.ruff_cache|node_modules|.venv|venv|env|conda-meta|*.egg-info|checkpoints|weights|*.pt|*.pth|*.onnx|*.engine|*.mp4|*.avi|*.mkv|*.jpg|*.jpeg|*.png|*.webp|*.bmp|*.zip|*.tar|*.gz|*.xml|*.jsonl|*.parquet|*.npy|*.npz|*.pkl|*.pickle|*.db|*.sqlite|*.csv' \
            -L 6 .
    else
        find . \
            -path './.git' -prune -o \
            -path '*/__pycache__' -prune -o \
            -path '*/node_modules' -prune -o \
            -path '*/.venv' -prune -o \
            -path '*/venv' -prune -o \
            -path '*/checkpoints' -prune -o \
            -path '*/weights' -prune -o \
            -type f -print |
            sort
    fi
    echo

    echo "============================================================"
    echo "3. SOURCE, CONFIG, SCRIPT, AND NOTEBOOK FILE INVENTORY"
    echo "============================================================"
    find src configs scripts notebooks docs \
        -type f \
        ! -path '*/__pycache__/*' \
        ! -name '*.pyc' \
        -printf '%p (%s bytes)\n' 2>/dev/null | sort
    echo

    echo "============================================================"
    echo "4. DOCUMENTATION FILES"
    echo "============================================================"
    find docs -type f \
        \( -iname '*.pdf' -o -iname '*.md' -o -iname '*.txt' \
        -o -iname '*.docx' -o -iname '*.rst' \) \
        -printf '%p (%s bytes)\n' 2>/dev/null | sort
    echo

    echo "============================================================"
    echo "5. GIT STATUS AND RECENT COMMITS"
    echo "============================================================"
    git status --short --branch 2>&1
    echo
    git log -5 --oneline 2>&1
    echo
    echo "Tracked file counts by top-level directory:"
    git ls-files 2>/dev/null |
        awk -F/ '{print $1}' |
        sort | uniq -c | sort -nr
    echo

    echo "============================================================"
    echo "6. PYTHON AND CONDA ENVIRONMENTS"
    echo "============================================================"
    echo "Shell: $SHELL"
    echo "Python on PATH:"
    command -v python 2>&1
    python --version 2>&1
    echo
    echo "Python 3:"
    command -v python3 2>&1
    python3 --version 2>&1
    echo
    echo "Conda executable:"
    command -v conda 2>&1
    echo
    if command -v conda >/dev/null 2>&1; then
        echo "--- Conda version ---"
        conda --version 2>&1
        echo
        echo "--- Conda environments ---"
        conda env list 2>&1
        echo
        echo "--- Active environment packages ---"
        conda list 2>&1
    fi
    echo
    echo "--- Python environment variables ---"
    env | grep -E '^(CONDA_DEFAULT_ENV|CONDA_PREFIX|VIRTUAL_ENV|CUDA_HOME|CUDA_PATH|LD_LIBRARY_PATH|PATH)=' |
        sed -E 's/(TOKEN|PASSWORD|SECRET|KEY)=[^ ]+/\1=[REDACTED]/Ig'
    echo

    echo "============================================================"
    echo "7. PYTORCH, CUDA, AND GPU"
    echo "============================================================"
    python - <<'PY'
import sys, platform, importlib.util

print("Python executable:", sys.executable)
print("Python version:", sys.version)
print("Platform:", platform.platform())

for name in ("torch", "torchvision", "torchaudio", "ultralytics",
             "numpy", "opencv-python", "cv2", "transformers",
             "timm", "pandas", "scikit-learn", "PIL"):
    try:
        spec = importlib.util.find_spec(name)
        print(f"{name}: {'installed/available' if spec else 'not found'}")
    except Exception as e:
        print(f"{name}: inspection error: {e}")

try:
    import torch
    print("\nPyTorch version:", torch.__version__)
    print("PyTorch CUDA build:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))
        print("  Capability:", torch.cuda.get_device_capability(i))
        print("  Memory (bytes):", torch.cuda.get_device_properties(i).total_memory)
except Exception as e:
    print("PyTorch inspection failed:", repr(e))
PY
    echo

    echo "============================================================"
    echo "8. NVIDIA DRIVER AND CUDA TOOLKIT"
    echo "============================================================"
    nvidia-smi 2>&1
    echo
    nvcc --version 2>&1
    echo

    echo "============================================================"
    echo "9. RELEVANT INSTALLED PACKAGES"
    echo "============================================================"
    python -m pip show torch torchvision torchaudio ultralytics \
        numpy opencv-python transformers timm pandas scikit-learn 2>&1
    echo

    echo "============================================================"
    echo "10. REPOSITORY SIZE SUMMARY"
    echo "============================================================"
    du -sh . 2>&1
    echo
    du -h --max-depth=1 . 2>&1 | sort -h
    echo

    echo "============================================================"
    echo "END OF REPORT"
    echo "============================================================"

} > "$REPORT" 2>&1

echo "Report created: $REPORT"
echo "Report size:"
wc -l -c "$REPORT"
echo
echo "Open it with:"
echo "less $REPORT"
