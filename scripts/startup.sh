#!/bin/bash
set -e

# Bulletproof Startup Script
echo "Starting Setup..."

# 1. System Updates
apt-get update -qq
apt-get install -y git zip python3-pip

# 2. Python Dependencies (Explicit Install)
# Installing here is safer than inside the script
echo "Installing Python Libraries..."
python3 -m pip install --upgrade pip
python3 -m pip install vllm datasets huggingface_hub torch transformers accelerate tqdm

# 3. Validation
if [ ! -f "scripts/curate_alexandria.py" ]; then
    echo "Error: scripts/curate_alexandria.py not found"
    exit 1
fi

# 4. Execution
echo "Running Curation..."
python3 scripts/curate_alexandria.py --total_samples 100000 --model "Qwen3-Next-80B-A3B-Thinking" --tensor_parallel 1

# 5. Archiving
echo "Zipping Data..."
if [ -d "data/alexandria" ]; then
    zip -r alexandria_100k.zip data/alexandria
    echo "Done. File: alexandria_100k.zip"
    
    # Optional: Auto-Upload to HF
    if [ ! -z "$HF_TOKEN" ] && [ ! -z "$HF_REPO_ID" ]; then
        echo "Uploading to HuggingFace ($HF_REPO_ID)..."
        huggingface-cli login --token "$HF_TOKEN"
        huggingface-cli upload "$HF_REPO_ID" alexandria_100k.zip --repo-type dataset
        echo "Upload Complete."
    fi
else
    echo "Error: No data directory found"
    exit 1
fi
