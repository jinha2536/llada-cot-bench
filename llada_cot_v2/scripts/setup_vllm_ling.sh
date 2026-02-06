#!/bin/bash
# Setup vLLM with Ling-mini-2.0 patch for Colab
# Run this before using --model ling

set -e

echo "=== Setting up vLLM with Ling patch ==="

# Check if vLLM is already installed with patch
if python -c "from vllm import LLM; LLM" 2>/dev/null; then
    echo "vLLM already installed. Testing Ling support..."
    if python -c "from vllm import LLM; LLM(model='inclusionAI/Ling-mini-2.0', dtype='bfloat16', trust_remote_code=True, max_model_len=1024)" 2>/dev/null; then
        echo "Ling support OK!"
        exit 0
    fi
fi

echo "Installing vLLM with Ling patch..."

# Clone vLLM v0.10.0
cd /content
rm -rf vllm
git clone -b v0.10.0 https://github.com/vllm-project/vllm.git
cd vllm

# Download and apply Ling patch
wget -q https://raw.githubusercontent.com/inclusionAI/Ling-V2/refs/heads/main/inference/vllm/bailing_moe_v2.patch
git apply bailing_moe_v2.patch

# Install vLLM (this takes ~5-10 minutes)
echo "Building vLLM (this may take 5-10 minutes)..."
pip install -e . --quiet

echo "=== vLLM setup complete ==="
