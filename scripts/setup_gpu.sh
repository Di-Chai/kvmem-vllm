#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv --python 3.12 --allow-existing .venv
uv pip install --python .venv/bin/python --torch-backend=cu130 -r requirements-cu130.lock
if [ ! -d .deps/vllm-gguf-plugin/.git ]; then
  git clone https://github.com/vllm-project/vllm-gguf-plugin.git .deps/vllm-gguf-plugin
fi
git -C .deps/vllm-gguf-plugin checkout --detach e2b8ad532b8b5ea175100202c30430c1d2b5e6a8
MAX_JOBS=2 TORCH_CUDA_ARCH_LIST=12.0 CUDA_HOME=/usr/local/cuda-13 \
  PATH="$PWD/.venv/bin:/usr/local/cuda-13/bin:$PATH" \
  uv pip install --python .venv/bin/python --no-build-isolation --no-deps .deps/vllm-gguf-plugin
uv pip install --python .venv/bin/python --no-build-isolation -e .
