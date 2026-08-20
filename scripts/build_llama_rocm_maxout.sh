#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-"$HOME/src/llama.cpp"}
BUILD=${BUILD:-"$ROOT/build-rocm-maxout"}
JOBS=${JOBS:-"$(nproc)"}
LLAMA_REF=${LLAMA_REF:-master}

need() { command -v "$1" >/dev/null 2>&1 || { echo "missing required command: $1" >&2; exit 2; }; }
need git
need cmake
need rocminfo

if [[ ! -d "$ROOT/.git" ]]; then
  git clone https://github.com/ggml-org/llama.cpp.git "$ROOT"
fi

git -C "$ROOT" fetch --tags --prune origin
git -C "$ROOT" checkout "$LLAMA_REF"
if [[ "$LLAMA_REF" == "master" || "$LLAMA_REF" == "main" ]]; then
  git -C "$ROOT" pull --ff-only
fi

# Compile for every distinct gfx target exposed by ROCm. This matters on hybrid
# APU + OCuLink dGPU hosts where the integrated and discrete GPUs are different
# architectures. GPU_TARGETS can be overridden explicitly with a semicolon list.
if [[ -n "${GPU_TARGET:-}" && -z "${GPU_TARGETS:-}" ]]; then
  GPU_TARGETS="$GPU_TARGET"
fi
GPU_TARGETS=${GPU_TARGETS:-$(rocminfo 2>/dev/null | awk '/^[[:space:]]*Name:[[:space:]]*gfx[0-9]+/{print $2}' | awk '!seen[$0]++' | paste -sd';' -)}
if [[ -z "${GPU_TARGETS:-}" ]]; then
  echo "Unable to determine gfx targets from rocminfo. Set GPU_TARGETS explicitly (for example gfx1150;gfx1201)." >&2
  exit 2
fi

ROCM_PATH=${ROCM_PATH:-/opt/rocm}
if [[ ! -d "$ROCM_PATH" ]]; then
  ROCM_PATH=$(hipconfig -R 2>/dev/null || true)
fi
if [[ -z "${ROCM_PATH:-}" || ! -d "$ROCM_PATH" ]]; then
  echo "ROCm path not found. Set ROCM_PATH." >&2
  exit 2
fi

export ROCM_PATH
export HIP_PATH=${HIP_PATH:-$ROCM_PATH}
export HIPCXX=${HIPCXX:-"$(hipconfig -l 2>/dev/null)/clang"}

rm -rf "$BUILD"
cmake -S "$ROOT" -B "$BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DGGML_HIP_GRAPHS=ON \
  -DGGML_HIP_ROCWMMA_FATTN=ON \
  -DGGML_CUDA_FA=ON \
  -DGPU_TARGETS="$GPU_TARGETS" \
  -DGGML_BACKEND_DL=ON \
  -DGGML_CPU_ALL_VARIANTS=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TOOLS=ON
cmake --build "$BUILD" --config Release -j"$JOBS"

BIN="$BUILD/bin"
"$BIN/llama-server" --version || true
"$BIN/llama-bench" --list-devices || true

echo "LLAMA_ROOT=$ROOT"
echo "LLAMA_BUILD=$BUILD"
echo "LLAMA_COMMIT=$(git -C "$ROOT" rev-parse HEAD)"
echo "GPU_TARGETS=$GPU_TARGETS"
echo "ROCM_PATH=$ROCM_PATH"
