#!/usr/bin/env bash
set -euo pipefail
# K2 native code runs only under the verified immutable research boundary:
# no network, nonroot, read-only root+source+model, no caps, bounded CPU/memory.
root=/home/scott/git/wt-lms-exp053-minicpm-local-20261022
private=/media/scott/SSD_4TB/exp053-private-20261016
output="$private/k2-machine-proxy-20261022-v1"
model=/home/scott/models/research/K2-Horizon-3.7B
source_snapshot=/media/scott/SSD_4TB/hf-cache/huggingface/hub/datasets--ehovy--race
overlay=/home/scott/git/.latent-minicpm-v47-venv/lib/python3.12/site-packages
cpu=/home/scott/ryzen-ai-venv/lib/python3.12/site-packages
image=auto-assist-wt-ab3bb64-hermes-adapter:latest
test -d "$root" && test -d "$model" && test -d "$overlay" && test -d "$cpu" && test -f "$source_snapshot/snapshots/2fec9fd81f1dc971569a9b729c43f2f0e6436637/high/test-00000-of-00001.parquet"
cas_file=$(readlink -f "$source_snapshot/snapshots/2fec9fd81f1dc971569a9b729c43f2f0e6436637/high/test-00000-of-00001.parquet")
case "$cas_file" in /media/scott/SSD_4TB/hf-cache/huggingface/hub/blobs/*) ;; *) echo 'HOLD: unexpected resolved source cache' >&2; exit 5 ;; esac
test -f "$cas_file" && test ! -L "$cas_file"
test -d "$output" && test ! -L "$output"
test "$(stat -c %u "$private")" = "$(id -u)"
test "$(stat -c %a "$private")" = 700
test "$(stat -c %u "$output")" = "$(id -u)"
test "$(stat -c %a "$output")" = 700
exec timeout 540s docker run --rm --pull never \
  --network none --read-only --cap-drop ALL --security-opt no-new-privileges \
  --pids-limit 96 --memory 28g --memory-swap 28g --cpus 2 \
  --user "$(id -u):$(id -g)" --tmpfs /tmp:rw,nosuid,nodev,size=512m,mode=1777 \
  -v "$overlay:/overlay:ro" -v "$cpu:/cpu:ro" \
  -v "$model:$model:ro" -v "$root:$root:ro" -v "$source_snapshot:$source_snapshot:ro" -v "$cas_file:$cas_file:ro" \
  -v "$private:$private:ro" -v "$output:$output:rw" \
  -e PYTHONPATH=/overlay:/cpu -e HOME=/tmp -e HF_HOME=/tmp/hf \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e HF_DATASETS_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1 \
  -e TOKENIZERS_PARALLELISM=false -e PYTHONDONTWRITEBYTECODE=1 \
  -e CUDA_VISIBLE_DEVICES='' -e HIP_VISIBLE_DEVICES='' \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  -e EXP053_K2_SANDBOX=networkless-readonly \
  --entrypoint python "$image" -B "$root/research/exp053_agent_k2_cpu.py" "$@"
