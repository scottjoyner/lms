#!/usr/bin/env bash
# usage: sweep_node.sh <name> <ip> <model> <max_tokens>
set -u
NAME=$1; IP=$2; MODEL=$3; MT=${4:-256}
OUT="$HOME/git/lms/runs/fleet-maxprod-20260823/${NAME}_results.jsonl"
echo "{\"node\":\"$NAME\",\"model\":\"$MODEL\",\"start\":\"$(date -Iseconds)\"}" >> "$OUT"
for c in 1 2 4 8; do
  python3 "$HOME/git/lms/runs/x1-370-gpu-maxprod-20260823/gpu_bench.py" \
    --base "http://$IP:1234/v1" --model "$MODEL" --conc $c --max-tokens "$MT" \
    | python3 -c "import sys,json;d=json.load(sys.stdin);d['node']='$NAME';d['model']='$MODEL';print(json.dumps(d))" \
    >> "$OUT" 2>/dev/null || echo "{\"node\":\"$NAME\",\"conc\":$c,\"error\":\"failed\"}" >> "$OUT"
  sleep 1
done
echo "{\"node\":\"$NAME\",\"done\":true,\"end\":\"$(date -Iseconds)\"}" >> "$OUT"
