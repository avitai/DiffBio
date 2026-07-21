#!/usr/bin/env bash
# Drive the computational-cost grid, one configuration per process so peak GPU memory
# is clean, and collect the JSON lines into benchmarks/results/singlecell/compute_cost.json.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
# Disable GPU preallocation so peak_bytes_in_use reflects true on-demand usage per config
# (JAX profiling best practice), not the default 75% BFC reservation.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
OUT=benchmarks/results/singlecell/compute_cost.json
mkdir -p "$(dirname "$OUT")"
: > "$OUT.tmp"

run() { # args passed straight to the worker
  uv run python -m benchmarks.singlecell.compute_cost "$@" 2>/dev/null | grep '^{' >> "$OUT.tmp"
}

# Main comparison at atlas scale (100k cells, 2000 features, k=10, batch 4096)
run --arm frozen          --n-cells 100000 --batch-size 4096
run --arm joint_minibatch --n-cells 100000 --batch-size 4096
run --arm joint_fullbatch --n-cells 100000

# Peak memory vs batch size (joint mini-batch) -- should scale with batch, not n_cells
run --arm joint_minibatch --n-cells 100000 --batch-size 512
run --arm joint_minibatch --n-cells 100000 --batch-size 16384

# Peak memory vs n_cells: mini-batch stays flat, full-batch grows with the atlas
run --arm joint_minibatch --n-cells 50000  --batch-size 4096
run --arm joint_minibatch --n-cells 200000 --batch-size 4096
run --arm joint_fullbatch --n-cells 50000
run --arm joint_fullbatch --n-cells 200000

python3 -c "import json,sys; rows=[json.loads(l) for l in open('$OUT.tmp')]; json.dump(rows, open('$OUT','w'), indent=2)"
rm -f "$OUT.tmp"
echo "=== compute-cost grid ==="
python3 -c "
import json
for r in json.load(open('$OUT')):
    print(f\"  {r['config']:16s} n={r['n_cells']:7d} batch={r['batch_size']:6d}  {r['ms_per_step']:7.2f} ms/step  {r['peak_gpu_mb']:8.1f} MB\")
"
echo "DONE -> $OUT"
