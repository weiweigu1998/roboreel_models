#!/bin/bash
# evaluate.sh — run an eval 3 times with seed_offset_base ∈ {0, 100000,
# 200000} to realize the protocol's "3 seed passes per cell". scripts/evaluation.py
# does one pass per invocation (env.reset seed = demo_id + seed_offset_base),
# so 3 invocations with different offsets give 3 distinct env-seed sets per
# demo.
#
# Each pass writes its own summary.json under <base_output_dir>/seed_offset_<N>/;
# Each pass's summary.json carries the per-cell counts and every rollout, so
# the three can be aggregated into per-task mean +/- SEM
# reporting mean ± SEM across the 3 passes (SEM = sample stdev with ddof=1,
# divided by √n).
#
# Usage:
#   scripts/evaluate.sh <base_cfg.json> <gpu> <python> [mode]
#     <base_cfg.json> : eval config JSON. Its `output_dir` is the base; the 3
#                       passes are written under it. If unset, defaults to
#                       /tmp/eval_<config-stem>.
#     <gpu>           : CUDA device id (e.g. 1).
#     <python>        : python interpreter — the eval conda env's bin/python
#                       (e.g. /<conda>/envs/<env>/bin/python).
#     [mode]          : "sequential" (default) | "parallel". Parallel runs all
#                       3 passes simultaneously on the same GPU — appropriate
#                       when each pass uses far less than the GPU's VRAM (the
#                       eval rollouts here are ~1 GB and ~10-15% util, so 3
#                       passes share one GPU comfortably and finish faster).
#                       Each pass's stdout is written to <pass_dir>/pass.log.
set -euo pipefail
[ $# -ge 3 ] && [ $# -le 4 ] || { echo "usage: $0 <base_cfg.json> <gpu> <python> [mode=sequential|parallel]"; exit 2; }
BASE_CFG="$1"; GPU="$2"; ENV_PY="$3"; MODE="${4:-sequential}"
[ "$MODE" = "sequential" ] || [ "$MODE" = "parallel" ] || {
    echo "mode must be 'sequential' or 'parallel' (got: $MODE)"; exit 2;
}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
[ -x "$ENV_PY" ]  || { echo "python not executable: $ENV_PY"; exit 1; }
[ -f "$BASE_CFG" ] || { echo "config not found: $BASE_CFG"; exit 1; }

BASE_OUT=$("$ENV_PY" - <<PY
import json, os
d = json.load(open("$BASE_CFG"))
out = d.get("output_dir")
if not out:
    out = "/tmp/eval_" + os.path.splitext(os.path.basename("$BASE_CFG"))[0]
print(out)
PY
)
mkdir -p "$BASE_OUT"
echo "[3pass] base output_dir: $BASE_OUT"

PIDS=()
for OFFSET in 0 100000 200000; do
    OUTDIR="$BASE_OUT/seed_offset_$OFFSET"
    CFG="$BASE_OUT/cfg_seed_offset_$OFFSET.json"
    mkdir -p "$OUTDIR"
    "$ENV_PY" - <<PY
import json
d = json.load(open("$BASE_CFG"))
d["seed_offset_base"] = $OFFSET
d["output_dir"] = "$OUTDIR"
json.dump(d, open("$CFG","w"), indent=2)
PY
    echo
    echo "=== pass: seed_offset_base=$OFFSET → $OUTDIR  (mode=$MODE) ==="
    ( cd "$REPO" && "$ENV_PY" scripts/evaluation.py --config "$CFG" --gpu "$GPU" \
        > "$OUTDIR/pass.log" 2>&1 ) &
    pid=$!
    PIDS+=("$pid")
    if [ "$MODE" = "sequential" ]; then
        wait "$pid"
    fi
done
if [ "$MODE" = "parallel" ]; then
    echo "[3pass] waiting on ${#PIDS[@]} parallel passes (pids: ${PIDS[*]})"
    for pid in "${PIDS[@]}"; do wait "$pid" || true; done
fi

echo
echo "=== aggregating 3 passes ==="
"$ENV_PY" - "$BASE_OUT" <<'PY'
import json, sys, math
from pathlib import Path
base = Path(sys.argv[1])
sums = []
for o in (0, 100000, 200000):
    sf = base / f"seed_offset_{o}" / "summary.json"
    if sf.is_file():
        sums.append(json.loads(sf.read_text()))
    else:
        print(f"  MISSING: {sf}")
if len(sums) < 2:
    print(f"need >=2 passes to aggregate; got {len(sums)}"); sys.exit(0)
cells = sorted({k for s in sums for k in s.get("table", {})})
agg = {}
for k in cells:
    rates = [s["table"][k]["rate"] for s in sums if k in s.get("table", {})]
    mean = sum(rates) / len(rates)
    var = sum((r - mean) ** 2 for r in rates) / max(len(rates) - 1, 1)
    agg[k] = {"per_pass_rate": rates, "mean": mean, "stdev": math.sqrt(var), "n_passes": len(rates)}
out = base / "aggregated_3pass.json"
out.write_text(json.dumps({
    "n_passes": len(sums),
    "seed_offsets": [s.get("seed_offset_base") for s in sums],
    "per_cell": agg,
}, indent=2))
print(f"Wrote {out}\n")
print(f"{'cell':40s}  mean ± stdev   (n passes)")
for k, v in agg.items():
    print(f"  {k:40s}  {v['mean']*100:5.1f}% +/- {v['stdev']*100:4.1f}%   (n={v['n_passes']})")
PY
echo
echo "[3pass] done."
