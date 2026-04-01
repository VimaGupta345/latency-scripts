#!/bin/bash
#
# Fill in missing batch sizes for all models.
# Uses reduced NUM_PROMPTS for bs=1,2 (they're slow).
#
# GPUs: 0,1,2,3,4,5,7 (skip 6)

set -euo pipefail
cd "$(dirname "$0")"

echo "============================================================"
echo "  Filling missing batch sizes for all models"
echo "============================================================"

# --- Wave 1: Small BS runs (bs=1,2) for 3 models + Q3-30B bs=2 ---
# These are slow, use fewer prompts. Run all TP=1 models in parallel (1 GPU each).
# Q3-30B bs=2:   GPU 0
# Q3-32B bs=1,2: GPU 1,2
# Q2-57B TP1 bs=1,2: GPU 3,5
# Total: 5 GPUs for 7 batch-size runs (skip 4,6)
echo ""
echo "── Wave 1: Small batch sizes (bs=1,2) ──"

# Q3-30B MoE: bs=2 only (bs=1 already done)
BATCH_SIZES="2" NUM_PROMPTS=512 \
./run_max_bs_sweep.sh \
    Qwen/Qwen3-30B-A3B-Instruct-2507 qwen3_30b \
    ../prowl/configs/qwen3_30b/qwen_do-nothing.json 1 "0" &
PID1=$!

# Q3-32B dense: bs=1,2
BATCH_SIZES="1 2" NUM_PROMPTS=512 \
./run_max_bs_sweep.sh \
    Qwen/Qwen3-32B qwen3_32b \
    none 1 "1,2" &
PID2=$!

# Q2-57B TP1: bs=1,2
BATCH_SIZES="1 2" NUM_PROMPTS=512 \
./run_max_bs_sweep.sh \
    Qwen/Qwen2-57B-A14B-Instruct qwen2_57b \
    ../prowl/configs/qwen/qwen_do-nothing.json 1 "3,5" &
PID3=$!

echo "  Waiting for wave 1 (PIDs: $PID1 $PID2 $PID3)..."
wait $PID1 || echo "  WARNING: Q3-30B bs=2 failed"
wait $PID2 || echo "  WARNING: Q3-32B bs=1,2 failed"
wait $PID3 || echo "  WARNING: Q2-57B TP1 bs=1,2 failed"
echo "  Wave 1 complete."

# --- Wave 2: Q2-57B TP2 bs=1,2 (needs 2 GPUs per slot) ---
echo ""
echo "── Wave 2: Q2-57B TP=2 bs=1,2 ──"

BATCH_SIZES="1 2" NUM_PROMPTS=512 \
./run_max_bs_sweep.sh \
    Qwen/Qwen2-57B-A14B-Instruct qwen2_57b_tp2 \
    ../prowl/configs/qwen/qwen_do-nothing.json 2 "0,1,2,3" &
PID4=$!

echo "  Waiting for wave 2 (PID: $PID4)..."
wait $PID4 || echo "  WARNING: Q2-57B TP2 bs=1,2 failed"
echo "  Wave 2 complete."

# --- Wave 3: Q3-30B MoE large BS (256,512,1024) ---
echo ""
echo "── Wave 3: Q3-30B MoE large batch sizes ──"

BATCH_SIZES="256 512 1024" NUM_PROMPTS=2048 \
./run_max_bs_sweep.sh \
    Qwen/Qwen3-30B-A3B-Instruct-2507 qwen3_30b \
    ../prowl/configs/qwen3_30b/qwen_do-nothing.json 1 "0,1,2" &
PID5=$!

echo "  Waiting for wave 3 (PID: $PID5)..."
wait $PID5 || echo "  WARNING: Q3-30B large BS failed"
echo "  Wave 3 complete."

echo ""
echo "============================================================"
echo "  All remaining experiments complete!"
echo "============================================================"
