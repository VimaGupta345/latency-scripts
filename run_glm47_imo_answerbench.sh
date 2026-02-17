#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "${SCRIPT_DIR}"

MODEL_PATH=${MODEL_PATH:-QuantTrio/GLM-4.7-AWQ}
BENCHMARK=${BENCHMARK:-imo_answerbench}

GPUS=${GPUS:-2,3,4,5}

GPUS="${GPUS}" \
MODEL_PATH="${MODEL_PATH}" \
BENCHMARK="${BENCHMARK}" \
./run_glm47_swebench.sh
