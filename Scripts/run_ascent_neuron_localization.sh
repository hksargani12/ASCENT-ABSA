#!/usr/bin/env bash
set -euo pipefail

MODEL="bert-base-uncased"
OUT_DIR="/models/ascent_neuron_outputs"
PREFIX="neurons"
BATCH_SIZE=20
NUM_BATCH=1
SEQ_LEN=128
GPU_ID=0
LOG_LEVEL="INFO"

# For Laptop14 dataset:
# DATA_PATH="/data/masked_laptop_train.json"
# DATASET="lap14"

# For Restaurant14 dataset:
# DATA_PATH="/data/masked_restaurant_train.json"
# DATASET="res14"

# For MAMS dataset:
# DATA_PATH="/data/masked_mams_train.json"
# DATASET="mams"

if [ -z "${DATA_PATH:-}" ] || [ -z "${DATASET:-}" ]; then
  echo "Error: Please set DATA_PATH and DATASET before running."
  exit 1
fi

python3 ./ascent_neuron_localization.py \
  --dataset_name "${DATASET}" \
  --data_path "${DATA_PATH}" \
  --bert_model "${MODEL}" \
  --output_dir "${OUT_DIR}" \
  --output_prefix "${PREFIX}" \
  --max_seq_length "${SEQ_LEN}" \
  --batch_size "${BATCH_SIZE}" \
  --num_batch "${NUM_BATCH}" \
  --do_lower_case \
  --get_pred \
  --get_ig_pred \
  --get_ig_gold \
  --gpus "${GPU_ID}" \
  --log_level "${LOG_LEVEL}"

echo "Neuron localization complete for dataset: ${DATASET}"
echo "Results saved in: ${OUT_DIR}/${DATASET}/"
