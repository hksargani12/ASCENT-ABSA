#!/usr/bin/env bash
set -euo pipefail

MODEL="bert-base-uncased"
OUT_DIR="/models/ascent_pretrain_outputs"
PREFIX="ascent_pretrain"
EPOCHS=10
BATCH_SIZE=64
LR=5e-5
WEIGHT_DECAY=5e-5
GPU_ID=0
SEQ_LEN=128
NUM_LABELS=2

LAMBDA_SCL=2.0
LAMBDA_MLM=1.0
LAMBDA_CLS=1.0
LOG_LEVEL="INFO"
SAVE_EACH_EPOCH="--save_each_epoch"

# For Amazon dataset:
# DATA_PATH="/data/amazon_lap_train_dataset.csv"
# DATASET="amazon"

# For Yelp dataset:
# DATA_PATH="/data/yelp_train_dataset.csv"
# DATASET="yelp"

if [ -z "${DATA_PATH:-}" ] || [ -z "${DATASET:-}" ]; then
  echo "Error: Please set DATA_PATH and DATASET before running."
  exit 1
fi

python3 ./ascent_pretrain.py \
  --train_data_path "${DATA_PATH}" \
  --dataset_name "${DATASET}" \
  --output_dir "${OUT_DIR}" \
  --output_prefix "${PREFIX}" \
  --bert_model "${MODEL}" \
  --do_lower_case \
  --max_seq_length "${SEQ_LEN}" \
  --train_epochs "${EPOCHS}" \
  --train_batch_size "${BATCH_SIZE}" \
  --learning_rate "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --num_labels "${NUM_LABELS}" \
  --lambda_scl "${LAMBDA_SCL}" \
  --lambda_mlm "${LAMBDA_MLM}" \
  --lambda_cls "${LAMBDA_CLS}" \
  --gpus "${GPU_ID}" \
  ${SAVE_EACH_EPOCH} \
  --log_level "${LOG_LEVEL}"

echo "Pretraining complete for dataset: ${DATASET}"
echo "Outputs saved in: ${OUT_DIR}/${DATASET}/"
