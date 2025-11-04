#!/usr/bin/env bash
set -euo pipefail

MODEL="bert-base-uncased"
OUT_DIR="/models/ascent_finetune_outputs"
PREFIX="ascent_ft"
SEQ_LEN=128
EPOCHS=10
BATCH=32
LR=5e-5
WD=5e-5
NUM_LABELS=3
GPU_ID=0
LOG_LEVEL="INFO"
SAVE_EACH_EPOCH="--save_each_epoch"
NEURONS_DIR="/data/neurons"  # contains neurons_<dataset>_<mode>.json

# Dataset paths (set one dataset at a time)
# Lap14:
# TRAIN_CSV="/data/lap14_train.csv"
# TEST_CSV="/data/lap14_test.csv"
# DATASET="lap14"

# Res14:
# TRAIN_CSV="/data/res14_train.csv"
# TEST_CSV="/data/res14_test.csv"
# DATASET="res14"

# MAMS:
# TRAIN_CSV="/data/mams_train.csv"
# TEST_CSV="/data/mams_test.csv"
# DATASET="mams"

# Mode: one of full|dormant|salient|random
# MODE="full"
# MODE="dormant"
# MODE="salient"
# MODE="random"

if [ -z "${TRAIN_CSV:-}" ] || [ -z "${TEST_CSV:-}" ] || [ -z "${DATASET:-}" ] || [ -z "${MODE:-}" ]; then
  echo "Error: set TRAIN_CSV, TEST_CSV, DATASET (lap14|res14|mams), MODE (full|dormant|salient|random)."
  exit 1
fi

PRETRAIN_ARG=()
if [ -n "${PRETRAIN_CKPT}" ]; then
  PRETRAIN_ARG=(--pretrained_checkpoint "${PRETRAIN_CKPT}")
fi

python3 ./ascent_finetuning.py \
  --dataset_name "${DATASET}" \
  --finetune_mode "${MODE}" \
  --train_data_path "${TRAIN_CSV}" \
  --test_data_path "${TEST_CSV}" \
  --output_dir "${OUT_DIR}" \
  --output_prefix "${PREFIX}" \
  --bert_model "${MODEL}" \
  --max_seq_length "${SEQ_LEN}" \
  --train_epochs "${EPOCHS}" \
  --train_batch_size "${BATCH}" \
  --learning_rate "${LR}" \
  --weight_decay "${WD}" \
  --num_labels "${NUM_LABELS}" \
  --gpus "${GPU_ID}" \
  --neurons_dir "${NEURONS_DIR}" \
  ${SAVE_EACH_EPOCH} \
  --log_level "${LOG_LEVEL}" \
  "${PRETRAIN_ARG[@]}"

echo "Finished ${MODE} fine-tuning on ${DATASET}. Outputs -> ${OUT_DIR}/${DATASET}/${MODE}/"
