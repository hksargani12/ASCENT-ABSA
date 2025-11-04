#!/usr/bin/env bash

# ASCENT Neuron Analysis Runner
# Usage:
# bash run_ascent_neuron_analysis.sh

# Example datasets:
# DATA_PATH="/data/neurons_lap14.sentiment.jsonl"
# DATA_PATH="/data/neurons_res14.sentiment.jsonl"
# DATA_PATH="/data/neurons_mams.sentiment.jsonl"

DATA_PATH="/data/neurons_mams.sentiment.jsonl"
OUTPUT_DIR="./outputs_neuron_analysis"
DATASET_NAME="mams"

python ascent_neuron_analysis.py \
  --input_file "$DATA_PATH" \
  --dataset_name "$DATASET_NAME" \
  --output_dir "$OUTPUT_DIR" \
  --use_ig_gold
