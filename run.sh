#!/usr/bin/env bash
DATA_DIR="data"

#MODEL_DIR="bert-base-chinese"
MODEL_DIR="./prev_model/roberta"
OUTPUT_DIR="./output/save_dict/"
PREDICT_DIR="./data/"
MAX_LENGTH=256

echo "Start running"

python run.py \
    --data_dir=${DATA_DIR} \
    --model_dir=${MODEL_DIR} \
    --output_dir=${OUTPUT_DIR} \
    --do_train \
    --max_length=${MAX_LENGTH} \
    --batch_size=64 \
    --test_batch=100 \
    --epochs=100 \
    --seed=2021 \
    --threshold=0.5 \
    --train_dataset=/train_triple.jsonl \
    --device=cuda:1




