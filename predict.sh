#!/usr/bin/env bash
DATA_DIR="data"

#MODEL_DIR="bert-base-chinese"
MODEL_DIR="./prev_model/RoBERTa_zh_Large"
OUTPUT_DIR="./output/save_dict/"
PREDICT_DIR="./data/"
MAX_LENGTH=256

python run.py \
    --data_dir=${DATA_DIR} \
    --model_dir=${MODEL_DIR} \
    --output_dir=${OUTPUT_DIR} \
    --max_length=${MAX_LENGTH} \
    --batch_size=16 \
    --do_test \
    --test_batch=100 \
    --epochs=100 \
    --seed=2021 \
    --threshold=0.7 \
    --train_dataset=/train_triple_aug.jsonl \
    --device=cuda:0