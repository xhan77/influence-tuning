#!/usr/bin/bash

trap "kill 0" EXIT

# ngpu=$1 # first argument
ngpu=5 # first argument through hardcoding (for sbatch)

seeds=("2021" "2022" "2023" "2024" "2025")

# label tuning with spur data
for i in $(seq 0 1 $((ngpu-1)))
do
CUDA_VISIBLE_DEVICES='0' python -W ignore bert_tagger.py --data_dir="../msgs_outputs/"\
    --bert_model="bert-base-uncased" --max_seq_length=64 --do_train --do_test --train_batch_size=64 --eval_batch_size=64\
    --learning_rate=2e-5 --num_train_epochs=3 --warmup_proportion=0.1 --seed=${seeds[i]}\
    --output_dir="tagger_outputs_msgs/e3_s${seeds[i]}/" --do_lower_case --full_bert --mode="msgs" --no_epoch_checkpoint_saving
done
