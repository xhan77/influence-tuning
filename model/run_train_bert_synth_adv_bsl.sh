#!/usr/bin/bash

trap "kill 0" EXIT

# ngpu=$1 # first argument
ngpu=5 # first argument through hardcoding (for sbatch)

seeds=("2021" "2022" "2023" "2024" "2025")

# domain adversarial training
for i in $(seq 0 1 $((ngpu-1)))
do
CUDA_VISIBLE_DEVICES='0' python -W ignore bert_tagger.py --data_dir="../synthetic_outputs/"\
    --bert_model="bert-base-uncased" --max_seq_length=64 --do_train --do_test --train_batch_size=64 --eval_batch_size=64\
    --learning_rate=5e-5 --num_train_epochs=10 --warmup_proportion=0.1 --seed=${seeds[i]}\
    --output_dir="tagger_outputs_synth_dann_alt_bsl_lr5e-5/e10_l0.1_s${seeds[i]}/" --do_lower_case --full_bert --mode="synth"\
    --adv_lambda_if_applying_dann=0.1 --no_epoch_checkpoint_saving
rm "tagger_outputs_synth_dann_alt_bsl_lr5e-5/e10_l0.1_s${seeds[i]}/pytorch_model.bin"

CUDA_VISIBLE_DEVICES='0' python -W ignore bert_tagger.py --data_dir="../synthetic_outputs/"\
    --bert_model="bert-base-uncased" --max_seq_length=64 --do_train --do_test --train_batch_size=64 --eval_batch_size=64\
    --learning_rate=5e-5 --num_train_epochs=10 --warmup_proportion=0.1 --seed=${seeds[i]}\
    --output_dir="tagger_outputs_synth_dann_alt_bsl_lr5e-5/e10_l0.3_s${seeds[i]}/" --do_lower_case --full_bert --mode="synth"\
    --adv_lambda_if_applying_dann=0.3 --no_epoch_checkpoint_saving
rm "tagger_outputs_synth_dann_alt_bsl_lr5e-5/e10_l0.3_s${seeds[i]}/pytorch_model.bin"
    
CUDA_VISIBLE_DEVICES='0' python -W ignore bert_tagger.py --data_dir="../synthetic_outputs/"\
    --bert_model="bert-base-uncased" --max_seq_length=64 --do_train --do_test --train_batch_size=64 --eval_batch_size=64\
    --learning_rate=5e-5 --num_train_epochs=10 --warmup_proportion=0.1 --seed=${seeds[i]}\
    --output_dir="tagger_outputs_synth_dann_alt_bsl_lr5e-5/e10_l1.0_s${seeds[i]}/" --do_lower_case --full_bert --mode="synth"\
    --adv_lambda_if_applying_dann=1.0 --no_epoch_checkpoint_saving
rm "tagger_outputs_synth_dann_alt_bsl_lr5e-5/e10_l1.0_s${seeds[i]}/pytorch_model.bin"
    
CUDA_VISIBLE_DEVICES='0' python -W ignore bert_tagger.py --data_dir="../synthetic_outputs/"\
    --bert_model="bert-base-uncased" --max_seq_length=64 --do_train --do_test --train_batch_size=64 --eval_batch_size=64\
    --learning_rate=5e-5 --num_train_epochs=10 --warmup_proportion=0.1 --seed=${seeds[i]}\
    --output_dir="tagger_outputs_synth_dann_alt_bsl_lr5e-5/e10_l3.0_s${seeds[i]}/" --do_lower_case --full_bert --mode="synth"\
    --adv_lambda_if_applying_dann=3.0 --no_epoch_checkpoint_saving
rm "tagger_outputs_synth_dann_alt_bsl_lr5e-5/e10_l3.0_s${seeds[i]}/pytorch_model.bin"
    
CUDA_VISIBLE_DEVICES='0' python -W ignore bert_tagger.py --data_dir="../synthetic_outputs/"\
    --bert_model="bert-base-uncased" --max_seq_length=64 --do_train --do_test --train_batch_size=64 --eval_batch_size=64\
    --learning_rate=5e-5 --num_train_epochs=20 --warmup_proportion=0.1 --seed=${seeds[i]}\
    --output_dir="tagger_outputs_synth_dann_alt_bsl_lr5e-5/e20_l0.1_s${seeds[i]}/" --do_lower_case --full_bert --mode="synth"\
    --adv_lambda_if_applying_dann=0.1 --no_epoch_checkpoint_saving
rm "tagger_outputs_synth_dann_alt_bsl_lr5e-5/e20_l0.1_s${seeds[i]}/pytorch_model.bin"
    
CUDA_VISIBLE_DEVICES='0' python -W ignore bert_tagger.py --data_dir="../synthetic_outputs/"\
    --bert_model="bert-base-uncased" --max_seq_length=64 --do_train --do_test --train_batch_size=64 --eval_batch_size=64\
    --learning_rate=5e-5 --num_train_epochs=20 --warmup_proportion=0.1 --seed=${seeds[i]}\
    --output_dir="tagger_outputs_synth_dann_alt_bsl_lr5e-5/e20_l0.3_s${seeds[i]}/" --do_lower_case --full_bert --mode="synth"\
    --adv_lambda_if_applying_dann=0.3 --no_epoch_checkpoint_saving
rm "tagger_outputs_synth_dann_alt_bsl_lr5e-5/e20_l0.3_s${seeds[i]}/pytorch_model.bin"
    
CUDA_VISIBLE_DEVICES='0' python -W ignore bert_tagger.py --data_dir="../synthetic_outputs/"\
    --bert_model="bert-base-uncased" --max_seq_length=64 --do_train --do_test --train_batch_size=64 --eval_batch_size=64\
    --learning_rate=5e-5 --num_train_epochs=20 --warmup_proportion=0.1 --seed=${seeds[i]}\
    --output_dir="tagger_outputs_synth_dann_alt_bsl_lr5e-5/e20_l1.0_s${seeds[i]}/" --do_lower_case --full_bert --mode="synth"\
    --adv_lambda_if_applying_dann=1.0 --no_epoch_checkpoint_saving
rm "tagger_outputs_synth_dann_alt_bsl_lr5e-5/e20_l1.0_s${seeds[i]}/pytorch_model.bin"
    
CUDA_VISIBLE_DEVICES='0' python -W ignore bert_tagger.py --data_dir="../synthetic_outputs/"\
    --bert_model="bert-base-uncased" --max_seq_length=64 --do_train --do_test --train_batch_size=64 --eval_batch_size=64\
    --learning_rate=5e-5 --num_train_epochs=20 --warmup_proportion=0.1 --seed=${seeds[i]}\
    --output_dir="tagger_outputs_synth_dann_alt_bsl_lr5e-5/e20_l3.0_s${seeds[i]}/" --do_lower_case --full_bert --mode="synth"\
    --adv_lambda_if_applying_dann=3.0 --no_epoch_checkpoint_saving
rm "tagger_outputs_synth_dann_alt_bsl_lr5e-5/e20_l3.0_s${seeds[i]}/pytorch_model.bin"

CUDA_VISIBLE_DEVICES='0' python -W ignore bert_tagger.py --data_dir="../synthetic_outputs/"\
    --bert_model="bert-base-uncased" --max_seq_length=64 --do_train --do_test --train_batch_size=64 --eval_batch_size=64\
    --learning_rate=5e-5 --num_train_epochs=40 --warmup_proportion=0.1 --seed=${seeds[i]}\
    --output_dir="tagger_outputs_synth_dann_alt_bsl_lr5e-5/e40_l0.1_s${seeds[i]}/" --do_lower_case --full_bert --mode="synth"\
    --adv_lambda_if_applying_dann=0.1 --no_epoch_checkpoint_saving
rm "tagger_outputs_synth_dann_alt_bsl_lr5e-5/e40_l0.1_s${seeds[i]}/pytorch_model.bin"
    
CUDA_VISIBLE_DEVICES='0' python -W ignore bert_tagger.py --data_dir="../synthetic_outputs/"\
    --bert_model="bert-base-uncased" --max_seq_length=64 --do_train --do_test --train_batch_size=64 --eval_batch_size=64\
    --learning_rate=5e-5 --num_train_epochs=40 --warmup_proportion=0.1 --seed=${seeds[i]}\
    --output_dir="tagger_outputs_synth_dann_alt_bsl_lr5e-5/e40_l0.3_s${seeds[i]}/" --do_lower_case --full_bert --mode="synth"\
    --adv_lambda_if_applying_dann=0.3 --no_epoch_checkpoint_saving
rm "tagger_outputs_synth_dann_alt_bsl_lr5e-5/e40_l0.3_s${seeds[i]}/pytorch_model.bin"
    
CUDA_VISIBLE_DEVICES='0' python -W ignore bert_tagger.py --data_dir="../synthetic_outputs/"\
    --bert_model="bert-base-uncased" --max_seq_length=64 --do_train --do_test --train_batch_size=64 --eval_batch_size=64\
    --learning_rate=5e-5 --num_train_epochs=40 --warmup_proportion=0.1 --seed=${seeds[i]}\
    --output_dir="tagger_outputs_synth_dann_alt_bsl_lr5e-5/e40_l1.0_s${seeds[i]}/" --do_lower_case --full_bert --mode="synth"\
    --adv_lambda_if_applying_dann=1.0 --no_epoch_checkpoint_saving
rm "tagger_outputs_synth_dann_alt_bsl_lr5e-5/e40_l1.0_s${seeds[i]}/pytorch_model.bin"
    
CUDA_VISIBLE_DEVICES='0' python -W ignore bert_tagger.py --data_dir="../synthetic_outputs/"\
    --bert_model="bert-base-uncased" --max_seq_length=64 --do_train --do_test --train_batch_size=64 --eval_batch_size=64\
    --learning_rate=5e-5 --num_train_epochs=40 --warmup_proportion=0.1 --seed=${seeds[i]}\
    --output_dir="tagger_outputs_synth_dann_alt_bsl_lr5e-5/e40_l3.0_s${seeds[i]}/" --do_lower_case --full_bert --mode="synth"\
    --adv_lambda_if_applying_dann=3.0 --no_epoch_checkpoint_saving
rm "tagger_outputs_synth_dann_alt_bsl_lr5e-5/e40_l3.0_s${seeds[i]}/pytorch_model.bin"
done
