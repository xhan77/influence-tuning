#!/usr/bin/bash

trap "kill 0" EXIT

# ngpu=$1 # first argument
ngpu=4 # first argument through hardcoding (for sbatch)

# for RTX 3090, processing 15 test examples in one run is possible
starti=("0" "10" "20" "30")
endi=("9" "19" "29" "39")

#### NO INFLUENCE TUNING ####

# synth vanilla
for i in $(seq 0 1 $((ngpu-1)))
do
CUDA_VISIBLE_DEVICES="$i" python bert_trackin.py --data_dir="synthetic_outputs/" --mode="synth"\
    --bert_model="bert-base-uncased" --max_seq_length=64 --do_lower_case --full_bert --influence_metric="cosine"\
    --train_batch_size=1 --eval_batch_size=1 --start_test_idx="${starti[i]}" --end_test_idx="${endi[i]}"\
    --output_dir="influence_logs/candidate_tagger_outputs_synth/" --trained_model_dir="model/candidate_tagger_outputs_synth/"\
    --num_recorded_epoch=10 &
done
wait

# synth no spur
for i in $(seq 0 1 $((ngpu-1)))
do
CUDA_VISIBLE_DEVICES="$i" python bert_trackin.py --data_dir="synthetic_outputs/no_spur/" --mode="synth"\
    --bert_model="bert-base-uncased" --max_seq_length=64 --do_lower_case --full_bert --influence_metric="cosine"\
    --train_batch_size=1 --eval_batch_size=1 --start_test_idx="${starti[i]}" --end_test_idx="${endi[i]}"\
    --output_dir="influence_logs/candidate_tagger_outputs_synth_no_spur/" --trained_model_dir="model/candidate_tagger_outputs_synth_no_spur/"\
    --num_recorded_epoch=10 &
done
wait

# msgs vanilla
for i in $(seq 0 1 $((ngpu-1)))
do
CUDA_VISIBLE_DEVICES="$i" python bert_trackin.py --data_dir="msgs_outputs/" --mode="msgs"\
    --bert_model="bert-base-uncased" --max_seq_length=64 --do_lower_case --full_bert --influence_metric="cosine"\
    --train_batch_size=1 --eval_batch_size=1 --start_test_idx="${starti[i]}" --end_test_idx="${endi[i]}"\
    --output_dir="influence_logs/candidate_tagger_outputs_msgs/" --trained_model_dir="model/candidate_tagger_outputs_msgs/"\
    --num_recorded_epoch=3 &
done
wait

#### INFLUENCE TUNING, remember to apply --interpret_for_coord_tuning flag if applicable ####

# synth IT
for i in $(seq 0 1 $((ngpu-1)))
do
CUDA_VISIBLE_DEVICES="$i" python bert_trackin.py --data_dir="synthetic_outputs/" --mode="synth"\
    --bert_model="bert-base-uncased" --max_seq_length=64 --do_lower_case --full_bert --influence_metric="cosine"\
    --train_batch_size=1 --eval_batch_size=1 --start_test_idx="${starti[i]}" --end_test_idx="${endi[i]}"\
    --output_dir="influence_logs/candidate_tagger_outputs_synth_IT_bolt/" --trained_model_dir="candidate_tagger_outputs_synth_IT_bolt/"\
    --num_recorded_epoch=4 --interpret_for_coord_tuning &
done
wait

# synth ET
for i in $(seq 0 1 $((ngpu-1)))
do
CUDA_VISIBLE_DEVICES="$i" python bert_trackin.py --data_dir="synthetic_outputs/" --mode="synth"\
    --bert_model="bert-base-uncased" --max_seq_length=64 --do_lower_case --full_bert --influence_metric="cosine"\
    --train_batch_size=1 --eval_batch_size=1 --start_test_idx="${starti[i]}" --end_test_idx="${endi[i]}"\
    --output_dir="influence_logs/candidate_tagger_outputs_synth_ET_bolt/" --trained_model_dir="candidate_tagger_outputs_synth_ET_bolt/"\
    --num_recorded_epoch=4 --interpret_for_coord_tuning &
done
wait

# msgs IT
for i in $(seq 0 1 $((ngpu-1)))
do
CUDA_VISIBLE_DEVICES="$i" python bert_trackin.py --data_dir="msgs_outputs/" --mode="msgs"\
    --bert_model="bert-base-uncased" --max_seq_length=64 --do_lower_case --full_bert --influence_metric="cosine"\
    --train_batch_size=1 --eval_batch_size=1 --start_test_idx="${starti[i]}" --end_test_idx="${endi[i]}"\
    --output_dir="influence_logs/candidate_tagger_outputs_msgs_IT_bolt/" --trained_model_dir="candidate_tagger_outputs_msgs_IT_bolt/"\
    --num_recorded_epoch=4 --interpret_for_coord_tuning &
done
wait

# msgs ET
for i in $(seq 0 1 $((ngpu-1)))
do
CUDA_VISIBLE_DEVICES="$i" python bert_trackin.py --data_dir="msgs_outputs/" --mode="msgs"\
    --bert_model="bert-base-uncased" --max_seq_length=64 --do_lower_case --full_bert --influence_metric="cosine"\
    --train_batch_size=1 --eval_batch_size=1 --start_test_idx="${starti[i]}" --end_test_idx="${endi[i]}"\
    --output_dir="influence_logs/candidate_tagger_outputs_msgs_ET_bolt/" --trained_model_dir="candidate_tagger_outputs_msgs_ET_bolt/"\
    --num_recorded_epoch=8 --interpret_for_coord_tuning &
done
wait
