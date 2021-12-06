# Synth IT
CUDA_VISIBLE_DEVICES="0" python -W ignore bert_tagger_inftun_no_hvd.py --data_dir="synthetic_outputs/"\
    --bert_model="bert-base-uncased" --max_seq_length=64 --do_train --do_test --train_batch_size=64 --eval_batch_size=64\
    --learning_rate=2e-5 --warmup_proportion=0.1 --do_lower_case --full_bert --mode="synth" --influence_metric="cosine"\
    --start_test_idx=0 --end_test_idx=1499 --num_pos_ex=5 --num_neg_ex=5\
    --extra_config_file="synth_candidate_configs/influence_tuning_config_adamoptim.yml" --alt_optim_plan

# Synth ET
CUDA_VISIBLE_DEVICES="1" python -W ignore bert_tagger_inftun_no_hvd.py --data_dir="synthetic_outputs/"\
    --bert_model="bert-base-uncased" --max_seq_length=64 --do_train --do_test --train_batch_size=64 --eval_batch_size=64\
    --learning_rate=2e-5 --warmup_proportion=0.1 --do_lower_case --full_bert --mode="synth" --influence_metric="cosine"\
    --start_test_idx=0 --end_test_idx=1499 --num_pos_ex=5 --num_neg_ex=5\
    --extra_config_file="synth_candidate_configs/embedding_tuning_config_adamoptim.yml" --alt_optim_plan --alt_embedding_tuning

# MSGS IT
CUDA_VISIBLE_DEVICES="2" python -W ignore bert_tagger_inftun_no_hvd.py --data_dir="msgs_outputs/"\
    --bert_model="bert-base-uncased" --max_seq_length=64 --do_train --do_test --train_batch_size=64 --eval_batch_size=64\
    --learning_rate=2e-5 --warmup_proportion=0.1 --do_lower_case --full_bert --mode="msgs" --influence_metric="cosine"\
    --start_test_idx=0 --end_test_idx=4999\
    --extra_config_file="msgs_candidate_configs/influence_tuning_config_adamoptim.yml" --alt_optim_plan

# MSGS ET
CUDA_VISIBLE_DEVICES="3" python -W ignore bert_tagger_inftun_no_hvd.py --data_dir="msgs_outputs/"\
    --bert_model="bert-base-uncased" --max_seq_length=64 --do_train --do_test --train_batch_size=64 --eval_batch_size=64\
    --learning_rate=2e-5 --warmup_proportion=0.1 --do_lower_case --full_bert --mode="msgs" --influence_metric="cosine"\
    --start_test_idx=0 --end_test_idx=4999\
    --extra_config_file="msgs_candidate_configs/embedding_tuning_config_adamoptim.yml" --alt_optim_plan --alt_embedding_tuning
