from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys
import pickle
import time
import math
from collections import defaultdict

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

import torch.autograd as autograd

from model.bert_util import *

import yaml
from argparse import Namespace

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

def gather_flat_grad(grads):
    views = []
    for p in grads:
        if p.data.is_sparse:
            view = p.data.to_dense().view(-1)
        else:
            view = p.data.view(-1)
        views.append(view)
    return torch.cat(views, 0)

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=False,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--mode",
                        default=None,
                        type=str,
                        required=True,
                        help="Training and testing mode (decides which datasets are used).")
    parser.add_argument("--num_recorded_epoch",
                        default=None,
                        type=int,
                        required=True,
                        help="Number of epochs the trained model went through.")

    ## Other parameters
    parser.add_argument("--cache_dir",
                        default="",
                        type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--trained_model_dir",
                        default="",
                        type=str,
                        help="Where is the fine-tuned (with the cloze-style LM objective) BERT model?")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--freeze_bert',
                        action='store_true',
                        help="Whether to freeze BERT")
    parser.add_argument('--full_bert',
                        action='store_true',
                        help="Whether to use full BERT")
    parser.add_argument('--num_train_samples',
                        type=int,
                        default=-1,
                        help="-1 for full train set, otherwise please specify")
    parser.add_argument('--test_idx',
                        type=int,
                        default=1,
                        help="test index we want to examine")
    parser.add_argument('--start_test_idx',
                        type=int,
                        default=-1,
                        help="when not -1, --test_idx will be disabled")
    parser.add_argument('--end_test_idx',
                        type=int,
                        default=-1,
                        help="when not -1, --test_idx will be disabled")
    parser.add_argument("--influence_metric",
                        default="",
                        type=str,
                        help="standard dot product metric or theta-relative")
    parser.add_argument("--extra_config_file",
                        type=str,
                        default="",
                        help="location of the extra config file containing hyperparameters")
    parser.add_argument('--interpret_for_coord_tuning',
                        action='store_true',
                        help="whether to check the resulting model checkpoing (use for coord tuning)")
    
    args = parser.parse_args()
    
    # load some hyperparameters from config file
    if args.extra_config_file:
        with open(args.extra_config_file, 'r') as extra_config_file:
            hp_config = yaml.safe_load(extra_config_file)
        # add extra hp config to the args namespace
        args_dict = vars(args)
        args_dict.update(hp_config)
        args = Namespace(**args_dict)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        #raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        logger.info("WARNING: Output directory already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Prepare data processor
    if 'synth' in args.mode:
        processor = SynthProcessor()
        label_list = processor.get_labels()
        num_labels = len(label_list)
    elif 'msgs' in args.mode:
        processor = MSGSProcessor()
        label_list = processor.get_labels()
        num_labels = len(label_list)
    else:
        raise ValueError("N/A")

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    
    # Prepare training data
    train_examples = None
    if args.mode == 'synth' or args.mode == 'msgs':
        train_examples = processor.get_train_examples(args.data_dir)
    else:
        raise ValueError("Check your args.mode")

    if args.mode == 'synth':
        train_features = convert_examples_to_features_pretokenized(
            train_examples, label_list, args.max_seq_length, tokenizer)
    else:
        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer)
    logger.info("***** Train set *****")
    logger.info("  Num examples = %d", len(train_examples))
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_id = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_guids = torch.tensor([f.guid for f in train_features], dtype=torch.long)
    all_note = torch.tensor([f.note for f in train_features], dtype=torch.long)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_id, all_guids, all_note)
    train_dataloader = DataLoader(train_data, sampler=SequentialSampler(train_data), batch_size=1)
    
    if args.mode == 'synth' or args.mode == 'msgs': # Han: change this for different purposes
        test_examples = processor.get_train_examples(args.data_dir)
    else:
        raise ValueError("Check your args.mode")
    
    if args.mode == 'synth':
        test_features = convert_examples_to_features_pretokenized(
            test_examples, label_list, args.max_seq_length, tokenizer)
    else:
        test_features = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length, tokenizer)
    logger.info("***** Test set *****")
    logger.info("  Num examples = %d", len(test_examples))
    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_label_id = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    all_guids = torch.tensor([f.guid for f in test_features], dtype=torch.long)
    all_note = torch.tensor([f.note for f in test_features], dtype=torch.long)
    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_id, all_guids, all_note)
    test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=1)
    
    test_idx = args.test_idx
    start_test_idx = args.start_test_idx
    end_test_idx = args.end_test_idx
    
    # final influence dict
    agg_influence_dict = defaultdict(list)

    # for alternative influence metrics
    # cos = nn.CosineSimilarity(dim=0, eps=1e-12)
    cos = nn.CosineSimilarity(dim=1, eps=1e-12) # with ihvp stack
    
    # checkpoint folders name
    ckpt_folder_list = []
    if args.interpret_for_coord_tuning:
        for k in range(1, args.num_recorded_epoch + 1):
            ckpt_folder_list.append(f"epoch_{k}_pre/")
            ckpt_folder_list.append(f"epoch_{k}_post/")
        ckpt_folder_list.append("")
    else:
        for k in range(args.num_recorded_epoch):
            ckpt_folder_list.append(f"epoch_{k}/")
    
    for ckpt_folder in ckpt_folder_list:
        # Prepare model
        model = MyBertForSequenceClassification.from_pretrained(os.path.join(args.trained_model_dir, ckpt_folder), num_labels=num_labels)
        model.to(device)
        param_optimizer = list(model.named_parameters())
        if args.freeze_bert:
            raise ValueError("Disabled for this project")
            frozen = ['bert']
        elif args.full_bert:
            frozen = []
        else:
            raise ValueError("Disabled for this project")
            frozen = ['bert.embeddings.',
                      'bert.encoder.layer.0.',
                      'bert.encoder.layer.1.',
                      'bert.encoder.layer.2.',
                      'bert.encoder.layer.3.',
                      'bert.encoder.layer.4.',
                      'bert.encoder.layer.5.',
                      'bert.encoder.layer.6.',
                      'bert.encoder.layer.7.',
                     ] # *** change here to filter out params we don't want to track ***
        param_influence = []
        for n, p in param_optimizer:
            if (not any(fr in n for fr in frozen)):
                param_influence.append(p)
            elif 'bert.embeddings.word_embeddings.' in n:
                pass # need gradients through embedding layer for computing saliency map
            else:
                p.requires_grad = False
        param_size = 0
        for p in param_influence:
            tmp_p = p.clone().detach()
            param_size += torch.numel(tmp_p)
        del tmp_p
        logger.info("  Parameter size = %d", param_size)
    
        # Calculate influence
        # influence_dict = dict()
        ihvp_dict = dict()

        for tmp_idx, (input_ids, input_mask, segment_ids, label_ids, guids, note) in enumerate(test_dataloader):
            if args.mode == 'synth' or args.mode == 'msgs':
                if args.start_test_idx != -1 and args.end_test_idx != -1:
                    if tmp_idx < args.start_test_idx:
                        continue
                    if tmp_idx > args.end_test_idx:
                        break
                else:
                    if tmp_idx < args.test_idx:
                        continue
                    if tmp_idx > args.test_idx:
                        break
            else:
                raise ValueError('n/a')

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            # influence_dict[tmp_idx] = np.zeros(len(train_examples))

            ######## L_TEST GRADIENT ########
            model.eval()
            model.zero_grad()
            test_loss = model(input_ids, segment_ids, input_mask, label_ids)
            test_grads = autograd.grad(test_loss, param_influence)
            ################

            ihvp_dict[tmp_idx] = gather_flat_grad(test_grads).detach().cpu() # put to CPU to save GPU memory

        # for tmp_idx in ihvp_dict.keys():
        #     ihvp_dict[tmp_idx] = ihvp_dict[tmp_idx].to(args.device)
        ihvp_stack = torch.stack([ihvp_dict[tmp_idx] for tmp_idx in sorted(ihvp_dict.keys())], dim=0).to(args.device)
        ihvp_dict_keys = sorted(ihvp_dict.keys())
        del ihvp_dict
        influence_list = []

        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        for train_idx, (_input_ids, _input_mask, _segment_ids, _label_ids, _guids, _note) in enumerate(tqdm(train_dataloader, desc="Train set index")):
            model.eval() #model.train()
            _input_ids = _input_ids.to(device)
            _input_mask = _input_mask.to(device)
            _segment_ids = _segment_ids.to(device)
            _label_ids = _label_ids.to(device)

            ######## L_TRAIN GRADIENT ########
            model.zero_grad()
            train_loss = model(_input_ids, _segment_ids, _input_mask, _label_ids)
            train_grads = autograd.grad(train_loss, param_influence)
            ################

            with torch.no_grad(): # Han: should be able to speed up greatly here!
                # for tmp_idx in ihvp_dict.keys():
                #     if args.influence_metric == "cosine":
                #         influence_dict[tmp_idx][train_idx] = cos(ihvp_dict[tmp_idx], gather_flat_grad(train_grads)).item()
                #     else:
                #         influence_dict[tmp_idx][train_idx] = torch.dot(ihvp_dict[tmp_idx], gather_flat_grad(train_grads)).item()
                if args.influence_metric == "cosine":
                    influence_list.append(cos(ihvp_stack, torch.unsqueeze(gather_flat_grad(train_grads), 0)).detach().cpu())
                elif args.influence_metric == "dotprod":
                    raise ValueError("check the influence metric")
                    influence_list.append(torch.matmul(ihvp_stack, gather_flat_grad(train_grads)).detach().cpu())
                else:
                    raise ValueError("check the influence metric")
                
        # for k, v in influence_dict.items():
        #     if k not in agg_influence_dict:
        #         agg_influence_dict[k] = v
        #     else:
        #         agg_influence_dict[k] = agg_influence_dict[k] + v
        for test_idx in ihvp_dict_keys:
            agg_influence_dict[test_idx].append(np.zeros(len(train_examples)))
        for train_i, train_i_inf in enumerate(influence_list):
            for test_idx, train_i_inf_on_test_j in zip(ihvp_dict_keys, train_i_inf):
                agg_influence_dict[test_idx][-1][train_i] = train_i_inf_on_test_j.item()
        del influence_list
    
    for k, v in agg_influence_dict.items():
        influence_filename = f"influence_test_idx_{k}.pkl"
        pickle.dump(v, open(os.path.join(args.output_dir, influence_filename), "wb"))

if __name__ == "__main__":
    main()
