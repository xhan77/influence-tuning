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

from transformers import AdamW, get_linear_schedule_with_warmup

import torch.autograd as autograd

from model.bert_util import *

# import horovod.torch as hvd
import yaml
from argparse import Namespace

try:
    import turibolt as bolt
except ImportError:
    bolt = None
bolt_config = None

# # init Horovod
# hvd.init()

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

def hv(loss, model_params, v):
    grad = autograd.grad(loss, model_params, create_graph=True, retain_graph=True)
#     import pdb; pdb.set_trace() # [(n, g.requires_grad) for g, (n, p) in zip(grad, args.model.named_parameters())]
    Hv = autograd.grad(grad, model_params, grad_outputs=v)
    return Hv

def reset_bertadam_state(optimizer):
    for group in optimizer.param_groups:
        for p in group['params']:
            state = optimizer.state[p]
            state['next_m'] = torch.zeros_like(p.data)
            state['next_v'] = torch.zeros_like(p.data)


def influence_tuning(model, args, train_dataloader, test_dataloader, train_examples, start_test_idx, end_test_idx,
                     param_influence, optimizer, agg_iloss_grads_buffer, iloss_logging_dict):
    it_start_time = time.time()
    te_drop_idx_list = random.sample(list(range(start_test_idx, end_test_idx + 1)),
                                     int(args.influence_tuning_instance_dropout * (end_test_idx - start_test_idx + 1)))
    for epoch in range(int(args.influence_tuning_epochs)):
        for tmp_idx, (te_input_ids, te_input_mask, te_segment_ids, te_label_ids, te_guids, te_note) in enumerate(test_dataloader):
            if args.mode == 'synth' or args.mode == 'msgs':
                if tmp_idx >= args.start_test_idx and tmp_idx <= args.end_test_idx:
                    # test_guid = guids.item()
                    test_note = te_note[0][1].item() # we need the spur information
                else:
                    continue
            else:
                raise ValueError("N/A")

            if tmp_idx in te_drop_idx_list:
                continue

            te_input_ids = te_input_ids.to(args.device)
            te_input_mask = te_input_mask.to(args.device)
            te_segment_ids = te_segment_ids.to(args.device)
            te_label_ids = te_label_ids.to(args.device)

            ######## L_TEST GRADIENT ########
            model.eval()
            model.zero_grad()
            test_loss = model(te_input_ids, te_segment_ids, te_input_mask, te_label_ids)
            test_grads = autograd.grad(test_loss, param_influence)
            test_grads_norm = torch.norm(gather_flat_grad(test_grads), p=2).item()
#             normed_test_grads = tuple([e / test_grads_norm for e in test_grads])
            ################

            # Han: setup pos and neg ex information
            if args.mode == 'synth' or args.mode == 'msgs':
                i_score_group_a, i_score_group_b = [], [] # I_A and I_B group to be averaged later
                grad_i_group_a, grad_i_group_a_cnt, grad_i_group_b, grad_i_group_b_cnt = None, 0, None, 0 # nabla I_a and nable I_B group
                
                full_pos_idx_list, full_neg_idx_list = [], [] # both pos and neg need to have the same label as test, pos means same spur as test
                for train_idx, (_, _, _, tr_label_ids, _, tr_note) in enumerate(train_dataloader):
                    if te_label_ids.item() != tr_label_ids.item():
                        continue
                    if tr_note[0][1].item() == test_note:
                        full_pos_idx_list.append(train_idx)
                    else:
                        full_neg_idx_list.append(train_idx)
#                 print("check same and diff note examples size:", len(full_pos_idx_list), len(full_neg_idx_list))
                pos_idx_list = random.sample(full_pos_idx_list, args.num_pos_ex)
                neg_idx_list = random.sample(full_neg_idx_list, args.num_neg_ex)

                if args.influence_tuning_invariant_epoch: # Han: debugging sanity check for influence tuning overfitting
                    pos_idx_list, neg_idx_list = full_pos_idx_list[:args.num_pos_ex], full_neg_idx_list[:args.num_neg_ex]
            else:
                raise ValueError("N/A")

            # start looping through training examples for a single test example
            for train_idx, (tr_input_ids, tr_input_mask, tr_segment_ids, tr_label_ids, _, _) in enumerate(train_dataloader):
                # Han: on which training examples we should provide supervision (need modification later)
                if args.mode == 'synth' or args.mode == 'msgs':
                    if train_idx in pos_idx_list:
                        under_pos_ex_mode = True
                    elif train_idx in neg_idx_list:
                        under_pos_ex_mode = False
                    else:
                        continue
                else:
                    raise ValueError("N/A")

                model.eval() # model.train()
                tr_input_ids = tr_input_ids.to(args.device)
                tr_input_mask = tr_input_mask.to(args.device)
                tr_segment_ids = tr_segment_ids.to(args.device)
                tr_label_ids = tr_label_ids.to(args.device)

                ######## L_TRAIN GRADIENT ########
                model.zero_grad()
                train_loss = model(tr_input_ids, tr_segment_ids, tr_input_mask, tr_label_ids)
                train_grads = autograd.grad(train_loss, param_influence)
                train_grads_norm = torch.norm(gather_flat_grad(train_grads), p=2).item()
#                 normed_train_grads = tuple([e / train_grads_norm for e in train_grads])
                ################

                with torch.no_grad(): # Han: should be able to speed up, and also explore other metrics as well (e.g., L2 distance)
                    if args.mode == "synth" or args.mode == 'msgs':
                        if args.influence_metric == "cosine":
                            i_score = nn.functional.cosine_similarity(gather_flat_grad(test_grads),\
                                gather_flat_grad(train_grads), dim=0, eps=1e-12).item()
                            if under_pos_ex_mode:
                                # save i_score to I_A
                                i_score_group_a.append(i_score)
                            else:
                                # save i_score to I_B
                                i_score_group_b.append(i_score)
                        else:
                            raise ValueError("N/A for this version")
                    else:
                        raise ValueError("N/A")
                        
                # dotprod
                grad_dotprod = torch.dot(gather_flat_grad(test_grads), gather_flat_grad(train_grads)).item()

                # HVP 1
                model.zero_grad()
                test_loss = model(te_input_ids, te_segment_ids, te_input_mask, te_label_ids)
                hvp_1 = hv(test_loss, param_influence, train_grads)

                # HVP 2
                model.zero_grad()
                train_loss = model(tr_input_ids, tr_segment_ids, tr_input_mask, tr_label_ids)
                hvp_2 = hv(train_loss, param_influence, test_grads)
                
                # HVP 3
                model.zero_grad()
                test_loss = model(te_input_ids, te_segment_ids, te_input_mask, te_label_ids)
                hvp_3 = hv(test_loss, param_influence, test_grads)
                
                # HVP 4
                model.zero_grad()
                train_loss = model(tr_input_ids, tr_segment_ids, tr_input_mask, tr_label_ids)
                hvp_4 = hv(train_loss, param_influence, train_grads)

                # aggregation
                if args.mode == "synth" or args.mode == 'msgs': # synth is aggregating within groups first and then updating the optimizer grad group
                    influence_loss_grads = tuple([e1 / (test_grads_norm * train_grads_norm)
                                                  + e2 / (test_grads_norm * train_grads_norm)
                                                  - e3 * grad_dotprod / ((test_grads_norm ** 3) * train_grads_norm)
                                                  - e4 * grad_dotprod / (test_grads_norm * (train_grads_norm ** 3))
                                                  for e1, e2, e3, e4 in zip(hvp_1, hvp_2, hvp_3, hvp_4)])
                    del hvp_1
                    del hvp_2
                    del hvp_3
                    del hvp_4
                    if under_pos_ex_mode:
                        if grad_i_group_a is None:
                            grad_i_group_a = influence_loss_grads
                            grad_i_group_a_cnt = 1
                        else:
                            grad_i_group_a = tuple([e1 + e2 for e1, e2 in zip(grad_i_group_a, influence_loss_grads)])
                            grad_i_group_a_cnt += 1
                        del influence_loss_grads
                    else:
                        if grad_i_group_b is None:
                            grad_i_group_b = influence_loss_grads
                            grad_i_group_b_cnt = 1
                        else:
                            grad_i_group_b = tuple([e1 + e2 for e1, e2 in zip(grad_i_group_b, influence_loss_grads)])
                            grad_i_group_b_cnt += 1
                        del influence_loss_grads
                    
                    if grad_i_group_a_cnt + grad_i_group_b_cnt == args.num_pos_ex + args.num_neg_ex: # I_A and I_B finish collecting
                        mean_i_score_group_a = np.mean(i_score_group_a)
                        mean_i_score_group_b = np.mean(i_score_group_b)
                        diff_influence_loss_grads = tuple([2 * (mean_i_score_group_a - mean_i_score_group_b)
                                                           * (e1 / grad_i_group_a_cnt - e2 / grad_i_group_b_cnt)
                                                           for e1, e2 in zip(grad_i_group_a, grad_i_group_b)])
                        del grad_i_group_a
                        del grad_i_group_b
                        
                        if agg_iloss_grads_buffer[0] is None:
                            agg_iloss_grads_buffer[0] = diff_influence_loss_grads
                            agg_iloss_grads_buffer[1] = 1
                        else:
                            agg_iloss_grads_buffer[0] = tuple([e1 + e2 for e1, e2 in zip(agg_iloss_grads_buffer[0], diff_influence_loss_grads)])
                            agg_iloss_grads_buffer[1] += 1
                        del diff_influence_loss_grads
                        
                        influence_loss = mean_i_score_group_a - mean_i_score_group_b
                        iloss_logging_dict[tmp_idx].append(influence_loss) # Han: logging the loss
                else:
                    raise ValueError("N/A")

                # optimize
                if agg_iloss_grads_buffer[1] > 0 and agg_iloss_grads_buffer[1] % args.influence_tuning_batch_size == 0:
                    model.train()
                    model.zero_grad()
                    for p, g in zip(param_influence, agg_iloss_grads_buffer[0]):
                        p.grad = g / agg_iloss_grads_buffer[1] * 1 # Han: 1 or -1 for debugging purpose
                        # print(torch.linalg.norm(p.grad, 1), torch.numel(p.grad))
                    optimizer.step()
                    model.zero_grad()
                    agg_iloss_grads_buffer[0] = None
                    agg_iloss_grads_buffer[1] = 0
        logger.info(f"  sampled influence epoch loss = {np.mean([e[-1] for e in iloss_logging_dict.values()])}")

    logger.info(f"  influence tuning time = {time.time() - it_start_time} sec")
    iloss_logging_dict.clear()

    
def embedding_tuning(model, args, train_dataloader, test_dataloader, train_examples, start_test_idx, end_test_idx,
                     param_influence, optimizer, agg_iloss_grads_buffer, iloss_logging_dict):
    it_start_time = time.time()
    te_drop_idx_list = random.sample(list(range(start_test_idx, end_test_idx + 1)),
                                     int(args.influence_tuning_instance_dropout * (end_test_idx - start_test_idx + 1)))
    for epoch in range(int(args.influence_tuning_epochs)):
        for tmp_idx, (te_input_ids, te_input_mask, te_segment_ids, te_label_ids, te_guids, te_note) in enumerate(test_dataloader):
            if args.mode == 'synth' or args.mode == 'msgs':
                if tmp_idx >= args.start_test_idx and tmp_idx <= args.end_test_idx:
                    # test_guid = guids.item()
                    test_note = te_note[0][1].item() # we need the spur information
                else:
                    continue
            else:
                raise ValueError("N/A")

            if tmp_idx in te_drop_idx_list:
                continue

            te_input_ids = te_input_ids.to(args.device)
            te_input_mask = te_input_mask.to(args.device)
            te_segment_ids = te_segment_ids.to(args.device)
            te_label_ids = te_label_ids.to(args.device)

            # Han: setup pos and neg ex information
            if args.mode == 'synth' or args.mode == 'msgs':
                i_score_group_a, i_score_group_b = [], [] # I_A and I_B group to be averaged later
                grad_i_group_a, grad_i_group_a_cnt, grad_i_group_b, grad_i_group_b_cnt = None, 0, None, 0 # nabla I_a and nable I_B group
                
                full_pos_idx_list, full_neg_idx_list = [], [] # both pos and neg need to have the same label as test, pos means same spur as test
                for train_idx, (_, _, _, tr_label_ids, _, tr_note) in enumerate(train_dataloader):
                    if te_label_ids.item() != tr_label_ids.item():
                        continue
                    if tr_note[0][1].item() == test_note:
                        full_pos_idx_list.append(train_idx)
                    else:
                        full_neg_idx_list.append(train_idx)
#                 print("check same and diff note examples size:", len(full_pos_idx_list), len(full_neg_idx_list))
                pos_idx_list = random.sample(full_pos_idx_list, args.num_pos_ex)
                neg_idx_list = random.sample(full_neg_idx_list, args.num_neg_ex)

                if args.influence_tuning_invariant_epoch: # Han: debugging sanity check for influence tuning overfitting
                    pos_idx_list, neg_idx_list = full_pos_idx_list[:args.num_pos_ex], full_neg_idx_list[:args.num_neg_ex]
            else:
                raise ValueError("N/A")

            # start looping through training examples for a single test example
            for train_idx, (tr_input_ids, tr_input_mask, tr_segment_ids, tr_label_ids, _, _) in enumerate(train_dataloader):
                # Han: on which training examples we should provide supervision (need modification later)
                if args.mode == 'synth' or args.mode == 'msgs':
                    if train_idx in pos_idx_list:
                        under_pos_ex_mode = True
                    elif train_idx in neg_idx_list:
                        under_pos_ex_mode = False
                    else:
                        continue
                else:
                    raise ValueError("N/A")

                model.eval() # model.train()
                tr_input_ids = tr_input_ids.to(args.device)
                tr_input_mask = tr_input_mask.to(args.device)
                tr_segment_ids = tr_segment_ids.to(args.device)
                tr_label_ids = tr_label_ids.to(args.device)
                
                # computing embedding distance
                model.zero_grad()
                _, test_embeds = model.bert(te_input_ids, te_segment_ids, te_input_mask, output_all_encoded_layers=False)
                _, train_embeds = model.bert(tr_input_ids, tr_segment_ids, tr_input_mask, output_all_encoded_layers=False)
                embedding_influence_loss = nn.functional.cosine_similarity(test_embeds.view(-1), train_embeds.view(-1), dim=0, eps=1e-12)
                influence_loss_grads = autograd.grad(embedding_influence_loss, param_influence)

                with torch.no_grad(): # Han: should be able to speed up, and also explore other metrics as well (e.g., L2 distance)
                    if args.mode == "synth" or args.mode == 'msgs':
                        if args.influence_metric == "cosine":
                            i_score = nn.functional.cosine_similarity(test_embeds.view(-1), train_embeds.view(-1), dim=0, eps=1e-12).item()
                            if under_pos_ex_mode:
                                # save i_score to I_A
                                i_score_group_a.append(i_score)
                            else:
                                # save i_score to I_B
                                i_score_group_b.append(i_score)
                        else:
                            raise ValueError("N/A for this version")
                    else:
                        raise ValueError("N/A")

                # aggregation
                if args.mode == "synth" or args.mode == 'msgs': # synth is aggregating within groups first and then updating the optimizer grad group
                    if under_pos_ex_mode:
                        if grad_i_group_a is None:
                            grad_i_group_a = influence_loss_grads
                            grad_i_group_a_cnt = 1
                        else:
                            grad_i_group_a = tuple([e1 + e2 for e1, e2 in zip(grad_i_group_a, influence_loss_grads)])
                            grad_i_group_a_cnt += 1
                        del influence_loss_grads
                    else:
                        if grad_i_group_b is None:
                            grad_i_group_b = influence_loss_grads
                            grad_i_group_b_cnt = 1
                        else:
                            grad_i_group_b = tuple([e1 + e2 for e1, e2 in zip(grad_i_group_b, influence_loss_grads)])
                            grad_i_group_b_cnt += 1
                        del influence_loss_grads
                    
                    if grad_i_group_a_cnt + grad_i_group_b_cnt == args.num_pos_ex + args.num_neg_ex: # I_A and I_B finish collecting
                        mean_i_score_group_a = np.mean(i_score_group_a)
                        mean_i_score_group_b = np.mean(i_score_group_b)
                        diff_influence_loss_grads = tuple([2 * (mean_i_score_group_a - mean_i_score_group_b)
                                                           * (e1 / grad_i_group_a_cnt - e2 / grad_i_group_b_cnt)
                                                           for e1, e2 in zip(grad_i_group_a, grad_i_group_b)])
                        del grad_i_group_a
                        del grad_i_group_b
                        
                        if agg_iloss_grads_buffer[0] is None:
                            agg_iloss_grads_buffer[0] = diff_influence_loss_grads
                            agg_iloss_grads_buffer[1] = 1
                        else:
                            agg_iloss_grads_buffer[0] = tuple([e1 + e2 for e1, e2 in zip(agg_iloss_grads_buffer[0], diff_influence_loss_grads)])
                            agg_iloss_grads_buffer[1] += 1
                        del diff_influence_loss_grads
                        
                        influence_loss = mean_i_score_group_a - mean_i_score_group_b
                        iloss_logging_dict[tmp_idx].append(influence_loss) # Han: logging the loss
                else:
                    raise ValueError("N/A")

                # optimize
                if agg_iloss_grads_buffer[1] > 0 and agg_iloss_grads_buffer[1] % args.influence_tuning_batch_size == 0:
                    model.train()
                    model.zero_grad()
                    for p, g in zip(param_influence, agg_iloss_grads_buffer[0]):
                        p.grad = g / agg_iloss_grads_buffer[1] * 1 # Han: 1 or -1 for debugging purpose
                        # print(torch.linalg.norm(p.grad, 1), torch.numel(p.grad))
                    optimizer.step()
                    model.zero_grad()
                    agg_iloss_grads_buffer[0] = None
                    agg_iloss_grads_buffer[1] = 0
        logger.info(f"  sampled influence epoch loss = {np.mean([e[-1] for e in iloss_logging_dict.values()])}")

    logger.info(f"  influence tuning time = {time.time() - it_start_time} sec")
    iloss_logging_dict.clear()
    
    
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
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run eval on the test set.")
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
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
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
    parser.add_argument('--correction_dir',
                        type=str,
                        default="",
                        help="directory for the correction file")
    parser.add_argument('--correction_size',
                        type=int,
                        default=0,
                        help="correct how many examples?")
    parser.add_argument('--coord_interval',
                        type=int,
                        default=100,
                        help="after how many normal steps should influence tuning kicks in?")
    parser.add_argument("--influence_metric",
                        default="",
                        type=str,
                        help="standard dot product metric or theta-relative")
    parser.add_argument('--only_check_last_epoch',
                        action='store_true',
                        help="check interpretations only at the last epoch instead of averaging over all")
    parser.add_argument("--num_pos_ex",
                        default=5,
                        type=int)
    parser.add_argument("--num_neg_ex",
                        default=5,
                        type=int)
    parser.add_argument("--loss_weight_neg_ex",
                        default=1.0,
                        type=float)
    parser.add_argument("--target_influence_downweight_neg_ex",
                        default=0.2,
                        type=float)
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
    parser.add_argument("--influence_tuning_batch_size",
                        default=8,
                        type=int,
                        help="Equivalent batch size for influence tuning")
    parser.add_argument("--influence_tuning_lr",
                        default=1e-4,
                        type=float,
                        help="learning rate for the influence tuning optimizer")
    parser.add_argument("--influence_tuning_instance_dropout",
                        default=0.95,
                        type=float,
                        help="influence tuning step cannot handle too many examples, only picking a few examples each round")
    parser.add_argument("--influence_tuning_epochs",
                        default=1,
                        type=int,
                        help="influence tuning inner epochs")
    parser.add_argument('--influence_tuning_invariant_epoch',
                        action='store_true',
                        help="whether to use the same set of training examples samples during the epochs of influence tuning")
    parser.add_argument('--alt_embedding_tuning',
                        action='store_true',
                        help="whether to use the alternative embedding tuning over the influence tuning")
    parser.add_argument('--no_epoch_checkpoint_saving',
                        action='store_true',
                        help="whether to save every training checkpoint")
    parser.add_argument("--extra_config_file",
                        type=str,
                        default="",
                        help="location of the extra config file containing hyperparameters")
    parser.add_argument('--alt_optim_plan',
                        action='store_true',
                        help="whether to use bertadam for the influence tuning optimizer")
    parser.add_argument("--confound_access_rate",
                        default=1.0,
                        type=float,
                        help="control for the size of the confound-available data")
    
    args = parser.parse_args()

    bolt_config = bolt.get_current_config()
    
    # load some hyperparameters from config file
    if args.extra_config_file:
        args_dict = vars(args)
        if bolt is None:
            with open(args.extra_config_file, 'r') as extra_config_file:
                hp_config = yaml.safe_load(extra_config_file)
        else:
            hp_config = bolt_config['parameters']
            hp_config['output_dir'] = os.path.join(bolt.ARTIFACT_DIR, 'coord_tagger_tagger')

        # add extra hp config to the args namespace
        args_dict.update(hp_config)
        args = Namespace(**args_dict)

#     # set device with Horovod local rank
#     torch.cuda.set_device(hvd.local_rank())

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    args.device = device

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_eval` or `do_test` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        logger.info("WARNING: Output directory ({}) already exists and is not empty.".format(args.output_dir))
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
    num_train_optimization_steps = None
    if args.do_train:
        if args.mode == 'synth' or args.mode == 'msgs':
            train_examples = processor.get_train_examples(args.data_dir)
            inftun_train_examples = processor.get_train_examples(args.data_dir)
            inftun_test_examples = processor.get_train_examples(args.data_dir) # Han: can be changed
            num_train_optimization_steps = int(
                len(train_examples) / args.train_batch_size) * args.num_train_epochs
        else:
            raise ValueError("Check your args.mode")

    # Prepare model
    cache_dir = args.cache_dir if args.cache_dir else os.path.join(PYTORCH_PRETRAINED_BERT_CACHE, 'distributed_{}'.format(-1))
    if args.trained_model_dir: # load in fine-tuned (with cloze-style LM objective) model
        if os.path.exists(os.path.join(args.output_dir, WEIGHTS_NAME)):
            previous_state_dict = torch.load(os.path.join(args.output_dir, WEIGHTS_NAME))
        else:
            from collections import OrderedDict
            previous_state_dict = OrderedDict()
        distant_state_dict = torch.load(os.path.join(args.trained_model_dir, WEIGHTS_NAME))
        previous_state_dict.update(distant_state_dict) # note that the final layers of previous model and distant model must have different attribute names!
        model = MyBertForSequenceClassification.from_pretrained(args.trained_model_dir, state_dict=previous_state_dict, num_labels=num_labels)
    else:
        model = MyBertForSequenceClassification.from_pretrained(args.bert_model, cache_dir=cache_dir, num_labels=num_labels)

    model.to(device)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    if args.freeze_bert: # freeze BERT if needed
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
        if args.alt_embedding_tuning: # classifier after [CLS] is not used when calculating embedding distance
            if 'classifier.' in n:
                continue
        if (not any(fr in n for fr in frozen)):
            param_influence.append(p)
        elif 'bert.embeddings.word_embeddings.' in n:
            pass # need gradients through embedding layer for computing saliency map
        else:
            p.requires_grad = False
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if (not any(fr in n for fr in frozen)) and (not any(nd in n for nd in no_decay))], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if (not any(fr in n for fr in frozen)) and (any(nd in n for nd in no_decay))], 'weight_decay': 0.0}
        ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)
    
    # influence tuning optimizer
    if args.alt_embedding_tuning:
        inftun_optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if (not any(fr in n for fr in frozen)) and ('classifier.' not in n)],
            'lr': args.influence_tuning_lr, 'weight_decay': 0.01}]
    else:
        inftun_optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if (not any(fr in n for fr in frozen))],
            'lr': args.influence_tuning_lr, 'weight_decay': 0.01}]

    # choice of inftun optimizer
    if args.alt_optim_plan:
        inftun_optimizer = BertAdam(inftun_optimizer_grouped_parameters, lr=args.influence_tuning_lr)
    else:
        inftun_optimizer = torch.optim.SGD(inftun_optimizer_grouped_parameters, lr=args.influence_tuning_lr)

#     # Horovod inftun optimizer
#     inftun_optimizer = hvd.DistributedOptimizer(inftun_optimizer)

    if args.do_train:
        global_step = 0
        if args.mode == 'synth':
            train_features = convert_examples_to_features_pretokenized(
                train_examples, label_list, args.max_seq_length, tokenizer)
        else:
            train_features = convert_examples_to_features(
                train_examples, label_list, args.max_seq_length, tokenizer)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_id = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        all_note = torch.tensor([f.note for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_id, all_note)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        
        # influence tuning preparation
        if args.mode == 'synth':
            inftun_train_features = convert_examples_to_features_pretokenized(
                inftun_train_examples, label_list, args.max_seq_length, tokenizer)
        else:
            inftun_train_features = convert_examples_to_features(
                inftun_train_examples, label_list, args.max_seq_length, tokenizer)
            
        if args.mode == 'synth':
            inftun_test_features = convert_examples_to_features_pretokenized(
                inftun_test_examples, label_list, args.max_seq_length, tokenizer)
        else:
            inftun_test_features = convert_examples_to_features(
                inftun_test_examples, label_list, args.max_seq_length, tokenizer)
            
        if args.confound_access_rate < 1:
            if args.mode == 'synth' or args.mode == 'msgs': # Han: currently train = test for our setup
                minority_spur_group_0 = []
                minority_spur_group_1 = []
                for f in inftun_train_features:
                    if f.label_id == 0 and f.note[1] == 1:
                        minority_spur_group_1.append(f)
                    elif f.label_id == 1 and f.note[1] == 0:
                        minority_spur_group_0.append(f)
                assert len(minority_spur_group_0) >= 5
                assert len(minority_spur_group_1) >= 5
                minority_spur_group_0 = minority_spur_group_0[:5] # can change
                minority_spur_group_1 = minority_spur_group_1[:5] # can change
                new_inftun_train_features = random.sample(inftun_train_features, int(args.confound_access_rate * len(inftun_train_features)) - 10) # can change
                new_inftun_train_features.extend(minority_spur_group_0)
                new_inftun_train_features.extend(minority_spur_group_1)
                test_effective_size = len(inftun_train_features) - int(args.influence_tuning_instance_dropout * len(inftun_train_features))
                inftun_train_features = new_inftun_train_features
                inftun_test_features = new_inftun_train_features
                random.shuffle(inftun_train_features)
                random.shuffle(inftun_test_features)
                args.start_test_idx = 0
                args.end_test_idx = len(inftun_test_features) - 1
                args.influence_tuning_instance_dropout = (len(inftun_test_features) - test_effective_size) / len(inftun_test_features) * 1.0
#                 print(args.influence_tuning_instance_dropout)
#                 print(args.end_test_idx)
            else:
                raise ValueError("n/a")
            
        inftun_all_input_ids = torch.tensor([f.input_ids for f in inftun_train_features], dtype=torch.long)
        inftun_all_input_mask = torch.tensor([f.input_mask for f in inftun_train_features], dtype=torch.long)
        inftun_all_segment_ids = torch.tensor([f.segment_ids for f in inftun_train_features], dtype=torch.long)
        inftun_all_label_id = torch.tensor([f.label_id for f in inftun_train_features], dtype=torch.long)
        inftun_all_guids = torch.tensor([f.guid for f in inftun_train_features], dtype=torch.long)
        inftun_all_note = torch.tensor([f.note for f in inftun_train_features], dtype=torch.long)
        inftun_train_data = TensorDataset(inftun_all_input_ids, inftun_all_input_mask, inftun_all_segment_ids,\
            inftun_all_label_id, inftun_all_guids, inftun_all_note)
        inftun_train_dataloader = DataLoader(inftun_train_data, sampler=SequentialSampler(inftun_train_data), batch_size=1)
            
        inftun_all_input_ids = torch.tensor([f.input_ids for f in inftun_test_features], dtype=torch.long)
        inftun_all_input_mask = torch.tensor([f.input_mask for f in inftun_test_features], dtype=torch.long)
        inftun_all_segment_ids = torch.tensor([f.segment_ids for f in inftun_test_features], dtype=torch.long)
        inftun_all_label_id = torch.tensor([f.label_id for f in inftun_test_features], dtype=torch.long)
        inftun_all_guids = torch.tensor([f.guid for f in inftun_test_features], dtype=torch.long)
        inftun_all_note = torch.tensor([f.note for f in inftun_test_features], dtype=torch.long)
        inftun_test_data = TensorDataset(inftun_all_input_ids, inftun_all_input_mask, inftun_all_segment_ids,\
            inftun_all_label_id, inftun_all_guids, inftun_all_note)

#         # distributed inftun test dataloader for Horovod (only works for scenario insensitive to test data order now, e.g., synth)
#         dist_inftun_test_sampler = torch.utils.data.distributed.DistributedSampler(
#             inftun_test_data, num_replicas=hvd.size(), rank=hvd.rank())
        inftun_test_dataloader = DataLoader(inftun_test_data, sampler=SequentialSampler(inftun_test_data), batch_size=1)

#         args.lissa_depth = int(args.lissa_depth_pct * len(inftun_train_examples))
#         agg_iloss_grads = None
#         agg_iloss_grads_count = 0
        agg_iloss_grads_buffer = [None, 0]
        iloss_logging_dict = defaultdict(list)

        # FORMAL TRAINING
        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            epoch_loss = []
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                model.zero_grad()
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids, note = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                loss.backward()
                optimizer.step() # Han: probably need reinit after the later influencne tuning (corrupted momentum information?)
#                 inftun_optimizer.synchronize()
                model.zero_grad()
                global_step += 1
                epoch_loss.append(loss.item())
                
                # influence tuning
                if (global_step % args.coord_interval == 0)\
                    and (num_train_optimization_steps - global_step > args.coord_interval / 2): # Han: keep the last label tuning long enough
#                     # Horovod parameter broadcast
#                     hvd.broadcast_parameters(model.state_dict(), root_rank=0)

                    # Save a training checkpoint
                    if not args.no_epoch_checkpoint_saving:
                        epoch_output_dir = os.path.join(args.output_dir, f"epoch_{int(global_step / args.coord_interval)}_pre/")
                        if not os.path.exists(epoch_output_dir):
                            os.makedirs(epoch_output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(epoch_output_dir, WEIGHTS_NAME)
                        torch.save(model_to_save.state_dict(), output_model_file)
                        output_config_file = os.path.join(epoch_output_dir, CONFIG_NAME)
                        with open(output_config_file, 'w') as f:
                            f.write(model_to_save.config.to_json_string())

                    if args.alt_embedding_tuning:
                        embedding_tuning(model, args, inftun_train_dataloader, inftun_test_dataloader,
                                         inftun_train_examples, args.start_test_idx, args.end_test_idx, param_influence,
                                         inftun_optimizer, agg_iloss_grads_buffer, iloss_logging_dict)
                    else:
                        influence_tuning(model, args, inftun_train_dataloader, inftun_test_dataloader,
                                         inftun_train_examples, args.start_test_idx, args.end_test_idx, param_influence,
                                         inftun_optimizer, agg_iloss_grads_buffer, iloss_logging_dict)
                
                    if args.alt_optim_plan:
                        reset_bertadam_state(optimizer) # reset label tuning optimizer's state
                        reset_bertadam_state(inftun_optimizer) # reset influence tuning optimizer's state
                        
                    # Save a training checkpoint
                    if not args.no_epoch_checkpoint_saving:
                        epoch_output_dir = os.path.join(args.output_dir, f"epoch_{int(global_step / args.coord_interval)}_post/")
                        if not os.path.exists(epoch_output_dir):
                            os.makedirs(epoch_output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                        output_model_file = os.path.join(epoch_output_dir, WEIGHTS_NAME)
                        torch.save(model_to_save.state_dict(), output_model_file)
                        output_config_file = os.path.join(epoch_output_dir, CONFIG_NAME)
                        with open(output_config_file, 'w') as f:
                            f.write(model_to_save.config.to_json_string())

            logger.info("  label tuning epoch loss = %f", np.mean(epoch_loss))

    if args.do_train:
        # Save a trained model and the associated configuration
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        torch.save(model_to_save.state_dict(), output_model_file)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
        with open(output_config_file, 'w') as f:
            f.write(model_to_save.config.to_json_string())

    if args.do_test:
        if args.mode == "synth" or args.mode == 'msgs':
            test_examples_seq = [processor.get_train_examples(args.data_dir),
                                 processor.get_dev_examples(args.data_dir),
                                 processor.get_test_examples(args.data_dir)]
            idx2setname = {0: 'train_set', 1: 'dev_set', 2: 'test_set'}
        else:
            raise ValueError("Check your args.mode")
            
        dev_test_combo = 0
        
        for set_idx, test_examples in enumerate(test_examples_seq):
            if args.mode == 'synth':
                test_features = convert_examples_to_features_pretokenized(
                    test_examples, label_list, args.max_seq_length, tokenizer)
            else:
                test_features = convert_examples_to_features(
                    test_examples, label_list, args.max_seq_length, tokenizer)
            logger.info("***** Running final test *****")
            logger.info("  Num examples = %d", len(test_examples))
            logger.info("  Batch size = %d", args.eval_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
            all_label_id = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
            all_guid = torch.tensor([f.guid for f in test_features], dtype=torch.long)
            all_note = torch.tensor([f.note for f in test_features], dtype=torch.long)
            test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_id, all_guid, all_note)
            # Run prediction for full data
            test_sampler = SequentialSampler(test_data)
            test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

            model.eval()
            test_loss, test_accuracy = 0, 0
            nb_test_steps, nb_test_examples = 0, 0

            error_guid_list = []

            for input_ids, input_mask, segment_ids, label_ids, guids, note in tqdm(test_dataloader, desc="Testing"):
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                segment_ids = segment_ids.to(device)
                label_ids = label_ids.to(device)

                with torch.no_grad():
                    tmp_test_loss = model(input_ids, segment_ids, input_mask, label_ids)
                    logits = model(input_ids, segment_ids, input_mask)

                logits = logits.detach().cpu().numpy()
                label_ids = label_ids.to('cpu').numpy()

                tmp_test_correct, tmp_test_total = accuracy(logits, label_ids)
                
#                 if tmp_test_correct != tmp_test_total:
#                     error_guid_list.append(guids.item())

                test_loss += tmp_test_loss.mean().item()
                test_accuracy += tmp_test_correct

                nb_test_examples += tmp_test_total
                nb_test_steps += 1

            test_loss = test_loss / nb_test_steps
            test_accuracy = test_accuracy / nb_test_examples
            result = {'test_loss': test_loss,
                      'test_accuracy': test_accuracy}
            if idx2setname[set_idx] == 'dev_set':
                bolt.send_metrics({'dev_%s'%k: v for k, v in result.items()})
                dev_test_combo += int(result['test_accuracy'] * 100000)
            elif idx2setname[set_idx] == 'test_set':
                bolt.send_metrics({'test_%s'%k: v for k, v in result.items()})
                dev_test_combo += result['test_accuracy']

            output_test_file = os.path.join(args.output_dir, f"{idx2setname[set_idx]}_test_results.txt")
            with open(output_test_file, "w") as writer:
                logger.info("***** Test results *****")
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                    writer.write("%s = %s\n" % (key, str(result[key])))
                    
            output_err_file = os.path.join(args.output_dir, f"{idx2setname[set_idx]}_error_guids.pkl")
            pickle.dump(error_guid_list, open(output_err_file, "wb"))
            
        bolt.send_metrics({'dev_test_combo': dev_test_combo})

if __name__ == "__main__":
    main()
