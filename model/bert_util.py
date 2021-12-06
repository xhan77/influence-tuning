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
import ast

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
from torch.autograd import Function

from datasets import load_dataset
from contextlib import contextmanager

class MyBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels):
        super(MyBertForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits
        
class GradientReversalFunction(Function): # from https://github.com/jvanvugt/pytorch-domain-adaptation
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)
        
class MyBertPlusAdvForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, num_labels, num_confounds, balancing_lambda=1):
        super(MyBertPlusAdvForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.num_confounds = num_confounds
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.adv_classifier = nn.Linear(config.hidden_size, num_confounds)
        self.grad_rev_layer = GradientReversal(balancing_lambda)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None, confounds=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        pooled_output_after_grad_rev = self.grad_rev_layer(pooled_output)
        confound_logits = self.adv_classifier(pooled_output_after_grad_rev)

        if (labels is not None) and (confounds is not None):
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            confound_loss_fct = CrossEntropyLoss()
            confound_loss = confound_loss_fct(confound_logits.view(-1, self.num_confounds), confounds.view(-1))
            return loss + confound_loss, loss, confound_loss
        elif labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        elif confounds is not None:
            raise ValueError("should not happen in our experiments")
            confound_loss_fct = CrossEntropyLoss()
            confound_loss = torch.clamp(confound_loss_fct(confound_logits.view(-1, self.num_confounds), confounds.view(-1)),
                                        max=math.log(self.num_confounds))
            return confound_loss
        else:
            return logits
        
class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None, note=[0]):
        """Constructs a InputExample.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.note = note

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, guid, note=[0]):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.guid = guid
        self.note = note
        
        
class SAProcessor_TextAttack(object):
    
    def get_train_examples(self, data_dir):
        """See base class."""
        sa_dataset = load_dataset("imdb")
        sa_dataset_list = [e for e in sa_dataset['train']]
        random.seed(2021)
        random.shuffle(sa_dataset_list)
        dataset = [InputExample(guid=i, text_a=e['text'], text_b=None, label=e['label']) for i, e in enumerate(sa_dataset_list)]
        return dataset
    
    def get_test_examples(self, data_dir):
        """See base class."""
        sa_dataset = load_dataset("imdb")
        sa_dataset_list = [e for e in sa_dataset['test']]
        random.seed(2022) # different seeds for train and test set
        random.shuffle(sa_dataset_list)
        dataset = [InputExample(guid=i, text_a=e['text'], text_b=None, label=e['label']) for i, e in enumerate(sa_dataset_list)]
        return dataset
    
    def get_train_orig_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "imdb_train_subset.csv")), "orig")
    
    def get_train_adv_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "imdb_train_subset.csv")), "adv")
    
    def get_test_orig_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "imdb_test_subset.csv")), "orig")
    
    def get_test_adv_examples(self, data_dir):
        """See base class."""
        return self._create_examples(self._read_csv(os.path.join(data_dir, "imdb_test_subset.csv")), "adv")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = i - 1
            if set_type == 'adv':
                label = int(float(line[0]))
                text_a = line[7]
            elif set_type == 'orig':
                label = int(float(line[0]))
                text_a = line[4]
            else:
                raise ValueError("set_type error")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
    
    def _read_csv(cls, input_file, quotechar='\"'):
        """Reads a tab separated value file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter=",", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines
        
        
class NLIProcessor_ANLI(object):
    
    def get_train_examples(self, data_dir):
        """See base class."""
        examples = self._create_examples(self._read_csv(os.path.join(data_dir, "ANLI_analysis_r1_dev.csv"))
                                         + self._read_csv(os.path.join(data_dir, "ANLI_analysis_r2_dev.csv")), "anli_analysis")
        random.seed(2021)
        random.shuffle(examples)
        return examples
    
    def get_test_examples(self, data_dir):
        """See base class."""
        examples = self._create_examples(self._read_csv(os.path.join(data_dir, "ANLI_analysis_r3_dev.csv")), "anli_analysis")
        random.seed(2022)
        random.shuffle(examples)
        return examples

    def get_labels(self):
        """See base class."""
        return ['e', 'n', 'c']
    
    def get_meta2idx(self):
        return {'N/A': 0, 'Numerical': 1, 'Basic': 2, 'Reference': 3, 'Tricky': 4, 'Reasoning': 5, 'Imperfection': 6,}
    
    def get_idx2meta(self):
        meta2idx = self.get_meta2idx()
        return {idx: meta for meta, idx in meta2idx.items()}

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        meta2idx = self.get_meta2idx()
        for (i, line) in enumerate(lines):
            guid = i
            label = line['gold_label']
            text_a = line['context']
            text_b = line['statement']
            note = []
            for tag in line['tags']:
                if tag in meta2idx:
                    note.append(meta2idx[tag])
                    break
            if len(note) == 0:
                note = [meta2idx['N/A']]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, note=note))
        return examples
    
    def _read_csv(cls, input_file, quotechar='\"'):
        """Reads a tab separated value file."""
        with open(input_file, newline='') as f:
            lines = []
            ratings_headers = dict() # key: index of columns, value: header name
            ratings_h2ix = dict()
            csvreader = csv.reader(f, delimiter=',', quotechar=quotechar)
            for _i, row in enumerate(csvreader):
                if _i == 0:
                    for _j, header in enumerate(row):
                        ratings_headers[_j] = header.strip()
                    ratings_h2ix = {_h: _ix for _ix, _h in ratings_headers.items()}
                    continue
                line = dict()
                line['context'] = row[ratings_h2ix['context']]
                line['statement'] = row[ratings_h2ix['statement']]
                line['gold_label'] = row[ratings_h2ix['gold_label']]
                line['A1Code'] = row[ratings_h2ix['A1Code']]
                line['tags'] = ast.literal_eval(row[ratings_h2ix['tags']])
                lines.append(line)
            return lines
        
        
class SynthProcessor(object):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_pkl(os.path.join(data_dir, "synth_train.pkl")), "synth_train")
    
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_pkl(os.path.join(data_dir, "synth_dev.pkl")), "synth_dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_pkl(os.path.join(data_dir, "synth_test.pkl")), "synth_test")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, elem) in enumerate(data):
            guid = i
            label = elem[1]
            text_a = elem[0]
            note = elem[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label, note=note))
        return examples

    def _read_pkl(self, input_file):
        """Reads a tab separated value file."""
        data = pickle.load(open(input_file, 'rb'))
        return data

    
class MSGSProcessor(object):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_pkl(os.path.join(data_dir, "msgs_train.pkl")), "msgs_train")
    
    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_pkl(os.path.join(data_dir, "msgs_dev.pkl")), "msgs_dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_pkl(os.path.join(data_dir, "msgs_test.pkl")), "msgs_test")

    def get_labels(self):
        """See base class."""
        return [0, 1]

    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, elem) in enumerate(data):
            guid = i
            label = elem[1]
            text_a = elem[0]
            note = elem[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, label=label, note=note))
        return examples

    def _read_pkl(self, input_file):
        """Reads a tab separated value file."""
        data = pickle.load(open(input_file, 'rb'))
        return data


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              guid=example.guid,
                              note=example.note))
    return features


def convert_examples_to_features_pretokenized(examples, label_list, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s.""" # max actual len of the synth data: 50

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens = example.text_a
        assert isinstance(tokens, list)

        bert_tokens = []
        orig_to_tok_map = [] # no use for our problem

        bert_tokens.append("[CLS]")
        for token in tokens:
            new_tokens = tokenizer.tokenize(token)
            if len(bert_tokens) + len(new_tokens) > max_seq_length - 1:
                # print("You shouldn't see this since the test set is already pre-separated.")
                break
            else:
                orig_to_tok_map.append(len(bert_tokens))
                bert_tokens.extend(new_tokens)
        bert_tokens.append("[SEP]")

        input_ids = tokenizer.convert_tokens_to_ids(bert_tokens)
    
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        segment_ids = [0] * max_seq_length

        assert len(segment_ids) == max_seq_length
        
        label_id = label_map[example.label]

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              guid=example.guid,
                              note=example.note))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def accuracy(out, label_ids):
    # axis-0: seqs in batch; axis-1: potential labels of seq
    outputs = np.argmax(out, axis=1)
    matched = outputs == label_ids
    num_correct = np.sum(matched)
    num_total = len(label_ids)
    return num_correct, num_total

@contextmanager
def timing(description: str) -> None:
    start_time = time.time()
    yield
    print(f"#### Time for {description}: {time.time() - start_time} sec ####")
