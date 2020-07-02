# -*- coding: utf-8 -*-

# # # #
# NER_BERT_CRF.py
# @author Zhibin.LU
# @created Fri Feb 15 2019 22:47:19 GMT-0500 (EST)
# @last-modified Sun Mar 31 2019 12:17:08 GMT-0400 (EDT)
# @website: https://louis-udm.github.io
# @description: Bert pytorch pretrainde model with or without CRF for NER
# The NER_BERT_CRF.py include 2 model:
# - model 1:
#   - This is just a pretrained BertForTokenClassification, For a comparision with my BERT-CRF model
# - model 2:
#   - A pretrained BERT with CRF model.
# - data set
#   - [CoNLL-2003](https://github.com/FuYanzhe2/Name-Entity-Recognition/tree/master/BERT-BiLSTM-CRF-NER/NERdata)
# # # #


# %%
import sys
import os
import time
import importlib
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim

from torch.utils.data.distributed import DistributedSampler
from torch.utils import data

from tqdm import tqdm, trange
import collections

from pytorch_pretrained_bert.modeling import BertModel, BertForTokenClassification, BertLayerNorm
import pickle
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer

def set_work_dir(local_path="ner_bert_crf", server_path="ner_bert_crf"):
    if (os.path.exists(os.getenv("HOME")+'/'+local_path)):
        os.chdir(os.getenv("HOME")+'/'+local_path)
    elif (os.path.exists(os.getenv("HOME")+'/'+server_path)):
        os.chdir(os.getenv("HOME")+'/'+server_path)
    else:
        raise Exception('Set work path error!')


def get_data_dir(local_path="ner_bert_crf", server_path="ner_bert_crf"):
    if (os.path.exists(os.getenv("HOME")+'/'+local_path)):
        return os.getenv("HOME")+'/'+local_path
    elif (os.path.exists(os.getenv("HOME")+'/'+server_path)):
        return os.getenv("HOME")+'/'+server_path
    else:
        raise Exception('get data path error!')


print('Python version ', sys.version)
print('PyTorch version ', torch.__version__)

set_work_dir()
print('Current dir:', os.getcwd())

cuda_yes = torch.cuda.is_available()
# cuda_yes = False
print('Cuda is available?', cuda_yes)
device = torch.device("cuda:0" if cuda_yes else "cpu")
print('Device:', device)

data_dir = os.path.join(get_data_dir(), 'data/')
# "Whether to run training."
do_train = True
# "Whether to run eval on the dev set."
do_eval = True
# "Whether to run the model in inference mode on the test set."
do_predict = True
# Whether load checkpoint file before train model
load_checkpoint = True
# "The vocabulary file that the BERT model was trained on."
max_seq_length = 180 #256
batch_size = 32 #32
# "The initial learning rate for Adam."
learning_rate0 = 5e-5
lr0_crf_fc = 8e-5
weight_decay_finetune = 1e-5 #0.01
weight_decay_crf_fc = 5e-6 #0.005
total_train_epochs = 15
gradient_accumulation_steps = 1
warmup_proportion = 0.1
output_dir = './output/'
bert_model_scale = 'bert-base-cased-pt-br'
vocab = 'vocab.txt'
do_lower_case = False
# eval_batch_size = 8
# predict_batch_size = 8
# "Proportion of training to perform linear learning rate warmup for. "
# "E.g., 0.1 = 10% of training."
# warmup_proportion = 0.1
# "How often to save the model checkpoint."
# save_checkpoints_steps = 1000
# "How many steps to make in each estimator call."
# iterations_per_loop = 1000


# %%
'''
Functions and Classes for read and organize data set
'''

class InputExample(object):
    """A single training/test example for NER."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example(a sentence or a pair of sentences).
          words: list of words of sentence
          labels_a/labels_b: (Optional) string. The label seqence of the text_a/text_b. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        # list of words of the sentence,example: [EU, rejects, German, call, to, boycott, British, lamb .]
        self.words = words
        # list of label sequence of the sentence,like: [B-ORG, O, B-MISC, O, O, O, B-MISC, O, O]
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data.
    result of convert_examples_to_features(InputExample)
    """

    def __init__(self, input_ids, input_mask, segment_ids,  predict_mask, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.predict_mask = predict_mask
        self.label_ids = label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file):
        """
        Reads a BIO data.
        """
        with open(input_file) as f:
            # out_lines = []
            out_lists = []
            entries = f.read().strip().split("\n\n")
            for entry in entries:
                words = []
                ner_labels = []
                pos_tags = []
                bio_pos_tags = []
                for line in entry.splitlines():
                    pieces = line.strip().split()
                    if len(pieces) < 1:
                        continue
                    word = pieces[0]
                    # if word == "-DOCSTART-" or word == '':
                    #     continue
                    words.append(word)
                    pos_tags.append(pieces[1])
                    bio_pos_tags.append(pieces[2])
                    ner_labels.append(pieces[-1])
                # sentence = ' '.join(words)
                # ner_seq = ' '.join(ner_labels)
                # pos_tag_seq = ' '.join(pos_tags) 
                # bio_pos_tag_seq = ' '.join(bio_pos_tags) 
                # out_lines.append([sentence, pos_tag_seq, bio_pos_tag_seq, ner_seq])
                # out_lines.append([sentence, ner_seq])
                out_lists.append([words,pos_tags,bio_pos_tags,ner_labels])
        return out_lists


class CoNLLDataProcessor(DataProcessor):
    '''
    CoNLL-2003
    '''

    def __init__(self):
        self._label_types = self.generate_labels()
        self._num_labels = len(self._label_types)
        self._label_map = {label: i for i,
                           label in enumerate(self._label_types)}

    def get_train_examples(self, data_dir):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "train.txt")))

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "valid.txt")))

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_data(os.path.join(data_dir, "test.txt")))

    def get_labels(self):
        return self._label_types

    def get_num_labels(self):
        return self.get_num_labels

    def get_label_map(self):
        return self._label_map
    
    def get_start_label_id(self):
        return self._label_map['[CLS]']

    def get_stop_label_id(self):
        return self._label_map['[SEP]']

    def _create_examples(self, all_lists):
        examples = []
        for (i, one_lists) in enumerate(all_lists):
            guid = i
            words = one_lists[0]
            labels = one_lists[-1]
            examples.append(InputExample(
                guid=guid, words=words, labels=labels))
        return examples

    def _create_examples2(self, lines):
        examples = []
        for (i, line) in enumerate(lines):
            guid = i
            text = line[0]
            ner_label = line[-1]
            examples.append(InputExample(
                guid=guid, text_a=text, labels_a=ner_label))
        return examples
    
    def generate_labels(self):
        resp = ['X', '[CLS]', '[SEP]', 'O',]
        with open("tags.txt") as tags:
            for line in tags:
                tag = line.strip('\n')
                resp.append("B-"+tag)    
                resp.append("I-"+tag)    
        resp = list(set(resp))
        return resp


def example2feature(example, tokenizer, label_map, max_seq_length):

    add_label = 'X'
    # tokenize_count = []
    tokens = ['[CLS]']
    predict_mask = [0]
    label_ids = [label_map['[CLS]']]
    for i, w in enumerate(example.words):
        # use bertTokenizer to split words
        # 1996-08-22 => 1996 - 08 - 22
        # sheepmeat => sheep ##me ##at
        sub_words = tokenizer.tokenize(w)
        if not sub_words:
            sub_words = ['[UNK]']
        # tokenize_count.append(len(sub_words))
        tokens.extend(sub_words)
        for j in range(len(sub_words)):
            if j == 0:
                predict_mask.append(1)
                label_ids.append(label_map[example.labels[i]])
            else:
                # '##xxx' -> 'X' (see bert paper)
                predict_mask.append(0)
                label_ids.append(label_map[add_label])

    # truncate
    if len(tokens) > max_seq_length - 1:
        print('Example No.{} is too long, length is {}, truncated to {}!'.format(example.guid, len(tokens), max_seq_length))
        tokens = tokens[0:(max_seq_length - 1)]
        predict_mask = predict_mask[0:(max_seq_length - 1)]
        label_ids = label_ids[0:(max_seq_length - 1)]
    tokens.append('[SEP]')
    predict_mask.append(0)
    label_ids.append(label_map['[SEP]'])

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    segment_ids = [0] * len(input_ids)
    input_mask = [1] * len(input_ids)

    feat=InputFeatures(
                # guid=example.guid,
                # tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                predict_mask=predict_mask,
                label_ids=label_ids)

    return feat

class NerDataset(data.Dataset):
    def __init__(self, examples, tokenizer, label_map, max_seq_length):
        self.examples=examples
        self.tokenizer=tokenizer
        self.label_map=label_map
        self.max_seq_length=max_seq_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        feat=example2feature(self.examples[idx], self.tokenizer, self.label_map, max_seq_length)
        return feat.input_ids, feat.input_mask, feat.segment_ids, feat.predict_mask, feat.label_ids

    @classmethod
    def pad(cls, batch):

        seqlen_list = [len(sample[0]) for sample in batch]
        maxlen = np.array(seqlen_list).max()

        f = lambda x, seqlen: [sample[x] + [0] * (seqlen - len(sample[x])) for sample in batch] # 0: X for padding
        input_ids_list = torch.LongTensor(f(0, maxlen))
        input_mask_list = torch.LongTensor(f(1, maxlen))
        segment_ids_list = torch.LongTensor(f(2, maxlen))
        predict_mask_list = torch.ByteTensor(f(3, maxlen))
        label_ids_list = torch.LongTensor(f(4, maxlen))

        return input_ids_list, input_mask_list, segment_ids_list, predict_mask_list, label_ids_list

def f1_score(y_true, y_pred):
    '''
    0,1,2,3 are [CLS],[SEP],[X],O
    '''
    ignore_id=3
    
    num_proposed = len(y_pred[y_pred>ignore_id])
    num_correct = (np.logical_and(y_true==y_pred, y_true>ignore_id)).sum()
    num_gold = len(y_true[y_true>ignore_id])

    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision*recall==0:
            f1=1.0
        else:
            f1=0

    return precision, recall, f1

#%%
'''
Prepare data set
'''
# random.seed(44)
np.random.seed(44)
torch.manual_seed(44)
if cuda_yes:
    torch.cuda.manual_seed_all(44)

# Load pre-trained model tokenizer (vocabulary)
conllProcessor = CoNLLDataProcessor()
label_list = conllProcessor.get_labels()
label_map = conllProcessor.get_label_map()
train_examples = conllProcessor.get_train_examples(data_dir)
dev_examples = conllProcessor.get_dev_examples(data_dir)
test_examples = conllProcessor.get_test_examples(data_dir)

total_train_steps = int(len(train_examples) / batch_size / gradient_accumulation_steps * total_train_epochs)

print("***** Running training *****")
print("  Num examples = %d"% len(train_examples))
print("  Batch size = %d"% batch_size)
print("  Num steps = %d"% total_train_steps)

tokenizer = BertTokenizer.from_pretrained(vocab, do_lower_case=do_lower_case)

train_dataset = NerDataset(train_examples,tokenizer,label_map,max_seq_length)
dev_dataset = NerDataset(dev_examples,tokenizer,label_map,max_seq_length)
test_dataset = NerDataset(test_examples,tokenizer,label_map,max_seq_length)

train_dataloader = data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4,
                                collate_fn=NerDataset.pad)

dev_dataloader = data.DataLoader(dataset=dev_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=NerDataset.pad)

test_dataloader = data.DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=NerDataset.pad)


#%%
'''
#####  Use only BertForTokenClassification  #####
'''
print('*** Use only BertForTokenClassification ***')

if load_checkpoint and os.path.exists(output_dir+'/ner_bert_checkpoint.pt'):
    checkpoint = torch.load(output_dir+'/ner_bert_checkpoint.pt', map_location='cpu')
    start_epoch = checkpoint['epoch']+1
    valid_acc_prev = checkpoint['valid_acc']
    valid_f1_prev = checkpoint['valid_f1']
    model = BertForTokenClassification.from_pretrained(
        bert_model_scale, state_dict=checkpoint['model_state'], num_labels=len(label_list))
    print('Loaded the pretrain NER_BERT model, epoch:',checkpoint['epoch'],'valid acc:', 
            checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])
else:
    start_epoch = 0
    valid_acc_prev = 0
    valid_f1_prev = 0
    model = BertForTokenClassification.from_pretrained(
        bert_model_scale, num_labels=len(label_list))

model.to(device)

# Prepare optimizer
named_params = list(model.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay_finetune},
    {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = BertAdam(optimizer_grouped_parameters, lr=learning_rate0, warmup=warmup_proportion, t_total=total_train_steps)
# optimizer = optim.Adam(model.parameters(), lr=learning_rate0)

def evaluate(model, predict_dataloader, batch_size, epoch_th, dataset_name):
    # print("***** Running prediction *****")
    model.eval()
    all_preds = []
    all_labels = []
    total=0
    correct=0
    start = time.time()
    with torch.no_grad():
        for batch in predict_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
            out_scores = model(input_ids, segment_ids, input_mask)
            # out_scores = out_scores.detach().cpu().numpy()
            _, predicted = torch.max(out_scores, -1)
            valid_predicted = torch.masked_select(predicted, predict_mask)
            valid_label_ids = torch.masked_select(label_ids, predict_mask)
            # print(len(valid_label_ids),len(valid_predicted),len(valid_label_ids)==len(valid_predicted))
            all_preds.extend(valid_predicted.tolist())
            all_labels.extend(valid_label_ids.tolist())
            total += len(valid_label_ids)
            correct += valid_predicted.eq(valid_label_ids).sum().item()

    test_acc = correct/total
    precision, recall, f1 = f1_score(np.array(all_labels), np.array(all_preds))
    end = time.time()
    print('Epoch:%d, Acc:%.2f, Precision: %.2f, Recall: %.2f, F1: %.2f on %s, Spend: %.3f minutes for evaluation' \
        % (epoch_th, 100.*test_acc, 100.*precision, 100.*recall, 100.*f1, dataset_name,(end-start)/60.0))
    print('--------------------------------------------------------------')
    return test_acc, f1


#%%
# train procedure using only BertForTokenClassification
# train_start = time.time()
global_step_th = int(len(train_examples) / batch_size / gradient_accumulation_steps * start_epoch)
# for epoch in trange(start_epoch, total_train_epochs, desc="Epoch"):
for epoch in range(start_epoch, total_train_epochs):
    tr_loss = 0
    train_start = time.time()
    model.train()
    optimizer.zero_grad()
    # for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        
        input_ids, input_mask, segment_ids, predict_mask, label_ids = batch
        loss = model(input_ids, segment_ids, input_mask, label_ids)

        if gradient_accumulation_steps > 1:
            loss = loss / gradient_accumulation_steps

        loss.backward()
        tr_loss += loss.item()

        if (step + 1) % gradient_accumulation_steps == 0:
            # modify learning rate with special warm up BERT uses
            lr_this_step = learning_rate0 * warmup_linear(global_step_th/total_train_steps, warmup_proportion)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_this_step
            optimizer.step()
            optimizer.zero_grad()
            global_step_th += 1
          
        print("Epoch:{}-{}/{}, CrossEntropyLoss: {} ".format(epoch, step, len(train_dataloader), loss.item()))
    
    print('--------------------------------------------------------------')
    print("Epoch:{} completed, Total training's Loss: {}, Spend: {}m".format(epoch, tr_loss, (time.time() - 
                                                                                             
                                                                                             
                                                                                             )/60.0))
    valid_acc, valid_f1 = evaluate(model, dev_dataloader, batch_size, epoch, 'Valid_set')
    # Save a checkpoint
    if valid_f1 > valid_f1_prev:
        # model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'valid_acc': valid_acc,
            'valid_f1': valid_f1, 'max_seq_length': max_seq_length, 'lower_case': do_lower_case},
                    os.path.join(output_dir, 'ner_bert_checkpoint.pt'))
        valid_f1_prev = valid_f1

evaluate(model, test_dataloader, batch_size, total_train_epochs-1, 'Test_set')

#%%
'''
Test_set prediction using the best epoch of NER_BERT model
'''
checkpoint = torch.load(output_dir+'/ner_bert_checkpoint.pt', map_location='cpu')
epoch = checkpoint['epoch']
valid_acc_prev = checkpoint['valid_acc']
valid_f1_prev = checkpoint['valid_f1']
model = BertForTokenClassification.from_pretrained(
    bert_model_scale, state_dict=checkpoint['model_state'], num_labels=len(label_list))
# if os.path.exists(output_dir+'/ner_bert_crf_checkpoint.pt'):
model.to(device)
print('Loaded the pretrain NER_BERT model, epoch:',checkpoint['epoch'],'valid acc:', 
        checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])

model.to(device)
# evaluate(model, train_dataloader, batch_size, total_train_epochs-1, 'Train_set')
evaluate(model, test_dataloader, batch_size, epoch, 'Test_set')

print(conllProcessor.get_label_map())
# print(test_examples[8].words)
# print(test_features[8].label_ids)
