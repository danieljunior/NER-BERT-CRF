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
from torch.utils.data import SequentialSampler

from tqdm import tqdm, trange
import collections

from pytorch_pretrained_bert.modeling import BertModel, BertForTokenClassification, BertLayerNorm
import pickle
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer
from data_utils import NERDataProcessor, NerDataset
from BERT_biLSTM_CRF import BERT_biLSTM_CRF
from BERT_CRF import BERT_CRF
import metric_utils
import argparse

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def get_optimizer(model, hp, total_train_steps):
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    new_param = ['transitions', 'hidden2label.weight', 'hidden2label.bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) \
            and not any(nd in n for nd in new_param)], 'weight_decay': hp.weight_decay_finetune},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) \
            and not any(nd in n for nd in new_param)], 'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer if n in ('transitions','hidden2label.weight')] \
            , 'lr':hp.lr0_crf_fc, 'weight_decay': hp.weight_decay_crf_fc},
        {'params': [p for n, p in param_optimizer if n == 'hidden2label.bias'] \
            , 'lr':hp.lr0_crf_fc, 'weight_decay': 0.0}
    ]
    return BertAdam(optimizer_grouped_parameters, lr=hp.learning_rate0, 
                    warmup=hp.warmup_proportion, t_total=total_train_steps)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate0)
    
    
    
if __name__=="__main__":
    print('Python version ', sys.version)
    print('PyTorch version ', torch.__version__)

    # set_work_dir()
    # print('Current dir:', os.getcwd())

    cuda_yes = torch.cuda.is_available()
    # cuda_yes = False
    print('Cuda is available?', cuda_yes)
    device = torch.device("cuda:0" if cuda_yes else "cpu")
    print('Device:', device)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert_model_scale", type=str, default="bert-base-cased-pt-br")
    parser.add_argument("--vocab", type=str, default="vocab.txt")
    parser.add_argument("--model", type=str, default="token") #token, crf, bilstm_crf
    parser.add_argument("--bert_output", type=str, default="last") #last , sum
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="./output/")
    parser.add_argument("--learning_rate0", type=float, default=1e-5)
    parser.add_argument("--lr0_crf_fc", type=float, default=8e-5)
    parser.add_argument("--weight_decay_finetune", type=float, default=1e-5)
    parser.add_argument("--weight_decay_crf_fc", type=float, default=5e-6)
    parser.add_argument("--gradient_accumulation_steps", type=float, default=1)
    # "Proportion of training to perform linear learning rate warmup for. "
    # "E.g., 0.1 = 10% of training."
    parser.add_argument("--warmup_proportion", type=float, default=0.1)
    parser.add_argument("--finetuning", dest="finetuning", action="store_true")
    parser.add_argument("--no_finetuning", dest="finetuning", action="store_false")
    parser.add_argument("--load_checkpoint", dest="load_checkpoint", action="store_true")
    parser.add_argument("--no_load_checkpoint", dest="load_checkpoint", action="store_false")
    parser.add_argument("--do_lower_case", dest="do_lower_case", action="store_true")
    parser.add_argument("--logdir", type=str, default="checkpoints/01")
    hp = parser.parse_args()


    np.random.seed(44)
    torch.manual_seed(44)
    if cuda_yes:
        torch.cuda.manual_seed_all(44)

    print('Loading data...')
    # Load pre-trained model tokenizer (vocabulary)
    nerDataProcessor = NERDataProcessor()
    label_list = nerDataProcessor.get_labels()
    label_map = nerDataProcessor.get_label_map()
    inv_label_map = {v: k for k, v in label_map.items()}
    train_examples = nerDataProcessor.get_train_examples(hp.data_dir)
    dev_examples = nerDataProcessor.get_dev_examples(hp.data_dir)
    test_examples = nerDataProcessor.get_test_examples(hp.data_dir)

    total_train_steps = int(len(train_examples) / hp.batch_size / hp.gradient_accumulation_steps * hp.n_epochs)

    print("***** Running training *****")
    print("  Num examples = %d"% len(train_examples))
    print("  Batch size = %d"% hp.batch_size)
    print("  Num steps = %d"% total_train_steps)

    tokenizer = BertTokenizer.from_pretrained(hp.vocab, do_lower_case=hp.do_lower_case)

    train_dataset = NerDataset(train_examples,tokenizer,label_map,hp.max_seq_length)
    dev_dataset = NerDataset(dev_examples,tokenizer,label_map,hp.max_seq_length)
    test_dataset = NerDataset(test_examples,tokenizer,label_map,hp.max_seq_length)

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
                                    batch_size=hp.batch_size,
                                    shuffle=True,
                                    num_workers=4,
                                    collate_fn=NerDataset.pad)

    dev_dataloader = torch.utils.data.DataLoader(dataset=dev_dataset,
                                    batch_size=hp.batch_size,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=NerDataset.pad)

    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset,
                                    batch_size=hp.batch_size,
                                    shuffle=False,
                                    num_workers=4,
                                    collate_fn=NerDataset.pad)

    start_label_id = nerDataProcessor.get_start_label_id()
    stop_label_id = nerDataProcessor.get_stop_label_id()

    print('Loading model...')
    bert_model = BertModel.from_pretrained(hp.bert_model_scale)
    if hp.model == 'bilstm_crf':
        model = BERT_biLSTM_CRF(bert_model, start_label_id, stop_label_id, len(label_list), 
                                hp.max_seq_length, hp.batch_size, device, hp.bert_output,
                                hp.finetuning)
    elif hp.model == 'crf':
        model = BERT_CRF(bert_model, start_label_id, stop_label_id, len(label_list), 
                         hp.max_seq_length, hp.batch_size, device, hp.bert_output, 
                         hp.finetuning)
    elif hp.model =='token':
        model = BertForTokenClassification.from_pretrained(
            hp.bert_model_scale, num_labels=len(label_list))


    if hp.load_checkpoint and os.path.exists(hp.output_dir+'/checkpoint.pt'):
        checkpoint = torch.load(hp.output_dir+'/checkpoint.pt', map_location='cpu')
        start_epoch = checkpoint['epoch']+1
        valid_acc_prev = checkpoint['valid_acc']
        valid_f1_prev = checkpoint['valid_f1']
        if hp.model =='token':
            model = BertForTokenClassification.from_pretrained(hp.bert_model_scale, 
                                                               state_dict=checkpoint['model_state'], 
                                                               num_labels=len(label_list))
        else:
            pretrained_dict=checkpoint['model_state']
            net_state_dict = model.state_dict()
            pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
            net_state_dict.update(pretrained_dict_selected)
            model.load_state_dict(net_state_dict)

        print('Loaded the pretrain model, epoch:', checkpoint['epoch'],'valid acc:', 
                checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])
    else:
        start_epoch = 0
        valid_acc_prev = 0
        valid_f1_prev = 0

    model.to(device)

    optimizer = get_optimizer(model, hp, total_train_steps)

    ############################ train procedure ######################################
    print('Trainning...')
    global_step_th = int(len(train_examples) / hp.batch_size / hp.gradient_accumulation_steps * start_epoch)
    for epoch in tqdm(range(start_epoch, hp.n_epochs)):
        tr_loss = 0
        train_start = time.time()
        model.train()
        optimizer.zero_grad()        
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, predict_mask, label_ids = batch

            if hp.model =='token':
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                
                if hp.gradient_accumulation_steps > 1:
                    loss = loss / hp.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()
            
            else:
                neg_log_likelihood = model.neg_log_likelihood(input_ids, segment_ids, 
                                                              input_mask, label_ids)

                if hp.gradient_accumulation_steps > 1:
                    neg_log_likelihood = neg_log_likelihood / hp.gradient_accumulation_steps

                neg_log_likelihood.backward()
                tr_loss += neg_log_likelihood.item()

            if (step + 1) % hp.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = hp.learning_rate0 * warmup_linear(global_step_th/total_train_steps, hp.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step_th += 1
            
            if hp.model =='token':
                print("Epoch:{}-{}/{}, CrossEntropyLoss: {} ".format(epoch, step, len(train_dataloader), loss.item()))
            else:        
                print("Epoch:{}-{}/{}, Negative loglikelihood: {} ".format(epoch, step, len(train_dataloader), neg_log_likelihood.item()))
        
        print("Epoch:{} completed, Total training's Loss: {}, Spend: {}m".format(epoch, tr_loss, (time.time() - train_start)/60.0))
        valid_acc, valid_f1 = metric_utils.evaluate(model, dev_dataloader, epoch, 
                                                    'Valid_set', inv_label_map, device)
        
        # Save a checkpoint
        if valid_f1 > valid_f1_prev:
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'valid_acc': valid_acc,
                'valid_f1': valid_f1, 'max_seq_length': hp.max_seq_length, 'lower_case': hp.do_lower_case},
                        os.path.join(hp.output_dir, 'checkpoint.pt'))
            valid_f1_prev = valid_f1
    print('Finished train!')
    ################################################################################################
    
    
    print('Last epoch in test set:')
    metric_utils.evaluate(model, test_dataloader, hp.n_epochs-1, 
                        'Test_set', inv_label_map, device)


    print('Best epoch in test set:')

    '''
    Test_set prediction using the best epoch of model
    '''
    checkpoint = torch.load(hp.output_dir+'/checkpoint.pt', map_location='cpu')
    epoch = checkpoint['epoch']
    valid_acc_prev = checkpoint['valid_acc']
    valid_f1_prev = checkpoint['valid_f1']
    pretrained_dict=checkpoint['model_state']
    net_state_dict = model.state_dict()
    pretrained_dict_selected = {k: v for k, v in pretrained_dict.items() if k in net_state_dict}
    net_state_dict.update(pretrained_dict_selected)
    model.load_state_dict(net_state_dict)
    # model = BertForTokenClassification.from_pretrained(
    #     bert_model_scale, state_dict=checkpoint['model_state'], num_labels=len(label_list))

    print('Loaded the pretrain  model, epoch:',checkpoint['epoch'],'valid acc:', 
        checkpoint['valid_acc'], 'valid f1:', checkpoint['valid_f1'])

    model.to(device)
    metric_utils.evaluate(model, test_dataloader, epoch, 'Test_set', inv_label_map, device)

    eval_sampler = SequentialSampler(test_dataset)
    demon_dataloader = data.DataLoader(dataset=test_dataset,
                                        sampler=eval_sampler,
                                        batch_size=hp.eval_batch_size,
                                        shuffle=False,
                                        num_workers=4,
                                        collate_fn=NerDataset.pad)
    metric_utils.evaluate(model, demon_dataloader, hp.n_epochs-1, 
                        'Test_set', inv_label_map, device, save_results=True)
