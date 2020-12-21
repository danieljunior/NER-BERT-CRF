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
import collections
from tqdm import tqdm, trange
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
from pytorch_pretrained_bert.modeling import BertLayerNorm
from transformers import BertModel
from transformers.modeling_longformer import LongformerSelfAttention
import metric_utils

class BertLongSelfAttention(LongformerSelfAttention):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        return super().forward(hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)


class BertLong(BertModel):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = BertLongSelfAttention(config, layer_id=i)

class Longformer_biLSTM_CRF(nn.Module):

    def __init__(self, bert_model, start_label_id, stop_label_id, num_labels, max_seq_length, 
                 batch_size, device, bert_output='concat', finetunning=False):
        super(Longformer_biLSTM_CRF, self).__init__()
        self.hidden_size = 768
        self.start_label_id = start_label_id
        self.stop_label_id = stop_label_id
        self.num_labels = num_labels
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.device=device
        self.bert_output = bert_output
        # use pretrainded BertModel 
        self.bert = bert_model
        self.finetunning = finetunning

        # biLSTM
        self.lstm = nn.LSTM(bidirectional=True, num_layers=2, input_size=3072, 
                            hidden_size=self.hidden_size//2, batch_first=True)
        self.dropout = torch.nn.Dropout(0.2)
        # Maps the output of the biLSTM into label space.
        self.hidden2label = nn.Linear(self.hidden_size, self.num_labels)

        # Matrix of transition parameters.  Entry i,j is the score of transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.num_labels, self.num_labels))

        # These two statements enforce the constraint that we never transfer *to* the start tag(or label),
        # and we never transfer *from* the stop label (the model would probably learn this anyway,
        # so this enforcement is likely unimportant)
        self.transitions.data[start_label_id, :] = -10000
        self.transitions.data[:, stop_label_id] = -10000

        #inicialização da camada linear
        nn.init.xavier_uniform_(self.hidden2label.weight)
        nn.init.constant_(self.hidden2label.bias, 0.0)
        # self.apply(self.init_bert_weights)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)): 
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def _forward_alg(self, feats):
        '''
        this also called alpha-recursion or forward recursion, to calculate log_prob of all barX 
        '''
        
        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]
        
        # alpha_recursion,forward, alpha(zt)=p(zt,bar_x_1:t)
        log_alpha = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(self.device)
        # normal_alpha_0 : alpha[0]=Ot[0]*self.PIs
        # self.start_label has all of the score. it is log,0 is p=1
        log_alpha[:, 0, self.start_label_id] = 0
        
        # feats: sentances -> word embedding -> lstm -> MLP -> feats
        # feats is the probability of emission, feat.shape=(1,tag_size)
        for t in range(1, T):
            log_alpha = (metric_utils.log_sum_exp_batch(self.transitions + log_alpha, axis=-1) + feats[:, t]).unsqueeze(1)

        # log_prob of all barX
        log_prob_all_barX = metric_utils.log_sum_exp_batch(log_alpha)
        return log_prob_all_barX

    def _score_sentence(self, feats, label_ids):
        ''' 
        Gives the score of a provided label sequence
        p(X=w1:t,Zt=tag1:t)=...p(Zt=tag_t|Zt-1=tag_t-1)p(xt|Zt=tag_t)...
        '''
        
        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        batch_transitions = self.transitions.expand(batch_size,self.num_labels,self.num_labels)
        batch_transitions = batch_transitions.flatten(1)

        score = torch.zeros((feats.shape[0],1)).to(self.device)
        # the 0th node is start_label->start_word,the probability of them=1. so t begin with 1.
        for t in range(1, T):
            score = score + \
                batch_transitions.gather(-1, (label_ids[:, t]*self.num_labels+label_ids[:, t-1]).view(-1,1)) \
                    + feats[:, t].gather(-1, label_ids[:, t].view(-1,1)).view(-1,1)
        return score

    def _viterbi_decode(self, feats):
        '''
        Max-Product Algorithm or viterbi algorithm, argmax(p(z_0:t|x_0:t))
        '''
        
        # T = self.max_seq_length
        T = feats.shape[1]
        batch_size = feats.shape[0]

        # batch_transitions=self.transitions.expand(batch_size,self.num_labels,self.num_labels)

        log_delta = torch.Tensor(batch_size, 1, self.num_labels).fill_(-10000.).to(self.device)
        log_delta[:, 0, self.start_label_id] = 0
        
        # psi is for the vaule of the last latent that make P(this_latent) maximum.
        psi = torch.zeros((batch_size, T, self.num_labels), dtype=torch.long).to(self.device)  # psi[0]=0000 useless
        for t in range(1, T):
            # delta[t][k]=max_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # delta[t] is the max prob of the path from  z_t-1 to z_t[k]
            log_delta, psi[:, t] = torch.max(self.transitions + log_delta, -1)
            # psi[t][k]=argmax_z1:t-1( p(x1,x2,...,xt,z1,z2,...,zt-1,zt=k|theta) )
            # psi[t][k] is the path choosed from z_t-1 to z_t[k],the value is the z_state(is k) index of z_t-1
            log_delta = (log_delta + feats[:, t]).unsqueeze(1)

        # trace back
        path = torch.zeros((batch_size, T), dtype=torch.long).to(self.device)

        # max p(z1:t,all_x|theta)
        max_logLL_allz_allx, path[:, -1] = torch.max(log_delta.squeeze(), -1)

        for t in range(T-2, -1, -1):
            # choose the state of z_t according the state choosed of z_t+1.
            path[:, t] = psi[:, t+1].gather(-1,path[:, t+1].view(-1,1)).squeeze()

        return max_logLL_allz_allx, path

    def get_bert_features(self, input_ids, segment_ids, input_mask):
        if self.finetunning:
            last_hidden_state, pooler_output, hidden_states = self.bert(input_ids, 
                                                                        token_type_ids=segment_ids, 
                                                                        attention_mask=input_mask, 
                                                                    )
        else:
            #não atualiza os pesos do bert        
            with torch.no_grad():
                last_hidden_state, pooler_output, hidden_states = self.bert(input_ids, 
                                                                            token_type_ids=segment_ids, 
                                                                            attention_mask=input_mask, 
                                                # output_all_encoded_layers=False
                                            )
        if self.bert_output == 'sum':
            # summed_last_4_layers
            return torch.stack(hidden_states[-4:]).sum(0)
        elif self.bert_output =='concat':
            pooled_output = torch.cat(tuple([hidden_states[i] for i in [-4, -3, -2, -1]]), dim=-1)
            return pooled_output[:, 0, :]
        # last_layer
        return hidden_states[-1]
    
    def get_bilstm_features(self, input_ids, segment_ids, input_mask):
        bert_seq_out = self.get_bert_features(input_ids, segment_ids, input_mask)
        # Get the emission scores from the BiLSTM
        bilstm_out, _ = self.lstm(bert_seq_out)
        bilstm_out = self.dropout(bilstm_out)
        bilstm_out = self.hidden2label(bilstm_out)
        return bilstm_out
        
    def neg_log_likelihood(self, input_ids, segment_ids, input_mask, label_ids):
        bilstm_feats = self.get_bilstm_features(input_ids, segment_ids, input_mask)
        forward_score = self._forward_alg(bilstm_feats)
        # p(X=w1:t,Zt=tag1:t)=...p(Zt=tag_t|Zt-1=tag_t-1)p(xt|Zt=tag_t)...
        gold_score = self._score_sentence(bilstm_feats, label_ids)
        # - log[ p(X=w1:t,Zt=tag1:t)/p(X=w1:t) ] = - log[ p(Zt=tag1:t|X=w1:t) ]
        return torch.mean(forward_score - gold_score)

    # this forward is just for predict, not for train
    # dont confuse this with _forward_alg above.
    def forward(self, input_ids, segment_ids, input_mask):
        # Get the emission scores from the BiLSTM
        bilstm_feats = self.get_bilstm_features(input_ids, segment_ids, input_mask)
        
        # Find the best path, given the features.
        score, label_seq_ids = self._viterbi_decode(bilstm_feats)
        return score, label_seq_ids
