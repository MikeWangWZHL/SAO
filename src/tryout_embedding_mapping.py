import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import pickle
import json
import matplotlib.pyplot as plt
from glob import glob
import time
import copy
from tqdm import tqdm

from transformers import BertTokenizer, BertLMHeadModel, BertConfig, BertModel
from data import ZuCo_dataset
from model import BrainTranslatorBert

# TODO: train a simple fully connected neural network to map EEG feature to BERT/BART hidden states

def get_input_mapping(pre_tokenized_text, tokenizer):
    idx = 1
    enc = [tokenizer.encode(x, add_special_tokens=False) for x in pre_tokenized_text]

    idx_mapping = []

    for token in enc:
        tokenoutput = []
        for ids in token:
            tokenoutput.append(idx)
            idx +=1
        idx_mapping.append(tokenoutput)
    return idx_mapping

"""add special tokens version"""
    # def get_mapped_hidden_states(hidden_states,idx_mapping):
    #     #TODO: handle special token <CLS> and <SEP>?
    #     #TODO: handle padding
    #     #TODO: output attention mask

    #     # hidden_states: batch * input_ids_seq_len * 768
    #     # idx_mapping: batch * original_tokens_len * n , where n >= 1
    #     batch_size = hidden_states.size()[0]
    #     hidden_size = hidden_states.size()[-1]
        
    #     mapped_hidden_states = torch.zeros(batch_size, len(idx_mapping[0]) + 2, hidden_size)
        
    #     for b in range(batch_size):
    #         # add in special token [CLS] hidden states:
    #         mapped_hidden_states[b][0] = hidden_states[b][0]

    #         for t in range(len(idx_mapping[b])):
    #             mapping_list = idx_mapping[b][t]
    #             if len(mapping_list) == 1:
    #                 # mapping to exactly one token:
    #                 idx = mapping_list[0]
    #                 # start with idx = 1
    #                 mapped_hidden_states[b][t+1] = hidden_states[b][idx] # 768
    #             else:
    #                 print(f'b = {b}, t = {t}, mapping_list = {mapping_list}')
    #                 mapped_token_hidden_states_list = [hidden_states[b][i] for i in mapping_list]
    #                 print(f'len of mapped_token_hidden_states_list: {len(mapped_token_hidden_states_list)}')
    #                 # take mean of all
    #                 mapped_hidden_states[b][t+1] = torch.mean(torch.stack(mapped_token_hidden_states_list), dim = 0)
    #                 print(f'size of mapped_hidden_states[b][t]:',mapped_hidden_states[b][t+1].size())
    #                 print()
            
    #         # add in special token [SEP] hidden states:
    #         mapped_hidden_states[b][-1] = hidden_states[b][-1]
            
    #     return mapped_hidden_states

"""don't add special tokens version"""
def get_mapped_hidden_states(hidden_states,idx_mapping):
    #TODO: handle padding
    #TODO: output attention mask

    # hidden_states: batch * input_ids_seq_len * 768
    # idx_mapping: batch * original_tokens_len * n , where n >= 1
    batch_size = hidden_states.size()[0]
    hidden_size = hidden_states.size()[-1]
    
    mapped_hidden_states = torch.zeros(batch_size, len(idx_mapping[0]), hidden_size)
    
    for b in range(batch_size):
        for t in range(len(idx_mapping[b])):
            mapping_list = idx_mapping[b][t]
            if len(mapping_list) == 1:
                # mapping to exactly one token:
                idx = mapping_list[0]
                # start with idx = 1
                mapped_hidden_states[b][t] = hidden_states[b][idx] # 768
            else:
                print(f'b = {b}, t = {t}, mapping_list = {mapping_list}')
                mapped_token_hidden_states_list = [hidden_states[b][i] for i in mapping_list]
                print(f'len of mapped_token_hidden_states_list: {len(mapped_token_hidden_states_list)}')
                # take mean of all
                mapped_hidden_states[b][t] = torch.mean(torch.stack(mapped_token_hidden_states_list), dim = 0)
                print(f'size of mapped_hidden_states[b][t]:',mapped_hidden_states[b][t].size())
                print()
        
    return mapped_hidden_states



"""try out:"""
pre_tokenized_text = ['Bread,', 'My', 'Sweet', 'has', 'so', 'many', 'flaws', 'would', 'easy', 'critics', 'to', 'shred']
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# tokenizer = BertTokenizer.from_pretrained('bart-base')

inputs = tokenizer(pre_tokenized_text, is_split_into_words = True, return_tensors = 'pt')
input_ids = inputs['input_ids'][0]
input_idx_mapping = [get_input_mapping(pre_tokenized_text,tokenizer)] # batch = 1

model = BertModel.from_pretrained('bert-base-uncased')
output = model(**inputs, output_hidden_states = True, return_dict = True)
output_last_hidden_states = output.last_hidden_state # 1 * 16 * 768
mapped_hidden_states = get_mapped_hidden_states(output_last_hidden_states, input_idx_mapping)

print()
print('size of mapped_hidden_states:', mapped_hidden_states.size())
"""not adding special token hiddenstates"""
assert torch.equal(mapped_hidden_states[0][1],output_last_hidden_states[0][3])
test_list = [output_last_hidden_states[0][1],output_last_hidden_states[0][2]]
assert torch.equal(mapped_hidden_states[0][0], torch.mean(torch.stack(test_list), dim = 0))
"""adding special token hiddenstates"""
# assert torch.equal(mapped_hidden_states[0][2],output_last_hidden_states[0][3])
# test_list = [output_last_hidden_states[0][1],output_last_hidden_states[0][2]]
# assert torch.equal(mapped_hidden_states[0][1], torch.mean(torch.stack(test_list), dim = 0))
print()
print(pre_tokenized_text)
print(input_ids)
print(f'len original tokens: {len(pre_tokenized_text)}')
print(f'len input ids: {len(input_ids)}')
print([tokenizer.decode([id_]) for id_ in input_ids])
print('input idx mapping:', input_idx_mapping)

'''decoding'''
# TODO: need to train the decoder for decoding the hidden states back to text?


"""try encoder decoder setting"""
# config = BertConfig.from_pretrained('bert-base-uncased')
# config.is_decoder = True
# # model_decoder = BertModel(config)
# model_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased', config=config)
# # decode_output = model_decoder(encoder_hidden_states = mapped_hidden_states, return_dict = True)
# # print(decode_output)
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# outputs = model_decoder(**inputs)
# prediction_logits = outputs.logits
# probs = prediction_logits[0].softmax(dim = 1)
# print('probs size:', probs.size())
# values, predictions = probs.topk(1)
# print('predictions before squeeze:',predictions.size())
# predictions = torch.squeeze(predictions)
# print('predictions:',predictions)
# # print('target mask:', target_mask_batch[idx])
# # print('[DEBUG]target tokens:',tokenizer.decode(target_ids_batch_copy[idx]))
# print('[DEBUG]predicted tokens:',tokenizer.decode(predictions))

config = BertConfig.from_pretrained('bert-base-uncased')
config.is_decoder = True
model_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased', config = config)
pretrainedLMhead = model_decoder.cls
# print(PretrainedLMhead)
prediction_scores = pretrainedLMhead(mapped_hidden_states)
print('prediction scores size:',prediction_scores.size())

probs = prediction_scores[0].softmax(dim = 1)
print('probs size:', probs.size())
values, predictions = probs.topk(1)
print('predictions before squeeze:',predictions.size())
predictions = torch.squeeze(predictions)
print('predictions:',predictions)
# print('target mask:', target_mask_batch[idx])
# print('[DEBUG]target tokens:',tokenizer.decode(target_ids_batch_copy[idx]))
decoded_tokens = tokenizer.decode(predictions)
print(f'[DEBUG]predicted tokens:"{decoded_tokens}"')


