import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import json
import matplotlib.pyplot as plt
from glob import glob
from transformers import BartTokenizer

def get_input_sample(sent_obj, tokenizer, eeg_type = 'TRT', bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], max_len = 48):
    
    def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands):
        frequency_features = []
        for band in bands:
            frequency_features.append(word_obj['word_level_EEG'][eeg_type][eeg_type+band])
        embedding = np.concatenate(frequency_features)
        assert len(embedding) == 105*len(bands)
        return torch.from_numpy(embedding)

    input_sample = {}
    # get target label
    target_string = sent_obj['content']
    target_tokenized = tokenizer(target_string, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt', return_attention_mask = True)
    input_sample['label'] = target_tokenized['input_ids'][0]
    
    # get input embeddings
    word_embeddings = []
    for word in sent_obj['word']:
        # add each word's EEG embedding as Tensors
        word_embeddings.append(get_word_embedding_eeg_tensor(word, eeg_type, bands = bands))
    # pad to max_len
    while len(word_embeddings) < max_len:
        word_embeddings.append(torch.zeros(105*len(bands)))

    input_sample['input_embeddings'] = torch.stack(word_embeddings) # max_len * (105*num_bands)

    # mask out padding tokens
    input_sample['input_attn_mask'] = torch.zeros(max_len) # 0 is masked out
    input_sample['input_attn_mask'][:len(sent_obj['word'])] = torch.ones(len(sent_obj['word'])) # 1 is not masked

    # mask out target padding for computing cross entropy loss
    input_sample['target_mask'] = target_tokenized['attention_mask'][0]

    return input_sample

class ZuCo_dataset(Dataset):
    def __init__(self, input_dataset_dict, phase, tokenizer, subject = 'ALL', eeg_type = 'TRT', bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2']):
        self.inputs = []
        self.tokenizer = tokenizer

        if subject == 'ALL':
            subjects = ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH'] # skip 'ZDN' 
        else:
            subjects = [subject]
        
        # take first 320 as trainset, 40 as dev and 40 as test
        if phase == 'train':
            print('[INFO]initializing a train set...')
            for key in subjects:
                for i in range(320):
                    self.inputs.append(get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands))
        elif phase == 'dev':
            print('[INFO]initializing a dev set...')
            for key in subjects:
                for i in range(320,360):
                    self.inputs.append(get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands))
        elif phase == 'test':
            print('[INFO]initializing a test set...')
            for key in subjects:
                for i in range(360,400):
                    self.inputs.append(get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands))
        
        print('[INFO]input size:', self.inputs[0]['input_embeddings'].size())
            
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        return input_sample['input_embeddings'], input_sample['input_attn_mask'], input_sample['label'], input_sample['target_mask']





"""for training mapping"""
def get_mapping_pairs_from_sentence(sent_obj, tokenizer, encoder, eeg_type = 'TRT', bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2']):
    
    #############################################################
    # helper functions
    def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands):
        frequency_features = []
        for band in bands:
            frequency_features.append(word_obj['word_level_EEG'][eeg_type][eeg_type+band])
        embedding = np.concatenate(frequency_features)
        assert len(embedding) == 105*len(bands)
        return torch.from_numpy(embedding)
    
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
                    mapped_token_hidden_states_list = [hidden_states[b][i] for i in mapping_list]
                    # take mean of all
                    mapped_hidden_states[b][t] = torch.mean(torch.stack(mapped_token_hidden_states_list), dim = 0)
                    
    return mapped_hidden_states
    #############################################################
    
    """get mapped hidden states"""
    # get tokens with fixation
    word_tokens_has_fixation = sent_obj['word_tokens_has_fixation']
    # get bert input ids
    inputs = tokenizer(word_tokens_has_fixation, is_split_into_words = True, return_tensors = 'pt')
    # get input ids mapping
    input_idx_mapping = [get_input_mapping(pre_tokenized_text,tokenizer)] # batch = 1
    # get pretrained hidden states
    output = encoder(**inputs, output_hidden_states = True, return_dict = True)
    # get mapped hidden states to
    output_last_hidden_states = output.last_hidden_state # 1 * seq_len_input_ids * 768
    mapped_hidden_states = get_mapped_hidden_states(output_last_hidden_states, input_idx_mapping) # 1 * seq_len_original * 768
    
    """get eeg embeddings"""
    eeg_embeddings = []
    assert len(sent_obj['word']) == len(word_tokens_has_fixation)
    for word_idx in range(len(sent_obj['word'])):
        word = sent_obj['word'][word_idx]
        assert word['content'] == word_tokens_has_fixation[word_idx]
        # add each word's EEG embedding as Tensors
        eeg_embeddings.append(get_word_embedding_eeg_tensor(word, eeg_type, bands = bands))
    
    assert len(mapped_hidden_states) == len(eeg_embeddings)
    # construct training pairs (eeg_vector, hidden_states)
    input_samples = []
    for i in range(len(mapped_hidden_states)):
        input_samples.append((eeg_embeddings[i], mapped_hidden_states[i]))
    return input_samples

class ZuCo_dataset_trainMapping(Dataset):
    def __init__(self, input_dataset_dict, phase, tokenizer, encoder, subject = 'ALL', eeg_type = 'TRT', bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2']):
        self.inputs = []
        self.tokenizer = tokenizer
        self.pretrained_encoder = encoder

        if subject == 'ALL':
            subjects = ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH'] # skip 'ZDN' 
        else:
            subjects = [subject]
        
        # take first 320 as trainset, 40 as dev and 40 as test
        if phase == 'train':
            print('[INFO]initializing a train set...')
            for key in subjects:
                for i in range(320):
                    mapping_pairs = get_mapping_pairs_from_sentence(input_dataset_dict[key][i],self.tokenizer, eeg_type = eeg_type, bands = bands)
                    for pair in mapping_pairs:
                        self.inputs.append(pair)
        elif phase == 'dev':
            print('[INFO]initializing a dev set...')
            for key in subjects:
                for i in range(320,360):
                    mapping_pairs = get_mapping_pairs_from_sentence(input_dataset_dict[key][i],self.tokenizer, eeg_type = eeg_type, bands = bands)
                    for pair in mapping_pairs:
                        self.inputs.append(pair)
        elif phase == 'test':
            print('[INFO]initializing a test set...')
            for key in subjects:
                for i in range(360,400):
                    mapping_pairs = get_mapping_pairs_from_sentence(input_dataset_dict[key][i],self.tokenizer, eeg_type = eeg_type, bands = bands)
                    for pair in mapping_pairs:
                        self.inputs.append(pair)
        print('[INFO]input size:', self.inputs[0]['input_embeddings'].size())
            
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        return input_sample[0], input_sample[1] # (eeg feature, hiddenstates feature)

'''sanity test'''
# dataset_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset_skip_zerofixation.pickle' 
# with open(dataset_path, 'rb') as handle:
#     whole_dataset_dict = pickle.load(handle)

# trainset = ZuCo_dataset(whole_dataset_dict, 'train', subject = 'ALL', eeg_type = 'FFD')
# print('size of trainset:',len(trainset))
# print('size of input embeddings:', trainset[0][0].size())
# print('size of input attention mask:', trainset[0][1].size())
# print('size of target ids:', trainset[0][2].size())



# max_token_len = 0
# for i in range(400):
#     key = 'ZAB'
#     # print(whole_dataset[key][i]['content'])
#     # print(len(whole_dataset[key][i]['word']))
#     if len(whole_dataset[key][i]['word']) > max_token_len:
#         max_token_len = len(whole_dataset[key][i]['word']) 
# print(max_token_len)
# for i in range(400):
#     for key in ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH']:
#         print(whole_dataset[key][i]['content'])
#     print('#############################################')
        # print(f'how many valid sentences [{key}]: {len(whole_dataset[key])}')
        # print(f'subject:{key}, first sentence: ', whole_dataset[key][0]['content'])
