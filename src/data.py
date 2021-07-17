import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import json
import matplotlib.pyplot as plt
from glob import glob
from transformers import BartTokenizer
from tqdm import tqdm

# TODO: add sentence level EEG tensor

# macro
sentiment_labels = json.load(open('/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task1-SR/sentiment_labels/sentiment_labels.json'))


def normalize_1d(input_tensor):
    # normalize a 1d tensor
    mean = torch.mean(input_tensor)
    std = torch.std(input_tensor)
    input_tensor = (input_tensor - mean)/std
    return input_tensor 


def get_input_sample(sent_obj, tokenizer, eeg_type = 'GD', bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], max_len = 56, add_CLS_token = False):
    
    def get_word_embedding_eeg_tensor(word_obj, eeg_type, bands):
        frequency_features = []
        for band in bands:
            frequency_features.append(word_obj['word_level_EEG'][eeg_type][eeg_type+band])
        word_eeg_embedding = np.concatenate(frequency_features)
        if len(word_eeg_embedding) != 105*len(bands):
            print(f'expect word eeg embedding dim to be {105*len(bands)}, but got {len(word_eeg_embedding)}, return None')
            return None
        # assert len(word_eeg_embedding) == 105*len(bands)
        return_tensor = torch.from_numpy(word_eeg_embedding)
        return normalize_1d(return_tensor)

    def get_sent_eeg(sent_obj, bands):
        sent_eeg_features = []
        for band in bands:
            key = 'mean'+band
            sent_eeg_features.append(sent_obj['sentence_level_EEG'][key])
        sent_eeg_embedding = np.concatenate(sent_eeg_features)
        assert len(sent_eeg_embedding) == 105*len(bands)
        return_tensor = torch.from_numpy(sent_eeg_embedding)
        return normalize_1d(return_tensor)

    if sent_obj is None:
        # print(f'  - skip bad sentence')   
        return None

    input_sample = {}
    # get target label
    target_string = sent_obj['content']
    target_tokenized = tokenizer(target_string, padding='max_length', max_length=max_len, truncation=True, return_tensors='pt', return_attention_mask = True)
    
    input_sample['target_ids'] = target_tokenized['input_ids'][0]
    
    # get sentence level EEG features
    sent_level_eeg_tensor = get_sent_eeg(sent_obj, bands)
    if torch.isnan(sent_level_eeg_tensor).any():
        # print('[NaN sent level eeg]: ', target_string)
        return None
    input_sample['sent_level_EEG'] = sent_level_eeg_tensor

    # get sentiment label
    # handle some wierd case
    if 'emp11111ty' in target_string:
        target_string = target_string.replace('emp11111ty','empty')
    if 'film.1' in target_string:
        target_string = target_string.replace('film.1','film.')
    
    if target_string in sentiment_labels:
        input_sample['sentiment_label'] = torch.tensor(sentiment_labels[target_string]+1) # 0:Negative, 1:Neutral, 2:Positive
    else:
        input_sample['sentiment_label'] = torch.tensor(-100) # dummy value

    # get input embeddings
    word_embeddings = []

    """add CLS token embedding at the front"""
    if add_CLS_token:
        word_embeddings.append(torch.ones(105*len(bands)))

    for word in sent_obj['word']:
        # add each word's EEG embedding as Tensors
        word_level_eeg_tensor = get_word_embedding_eeg_tensor(word, eeg_type, bands = bands)
        # check none, for v2 dataset
        if word_level_eeg_tensor is None:
            return None
        # check nan:
        if torch.isnan(word_level_eeg_tensor).any():
            # print()
            # print('[NaN ERROR] problem sent:',sent_obj['content'])
            # print('[NaN ERROR] problem word:',word['content'])
            # print('[NaN ERROR] problem word feature:',word_level_eeg_tensor)
            # print()
            return None
            

        word_embeddings.append(word_level_eeg_tensor)
    # pad to max_len
    while len(word_embeddings) < max_len:
        word_embeddings.append(torch.zeros(105*len(bands)))

    input_sample['input_embeddings'] = torch.stack(word_embeddings) # max_len * (105*num_bands)

    # mask out padding tokens
    input_sample['input_attn_mask'] = torch.zeros(max_len) # 0 is masked out

    if add_CLS_token:
        input_sample['input_attn_mask'][:len(sent_obj['word'])+1] = torch.ones(len(sent_obj['word'])+1) # 1 is not masked
    else:
        input_sample['input_attn_mask'][:len(sent_obj['word'])] = torch.ones(len(sent_obj['word'])) # 1 is not masked
    

    # mask out padding tokens reverted: handle different use case: this is for pytorch transformers
    input_sample['input_attn_mask_invert'] = torch.ones(max_len) # 1 is masked out

    if add_CLS_token:
        input_sample['input_attn_mask_invert'][:len(sent_obj['word'])+1] = torch.zeros(len(sent_obj['word'])+1) # 0 is not masked
    else:
        input_sample['input_attn_mask_invert'][:len(sent_obj['word'])] = torch.zeros(len(sent_obj['word'])) # 0 is not masked

    

    # mask out target padding for computing cross entropy loss
    input_sample['target_mask'] = target_tokenized['attention_mask'][0]
    input_sample['seq_len'] = len(sent_obj['word'])
    
    # clean 0 length data
    if input_sample['seq_len'] == 0:
        print('discard length zero instance: ', target_string)
        return None

    return input_sample

class ZuCo_dataset(Dataset):
    def __init__(self, input_dataset_dicts, phase, tokenizer, subject = 'ALL', eeg_type = 'GD', bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'], setting = 'unique_sent', is_add_CLS_token = False):
        self.inputs = []
        self.tokenizer = tokenizer

        if not isinstance(input_dataset_dicts,list):
            input_dataset_dicts = [input_dataset_dicts]
        print(f'[INFO]loading {len(input_dataset_dicts)} task datasets')
        for input_dataset_dict in input_dataset_dicts:
            if subject == 'ALL':
                subjects = list(input_dataset_dict.keys())
                print('[INFO]using subjects: ', subjects)
                # if version == 'v1':
                #     subjects = ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH'] # skip 'ZDN' 
                # elif version == 'v2':
                #     subjects = ['YSD','YAC','YDG','YLS','YMS','YHS','YIS','YSL','YFR','YTL','YRH','YAG','YFS','YRK','YAK','YRP','YDR','YMD']
            else:
                subjects = [subject]
            
            total_num_sentence = len(input_dataset_dict[subjects[0]])
            
            train_divider = int(0.8*total_num_sentence)
            dev_divider = train_divider + int(0.1*total_num_sentence)
            
            print(f'train divider = {train_divider}')
            print(f'dev divider = {dev_divider}')

            if setting == 'unique_sent':
                # take first 320 as trainset, 40 as dev and 40 as test
                if phase == 'train':
                    print('[INFO]initializing a train set...')
                    for key in subjects:
                        for i in range(train_divider):
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                elif phase == 'dev':
                    print('[INFO]initializing a dev set...')
                    for key in subjects:
                        for i in range(train_divider,dev_divider):
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                elif phase == 'test':
                    print('[INFO]initializing a test set...')
                    for key in subjects:
                        for i in range(dev_divider,total_num_sentence):
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
            elif setting == 'unique_subj':
                print('warning!!! only implemented for v1 dataset ')
                # subject ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW'] for train
                # subject ['ZMG'] for dev
                # subject ['ZPH'] for test
                if phase == 'train':
                    print(f'[INFO]initializing a train set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH','ZKW']:
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                if phase == 'dev':
                    print(f'[INFO]initializing a dev set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in ['ZMG']:
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
                if phase == 'test':
                    print(f'[INFO]initializing a test set using {setting} setting...')
                    for i in range(total_num_sentence):
                        for key in ['ZPH']:
                            input_sample = get_input_sample(input_dataset_dict[key][i],self.tokenizer,eeg_type,bands = bands, add_CLS_token = is_add_CLS_token)
                            if input_sample is not None:
                                self.inputs.append(input_sample)
            print('++ adding to dataset, now we have:', len(self.inputs))

        print('[INFO]input size:', self.inputs[0]['input_embeddings'].size())
        print()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        return (
            input_sample['input_embeddings'], 
            input_sample['seq_len'],
            input_sample['input_attn_mask'], 
            input_sample['input_attn_mask_invert'],
            input_sample['target_ids'], 
            input_sample['target_mask'], 
            input_sample['sentiment_label'], 
            input_sample['sent_level_EEG']
        )
        # keys: input_embeddings, input_attn_mask, input_attn_mask_invert, target_ids, target_mask, 
        




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
        return_tensor = torch.from_numpy(embedding)
        return normalize_1d(return_tensor)

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
        assert batch_size == 1

        mapped_hidden_states = []
        
        for b in range(batch_size):
            for mapping_list in idx_mapping[b]:
                if len(mapping_list) == 1:
                    # mapping to exactly one token:
                    idx = mapping_list[0]
                    mapped_hidden_states.append(hidden_states[b][idx]) # 768
                else:
                    mapped_token_hidden_states_list = [hidden_states[b][i] for i in mapping_list]
                    # take mean of all
                    mapped_hidden_states.append(torch.mean(torch.stack(mapped_token_hidden_states_list), dim = 0))
                    
        return mapped_hidden_states
    #############################################################
    
    """get mapped hidden states"""
    # get tokens with fixation
    word_tokens_has_fixation = sent_obj['word_tokens_has_fixation']
    if len(word_tokens_has_fixation) < 1:
        return []
    # get bert input ids
    inputs = tokenizer(word_tokens_has_fixation, is_split_into_words = True, return_tensors = 'pt')
    # get input ids mapping
    input_idx_mapping = [get_input_mapping(word_tokens_has_fixation,tokenizer)] # batch = 1
    # get pretrained hidden states
    output = encoder(**inputs, output_hidden_states = True, return_dict = True)
    # get mapped hidden states to
    output_last_hidden_states = output.last_hidden_state.detach() # 1 * seq_len_input_ids * 768
    mapped_hidden_states = get_mapped_hidden_states(output_last_hidden_states, input_idx_mapping) # 1 * seq_len_original * 768
    
    """get eeg embeddings"""
    target_tokens = []
    eeg_embeddings = []
    assert len(sent_obj['word']) == len(word_tokens_has_fixation)
    for word_idx in range(len(sent_obj['word'])):
        word = sent_obj['word'][word_idx]
        assert word['content'] == word_tokens_has_fixation[word_idx]
        # add each word's EEG embedding as Tensors
        word_level_eeg_tensor = get_word_embedding_eeg_tensor(word, eeg_type, bands = bands)

        # check nan:
        if torch.isnan(word_level_eeg_tensor).any():
            print()
            print('[NaN ERROR] problem sent:',sent_obj['content'])
            print('[NaN ERROR] problem word:',word['content'])
            # print('[NaN ERROR] problem word feature:',word_level_eeg_tensor)
            print()
            return []

        eeg_embeddings.append(word_level_eeg_tensor)
        target_tokens.append(word['content'])

    assert len(mapped_hidden_states) == len(eeg_embeddings)
    assert len(mapped_hidden_states) == len(target_tokens)
    # construct training pairs (eeg_vector, hidden_states)
    input_samples = []
    for i in range(len(mapped_hidden_states)):
        input_samples.append((eeg_embeddings[i], mapped_hidden_states[i], target_tokens[i]))
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
            for key in tqdm(subjects):
                for i in range(320):
                    mapping_pairs = get_mapping_pairs_from_sentence(input_dataset_dict[key][i],self.tokenizer,self.pretrained_encoder, eeg_type = eeg_type, bands = bands)
                    if mapping_pairs != []:
                        for pair in mapping_pairs:
                            self.inputs.append(pair)
            print(f'[INFO]eeg_embedding_size: {len(self.inputs[0][0])}, target_hiddenstates_size: {len(self.inputs[0][1])}')
        elif phase == 'dev':
            print('[INFO]initializing a dev set...')
            for key in tqdm(subjects):
                for i in range(320,360):
                    mapping_pairs = get_mapping_pairs_from_sentence(input_dataset_dict[key][i],self.tokenizer,self.pretrained_encoder, eeg_type = eeg_type, bands = bands)
                    if mapping_pairs != []:
                        for pair in mapping_pairs:
                            self.inputs.append(pair)
            print(f'[INFO]eeg_embedding_size: {len(self.inputs[0][0])}, target_hiddenstates_size: {len(self.inputs[0][1])}')
        elif phase == 'test':
            print('[INFO]initializing a test set...')
            for key in tqdm(subjects):
                for i in range(360,400):
                    mapping_pairs = get_mapping_pairs_from_sentence(input_dataset_dict[key][i],self.tokenizer,self.pretrained_encoder, eeg_type = eeg_type, bands = bands)
                    if mapping_pairs != []:
                        for pair in mapping_pairs:
                            self.inputs.append(pair)
            
    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_sample = self.inputs[idx]
        return input_sample[0], input_sample[1], input_sample[2] # (eeg feature, hiddenstates feature, target_token)

'''sanity test'''
if __name__ == '__main__':
    whole_dataset_dicts = []
    
    dataset_path_task1 = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset-with-tokens_6-25.pickle' 
    with open(dataset_path_task1, 'rb') as handle:
        whole_dataset_dicts.append(pickle.load(handle))

    dataset_path_task2 = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task2-NR/pickle/task2-NR-dataset-with-tokens_7-10.pickle' 
    with open(dataset_path_task2, 'rb') as handle:
        whole_dataset_dicts.append(pickle.load(handle))

    # dataset_path_task3 = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset-with-tokens_7-10.pickle' 
    # with open(dataset_path_task3, 'rb') as handle:
    #     whole_dataset_dicts.append(pickle.load(handle))

    dataset_path_task2_v2 = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset-with-tokens_7-15.pickle' 
    with open(dataset_path_task2_v2, 'rb') as handle:
        whole_dataset_dicts.append(pickle.load(handle))

    print()
    for key in whole_dataset_dicts[0]:
        print(f'task2_v2, sentence num in {key}:',len(whole_dataset_dicts[0][key]))
    print()

    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    dataset_setting = 'unique_sent'
    subject_choice = 'ALL'
    # subject_choice = 'ZAB'
    print(f'![Debug]using {subject_choice}')
    eeg_type_choice = 'GD'
    print(f'[INFO]eeg type {eeg_type_choice}')
    # bands_choice = ['_t1'] 
    bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] 
    print(f'[INFO]using bands {bands_choice}')
    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
    dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
    # print(train_set[0])
    # print(train_set[1])
    # print(train_set[2])
    print('trainset size:',len(train_set))
    print('devset size:',len(dev_set))
    print('testset size:',len(test_set))
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
