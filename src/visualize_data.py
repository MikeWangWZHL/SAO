import os
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
import json
import matplotlib.pyplot as plt
from glob import glob
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from textwrap import wrap
from collections import defaultdict
from numpy import linalg
from tqdm import tqdm



dataset_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset_skip_zerofixation.pickle' 
with open(dataset_path, 'rb') as handle:
    whole_dataset = pickle.load(handle)


def get_word_embedding_eeg_features(word_obj, eeg_type, bands):
    frequency_features = []
    for band in bands:
        frequency_features.append(word_obj['word_level_EEG'][eeg_type][eeg_type+band])
    return frequency_features

def plot_word_level_EEG(features, bands, word_content, word_idx, sent_content, sent_idx, subject):
    assert len(features) == len(bands)
    fig, axs = plt.subplots(len(features),1,figsize=(10,20))
    sent_content = "\n".join(wrap(sent_content, 20))
    fig.suptitle(f'sent: {sent_content}\n\nword: {word_content}\n\n\n\n')
    fig.tight_layout()
    for i in range(len(features)):
        # axs[i].hist(features[i], 105, density=False, facecolor='g', alpha=0.75)
        axs[i].plot(features[i])
        axs[i].set_title(f'band: {bands[i]}')
    fig.savefig(f'/shared/nas/data/m1/wangz3/SAO_project/SAO/visualization/EEG/{sent_idx}_{word_idx}_{word_content}_{subject}_EEG.png')

def plot_word_level_EEG_concatenate(features, bands, word_content, word_idx, sent_content, sent_idx, subject):
    assert len(features) == len(bands)
    feature_concatenate = np.concatenate(features)
    fig, ax = plt.subplots(1, figsize = (20,8))
    sent_content = "\n".join(wrap(sent_content, 100))
    fig.suptitle(f'sent: {sent_content}\n\nword: {word_content}\n\n\n\n')
    ax.plot(feature_concatenate)
    fig.savefig(f'/shared/nas/data/m1/wangz3/SAO_project/SAO/visualization/EEG/{sent_idx}_{word_idx}_{word_content}_{subject}_EEGconcat.png')


def plot_word_embedding(features, word_content, word_idx, sent_content,  sent_idx, subject):
    if word_content in ['<s>','</s>',',','.']:
        return 
    fig, ax = plt.subplots(1, figsize = (20,8))
    sent_content = "\n".join(wrap(sent_content, 100))
    fig.suptitle(f'sent: {sent_content}\n\nword: {word_content}\n\n\n\n')
    ax.plot(features)
    fig.savefig(f'/shared/nas/data/m1/wangz3/SAO_project/SAO/visualization/Bart/{sent_idx}_{word_idx}_{word_content}_{subject}_Bart.png')

def add_neighbor_word_EEG(neighbor_word_dict, sent_obj, sent_idx, word_idx, word_content, bands, eeg_type = 'TRT'):
    if word_idx == len(sent_obj['word'])-1:
        neighbor_word_dict['EEG'][word_content].append((sent_idx,word_idx-1,sent_obj['word'][word_idx-1]['content'],np.concatenate(get_word_embedding_eeg_features(sent_obj['word'][word_idx-1], eeg_type = eeg_type, bands = bands))))
    else:
        neighbor_word_dict['EEG'][word_content].append((sent_idx,word_idx+1,sent_obj['word'][word_idx+1]['content'],np.concatenate(get_word_embedding_eeg_features(sent_obj['word'][word_idx+1], eeg_type = eeg_type, bands = bands))))

def add_neighbor_word_BART(neighbor_word_dict, input_ids, sent_idx, word_idx, word_content, tokenizer, bart_embedding):
    if word_idx == len(input_ids)-1:
        nb_token_idx = word_idx - 1
    else:
        nb_token_idx = word_idx + 1
    nb_token_id = input_ids[nb_token_idx]
    nb_token_content = tokenizer.decode(nb_token_id)
    nb_token_embedding = bart_embedding(nb_token_id).detach().cpu().numpy()
    neighbor_word_dict['BART'][word_content].append((sent_idx,nb_token_idx,nb_token_content,nb_token_embedding))



# Bart
pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
bart_embedding = pretrained_bart.get_input_embeddings()
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

# word dict
same_word_dict = {'EEG':defaultdict(list), 'BART':defaultdict(list)}
neighbor_word_dict = {'EEG':defaultdict(list), 'BART':defaultdict(list)}

print('start constructing dict...')
for i in tqdm(range(400)):
    # for key in ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH']:
    for key in ['ZAB']:
        sent_obj = whole_dataset[key][i]
        sent_content = sent_obj['content']
        for word_idx in range(len(sent_obj['word'])):
            word = sent_obj['word'][word_idx]
            word_content = word['content']
            bands = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2']
            eeg_type = 'TRT'
            # bands = ['_g2']
            word_frequency_features = get_word_embedding_eeg_features(word, eeg_type = eeg_type, bands = bands)
            # print(word_frequency_features[0])
            """plot word EEG"""
            # plot_word_level_EEG(word_frequency_features, bands, word_content, word_idx, sent_content, i, key)
            # plot_word_level_EEG_concatenate(word_frequency_features, bands, word_content, word_idx, sent_content, i, key)
            
            """add to word dict"""
            same_word_dict['EEG'][word_content].append((i,word_idx,np.concatenate(word_frequency_features)))
            add_neighbor_word_EEG(neighbor_word_dict, sent_obj, i, word_idx, word_content, bands, eeg_type = eeg_type)

        # check Bart embedding visualization
        input_ids = tokenizer(sent_content, truncation=True, return_tensors='pt')['input_ids'][0]
        for token_idx in range(len(input_ids)):
            token_id = input_ids[token_idx]
            token_content = tokenizer.decode(token_id)
            # print('token_content:', token_content)
            token_embedding = bart_embedding(token_id).detach().cpu().numpy()
            # print('token_embedding_size:', token_embedding.shape)
            # plot_word_embedding(token_embedding, token_content, token_idx, sent_content, i, key)
            
            """add to word dict"""
            same_word_dict['BART'][token_content].append((i,token_idx,token_embedding))
            add_neighbor_word_BART(neighbor_word_dict, input_ids, i, token_idx, token_content, tokenizer, bart_embedding)


print('====================================================')
# print(same_word_dict['EEG'].keys()) 
print('word num EEG:', len(same_word_dict['EEG'].keys())) 
print('same word <and>:', len(same_word_dict['EEG']['and']))
print('neighbor word <and>:',len(neighbor_word_dict['EEG']['and']))
print('====================================================')
# print(same_word_dict['BART'].keys()) 
print('token num BART:', len(same_word_dict['BART'].keys())) 
print('same word < and>:', len(same_word_dict['BART'][' and']))
print('neighbor word < and>:',len(neighbor_word_dict['BART'][' and']))
print('====================================================')
print('====================================================')


print()
print()
print('check sim EEG:')
"""check similarity"""
# EEG
show_count = 0
for word_content, same_words in same_word_dict['EEG'].items():
    if len(same_words) > 1:
        src_word = same_words[0] # (sent_idx, word_idx, feature)
        target_word = same_words[1] # (sent_idx, word_idx, feature)
        nb_word = neighbor_word_dict['EEG'][word_content][0] # (sent_idx, word_idx, content, feature)

        # normalize feature
        src_feature_normalized = src_word[2] / linalg.norm(src_word[2])
        target_feature_normalized = target_word[2] / linalg.norm(target_word[2])
        nb_feature_normalized = nb_word[3] / linalg.norm(nb_word[3])

        # calculate l2 distance
        l2_distance_same = linalg.norm(src_feature_normalized - target_feature_normalized) 
        l2_distance_nb = linalg.norm(src_feature_normalized - nb_feature_normalized) 
        print(f'L2 distance same ({src_word[0]}_{src_word[1]}_{word_content} vs {target_word[0]}_{target_word[1]}_{word_content}) = {l2_distance_same}',)
        print(f'L2 distance neighbor ({src_word[0]}_{src_word[1]}_{word_content} vs {nb_word[0]}_{nb_word[1]}_{nb_word[2]}) = {l2_distance_nb}',)
        print('#####################################################################################')
        show_count += 1
    if show_count == 10:
        break
# BART
print()
print()
print('check sim BART embedding layer:')
show_count = 0
for word_content, same_words in same_word_dict['BART'].items():
    if len(same_words) > 1:
        src_word = same_words[0] # (sent_idx, word_idx, feature)
        target_word = same_words[1] # (sent_idx, word_idx, feature)
        nb_word = neighbor_word_dict['BART'][word_content][0] # (sent_idx, word_idx, content, feature)

        # normalize
        src_feature_normalized = src_word[2] / linalg.norm(src_word[2])
        target_feature_normalized = target_word[2] / linalg.norm(target_word[2])
        nb_feature_normalized = nb_word[3] / linalg.norm(nb_word[3])

        # calculate l2 distance
        l2_distance_same = linalg.norm(src_feature_normalized - target_feature_normalized) 
        l2_distance_nb = linalg.norm(src_feature_normalized - nb_feature_normalized)
        print(f'L2 distance same ({src_word[0]}_{src_word[1]}_{word_content} vs {target_word[0]}_{target_word[1]}_{word_content}) = {l2_distance_same}',)
        print(f'L2 distance neighbor ({src_word[0]}_{src_word[1]}_{word_content} vs {nb_word[0]}_{nb_word[1]}_{nb_word[2]}) = {l2_distance_nb}',)
        print('#####################################################################################')
        show_count += 1
    if show_count == 10:
        break
    

