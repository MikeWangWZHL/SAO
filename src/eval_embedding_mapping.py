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
from data import ZuCo_dataset, ZuCo_dataset_trainMapping
from model import EEG2BertMapping

def show_prediction(pretrainedLMhead, mapped_hidden_states, target_tokens):
    print('output size:',mapped_hidden_states.size())
    prediction_scores = pretrainedLMhead(mapped_hidden_states)
    print('prediction_scores size:',prediction_scores.size())
    # probs = prediction_scores[0].softmax(dim = 1) # perform on [batch_size*seq_len*vocab_size] tensor
    probs = prediction_scores.softmax(dim = 1) # perform on [batch_size*vocab_size] tensor
    print('probs size:', probs.size())
    values, predictions = probs.topk(1)
    print('predictions before squeeze:',predictions.size())
    predictions = torch.squeeze(predictions)
    print('predictions:',predictions)
    # print('target mask:', target_mask_batch[idx])
    # print('[DEBUG]target tokens:',tokenizer.decode(target_ids_batch_copy[idx]))
    decoded_tokens = tokenizer.decode(predictions)
    print(f'predicted tokens:"{decoded_tokens}"')
    print(f'target tokens:"{target_tokens}"')

def eval_model(dataloaders, device, model, criterion):
    """set up pretrained LM for prediction"""
    config = BertConfig.from_pretrained('bert-base-uncased')
    config.is_decoder = True
    model_decoder = BertLMHeadModel.from_pretrained('bert-base-uncased', config = config)
    pretrainedLMhead = model_decoder.cls
    pretrainedLMhead.to(device)
    
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    model.eval()   # Set model to evaluate mode

    running_loss = 0.0

    phase = 'test'

    count = 0
    # Iterate over data.
    for input_EEG_embeddings, target_hiddenstates, target_tokens in tqdm(dataloaders[phase]):
        # target_tokens: string

        # load in batch
        input_EEG_embeddings = input_EEG_embeddings.to(device).float()
        target_hiddenstates = target_hiddenstates.to(device)

        # forward
        output = model(input_EEG_embeddings) # batch * 768
        
        if count % 100 == 0:
            show_prediction(pretrainedLMhead, output, target_tokens[0])
        
        '''loss'''
        assert output.size() == target_hiddenstates.size()
        loss = criterion(output, target_hiddenstates)
        if torch.isnan(loss):
            print(input_EEG_embeddings) 
            print(target_hiddenstates)
            print('[nan ERROR!] EXIT')
            quit()
        
        # statistics
        running_loss += loss.item() * input_EEG_embeddings.size()[0] # batch loss
        count += 1
        
    epoch_loss = running_loss / dataset_sizes[phase]
    print('{} Loss: {:.4f}'.format(phase, epoch_loss))

if __name__ == '__main__':
    
    ''' config param'''
    batch_size = 1
    print(f'![Debug] using batch size {batch_size}')
    save_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints' 
    output_checkpoint_name = save_path + '/EEG_to_bert_hiddenstates_mapping_net.pt'
    
    subject_choice = 'ALL'
    # subject_choice = 'ZAB'
    print(f'![Debug]using {subject_choice}')
    eeg_type_choice = 'TRT'
    print(f'[INFO]eeg type {eeg_type_choice}')
    # bands_choice = ['_t1'] 
    bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] 
    print(f'[INFO]using bands {bands_choice}')


    
    ''' set random seeds '''
    seed_val = 312
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


    ''' set up device '''
    # use cuda
    if torch.cuda.is_available():  
        dev = "cuda:3" 
    else:  
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3  
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')

    '''set up tokenizer and pretrained encoder'''
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    pretrained_encoder = BertModel.from_pretrained('bert-base-uncased')

    ''' set up dataloader '''
    dataset_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset-with-tokens_6-25.pickle' 
    with open(dataset_path, 'rb') as handle:
        whole_dataset_dict = pickle.load(handle)
    
    # test dataset
    test_set = ZuCo_dataset_trainMapping(whole_dataset_dict, 'test', tokenizer, pretrained_encoder,subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice)

    dataset_sizes = {'test':len(test_set)}
    print('[INFO]test_set size: ', len(test_set))
    
    test_dataloader = DataLoader(test_set, batch_size = 1, shuffle=False, num_workers=4)
    # dataloaders
    dataloaders = {'test':test_dataloader}

    ''' set up model '''
    checkpoint_path = "/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints/EEG_to_bert_hiddenstates_mapping_net.pt"
    model = EEG2BertMapping(in_feature = 840, hidden_size = 512, out_feature = 768)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    
    ''' set up loss function '''
    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()

    print('=== start eval ... ===')
    # return best loss model from step1 training
    model = eval_model(dataloaders, device, model, criterion)
