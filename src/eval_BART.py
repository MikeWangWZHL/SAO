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

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
from data import ZuCo_dataset
from model import BrainTranslator

#TODO: not working, cannot treat the word level EEG data as a kind of word embedding?


def eval_model(dataloaders, device, tokenizer, criterion, model):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

    model.eval()   # Set model to evaluate mode
    running_loss = 0.0

    # Iterate over data.
    sample_count = 0    
    for input_embeddings, input_masks, input_mask_invert, target_ids, target_mask in dataloaders['test']:
        # load in batch
        input_embeddings_batch = input_embeddings.to(device).float()
        input_masks_batch = input_masks.to(device)
        target_ids_batch = target_ids.to(device)
        input_mask_invert_batch = input_mask_invert.to(device)
        if sample_count % 1 == 0:
            print('target tokens:',tokenizer.decode(target_ids_batch[0]))
        
        """replace padding ids in target_ids with -100"""
        target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100 

        # target_ids_batch_label = target_ids_batch.clone().detach()
        # target_ids_batch_label[target_ids_batch_label == tokenizer.pad_token_id] = -100

        # forward
        seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch)

        """calculate loss"""
        # logits = seq2seqLMoutput.logits # 8*48*50265
        # logits = logits.permute(0,2,1) # 8*50265*48

        # loss = criterion(logits, target_ids_batch_label) # calculate cross entropy loss only on encoded target parts
        # NOTE: my criterion not used
        loss = seq2seqLMoutput.loss # use the BART language modeling loss


        # check some output
        if sample_count % 1 == 0:
            # print('target size:', target_ids_batch.size(), ',original logits size:', logits.size())
            logits = seq2seqLMoutput.logits # 8*48*50265
            # logits = logits.permute(0,2,1)
            # print('permuted logits size:', logits.size())
            probs = logits[0].softmax(dim = 1)
            # print('probs size:', probs.size())
            values, predictions = probs.topk(1)
            # print('predictions before squeeze:',predictions.size())
            predictions = torch.squeeze(predictions)
            # print('predictions:',predictions)
            # print('target mask:', target_mask_batch[idx])

            print('predicted tokens:',tokenizer.decode(predictions))
            print('################################################')
            print()

        sample_count += 1
        # statistics
        running_loss += loss.item() * input_embeddings_batch.size()[0] # batch loss
        # print('[DEBUG]loss:',loss.item())
        # print('#################################')

    epoch_loss = running_loss / dataset_sizes['test_set']
    print('test loss: {:4f}'.format(epoch_loss))




if __name__ == '__main__':    
    ''' config param'''
    batch_size = 1
    
    subject_choice = 'ALL'
    # subject_choice = 'ZAB'
    # print(f'![Debug]using {subject_choice}')
    print(f'![Debug]using ZPH')
    eeg_type_choice = 'TRT'
    print(f'[INFO]eeg type {eeg_type_choice}')
    # bands_choice = ['_t1'] 
    bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] 
    print(f'[INFO]using bands {bands_choice}')
    dataset_setting = 'unique_subj'

    
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


    ''' set up dataloader '''
    # dataset_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset_skip_zerofixation.pickle' 
    dataset_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset-with-tokens_6-25.pickle' 
    with open(dataset_path, 'rb') as handle:
        whole_dataset_dict = pickle.load(handle)
    
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    # test dataset
    test_set = ZuCo_dataset(whole_dataset_dict, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)

    dataset_sizes = {"test_set":len(test_set)}
    print('[INFO]test_set size: ', len(test_set))
    
    # dataloaders
    test_dataloader = DataLoader(test_set, batch_size = 1, shuffle=False, num_workers=4)

    dataloaders = {'test':test_dataloader}

    ''' set up model '''
    # checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints/finetune_BART_EEG_feature_2steptraining_b32_20_15_51e-5_51e-7_use_label_unique_subj_setting.pt'
    checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints/last/finetune_BART_EEG_feature_2steptraining_b32_20_15_51e-5_51e-7_use_label_unique_subj_setting_7-1_no_PE_add_srcmask.pt'
    pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    # pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    model = BrainTranslator(pretrained_bart, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)
    # model = BrainTranslator(pretrained_bart, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=5, additional_encoder_dim_feedforward = 1024)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    ''' eval '''
    eval_model(dataloaders, device, tokenizer, criterion, model)

