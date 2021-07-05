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

def train_model(dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=25, checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints/test.pt'):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000000


    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for input_EEG_embeddings, target_hiddenstates, target_token in tqdm(dataloaders[phase]):
                # load in batch
                input_EEG_embeddings = input_EEG_embeddings.to(device).float()
                target_hiddenstates = target_hiddenstates.to(device)
                
                # print('[DEBUG]',input_EEG_embeddings[0])
                # print('[DEBUG]',target_hiddenstates[0])

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                output = model(input_EEG_embeddings) # batch * 768
                
                '''loss'''
                assert output.size() == target_hiddenstates.size()
                loss = criterion(output, target_hiddenstates)
                if torch.isnan(loss):
                    print(input_EEG_embeddings) 
                    print(target_hiddenstates)
                    print('[nan ERROR!] EXIT')
                    quit()
                
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # print('[DEBUG]loss:',loss)

                # statistics
                running_loss += loss.item() * input_EEG_embeddings.size()[0] # batch loss
                

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'dev' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                '''save checkpoint'''
                torch.save(model.state_dict(), checkpoint_path)
                print(f'update checkpoint: {checkpoint_path}')
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    
    ''' config param'''
    num_epoch = 25
    batch_size = 64
    print(f'![Debug] using train batch size {batch_size}')
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
    # train dataset
    train_set = ZuCo_dataset_trainMapping(whole_dataset_dict, 'train', tokenizer, pretrained_encoder, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice)
    # dev dataset
    dev_set = ZuCo_dataset_trainMapping(whole_dataset_dict, 'dev', tokenizer, pretrained_encoder,subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice)
    # test dataset
    # test_set = ZuCo_dataset_trainMapping(whole_dataset_dict, 'test', tokenizer, pretrained_encoder,subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice)

    dataset_sizes = {'train': len(train_set), 'dev': len(dev_set)}
    print('[INFO]train_set size: ', len(train_set))
    print('[INFO]dev_set size: ', len(dev_set))
    
    # train dataloader
    train_dataloader = DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=4)
    # dev dataloader
    val_dataloader = DataLoader(dev_set, batch_size = 1, shuffle=False, num_workers=4)
    # dataloaders
    dataloaders = {'train':train_dataloader, 'dev':val_dataloader}

    ''' set up model '''
    model = EEG2BertMapping(in_feature = 840, hidden_size = 512, out_feature = 768)
    model.to(device)
    
    ''' training loop '''
    ''' set up optimizer and scheduler'''
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    ''' set up loss function '''
    # criterion = nn.L1Loss()
    criterion = nn.MSELoss()

    print('=== start training ... ===')
    # return best loss model from step1 training
    model = train_model(dataloaders, device, model, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epoch, checkpoint_path = output_checkpoint_name)
