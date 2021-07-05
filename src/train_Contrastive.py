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

from transformers import BertTokenizer, BertConfig, BertModel
from data import ZuCo_dataset
from model import ContrastiveBrainTextEncoder

# NOTE: not quite working ...

def train_model(dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=25, checkpoint_path_best = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints/best/test.pt',checkpoint_path_last = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints/last/test.pt'):
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
            for input_EEG_features, _, input_EEG_masks, input_text_ids, input_text_attention_masks in tqdm(dataloaders[phase]):
                # load in batch
                input_EEG_features = input_EEG_features.to(device).float()
                input_EEG_masks = input_EEG_masks.to(device)
                input_text_ids = input_text_ids.to(device)
                input_text_attention_masks = input_text_attention_masks.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                logits_per_EEG, logits_per_text = model(input_EEG_features, input_EEG_masks, input_text_ids, input_text_attention_masks)

                assert logits_per_EEG.size()[0] == input_EEG_features.size()[0]
                assert logits_per_EEG.size()[1] == input_EEG_features.size()[0]
                assert logits_per_text.size()[0] == input_text_ids.size()[0]
                assert logits_per_text.size()[1] == input_text_ids.size()[0]

                '''loss'''
                labels = torch.arange(logits_per_EEG.size()[0]).to(device)
                # criterion = cross entropy loss
                loss_eeg = criterion(logits_per_EEG, labels)
                loss_text = criterion(logits_per_text, labels)
                loss = loss_eeg + loss_text

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * input_EEG_features.size()[0] # batch loss

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]

            print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'dev' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                '''save checkpoint'''
                torch.save(model.state_dict(), checkpoint_path_best)
                print(f'update best checkpoint: {checkpoint_path_best}')
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    torch.save(model.state_dict(), checkpoint_path_last)
    print(f'update last checkpoint: {checkpoint_path_last}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == '__main__':
    
    ''' config param'''
    num_epochs_step1 = 10
    num_epochs_step2 = 10
    step1_lr = 1e-5
    step2_lr = 1e-6

    batch_size = 32
    dataset_setting = 'unique_subj'
    # print('![Debug] using train batch size 1')
    save_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints' 
    model_name = f'contrastive_learning_b{batch_size}_1e-5_1e-6_{dataset_setting}'
    output_checkpoint_name_best = save_path + f'/best/{model_name}.pt' 
    output_checkpoint_name_last = save_path + f'/last/{model_name}.pt' 
    output_log_file_name = f'/shared/nas/data/m1/wangz3/SAO_project/SAO/log/{model_name}.txt'


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

    '''set up tokenizer'''
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    
    ''' set up dataloader '''
    dataset_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset-with-tokens_6-25.pickle' 
    with open(dataset_path, 'rb') as handle:
        whole_dataset_dict = pickle.load(handle)
    # train dataset
    train_set = ZuCo_dataset(whole_dataset_dict, 'train', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice,setting = dataset_setting)
    # dev dataset
    dev_set = ZuCo_dataset(whole_dataset_dict, 'dev', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice,setting = dataset_setting)
    # test dataset
    # test_set = ZuCo_dataset(whole_dataset_dict, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice)

    dataset_sizes = {'train': len(train_set), 'dev': len(dev_set)}
    print('[INFO]train_set size: ', len(train_set))
    print('[INFO]dev_set size: ', len(dev_set))
    
    # train dataloader
    train_dataloader = DataLoader(train_set, batch_size = batch_size, shuffle=True, num_workers=4)
    # dev dataloader
    val_dataloader = DataLoader(dev_set, batch_size = batch_size, shuffle=False, num_workers=4)
    # dataloaders
    dataloaders = {'train':train_dataloader, 'dev':val_dataloader}

    ''' set up model '''
    pretrained_bert = BertModel.from_pretrained('bert-base-cased')
    model = ContrastiveBrainTextEncoder(pretrained_bert, in_feature = 105*len(bands_choice), embed_dim = 768)
    model.to(device)
    




    ''' training loop '''
    ######################################################
    '''step one trainig: freeze most of BERT params'''
    ######################################################
    # closely follow BART paper
    for name, param in model.named_parameters():
        if param.requires_grad and 'TextEncoder' in name:
            param.requires_grad = False

    # sanity check
    for name, param in model.named_parameters():
        if param.requires_grad:
            print('step1 training:',name)

    ''' set up optimizer and scheduler'''
    optimizer_step1 = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=step1_lr, momentum=0.9)

    exp_lr_scheduler_step1 = lr_scheduler.StepLR(optimizer_step1, step_size=10, gamma=0.1)

    ''' set up loss function '''
    criterion = nn.CrossEntropyLoss()

    print('=== start Step1 training ... ===')
    # return best loss model from step1 training
    model = train_model(dataloaders, device, model, criterion, optimizer_step1, exp_lr_scheduler_step1, num_epochs=num_epochs_step1, checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last)
    
    ######################################################
    '''step two trainig: update whole model for a few iterations'''
    ######################################################
    for name, param in model.named_parameters():
        param.requires_grad = True

    ''' set up optimizer and scheduler'''
    optimizer_step2 = optim.SGD(model.parameters(), lr=step2_lr, momentum=0.9)

    exp_lr_scheduler_step2 = lr_scheduler.StepLR(optimizer_step2, step_size=10, gamma=0.1)

    ''' set up loss function '''
    criterion = nn.CrossEntropyLoss()
    
    print()
    print('=== start Step2 training ... ===')
    trained_model = train_model(dataloaders, device, model, criterion, optimizer_step2, exp_lr_scheduler_step2, num_epochs=num_epochs_step2, checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last)

    # '''save checkpoint'''
    # torch.save(trained_model.state_dict(), os.path.join(save_path,output_checkpoint_name))