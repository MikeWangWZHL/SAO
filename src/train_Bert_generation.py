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

from transformers import BertTokenizer, BertLMHeadModel, BertConfig
from data import ZuCo_dataset
from model import BrainTranslatorBert

#TODO: not working, cannot treat the word level EEG data as a kind of word embedding?


def train_model(dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=25, checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints/test.pt'):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    since = time.time()
    
    # for debuging #TODO: delete later
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    
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
            for input_embeddings, input_masks, input_mask_invert, target_ids, target_mask in tqdm(dataloaders[phase]):
                # load in batch
                input_embeddings_batch = input_embeddings.to(device).float()
                input_masks_batch = input_masks.to(device)
                input_mask_invert_batch = input_mask_invert.to(device)
                target_ids_batch = target_ids.to(device)
                # for debug:
                target_ids_batch_copy = torch.clone(target_ids_batch).detach() 
                """replace padding ids in target_ids with -100"""
                target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100
                
                # target_mask_batch = target_mask.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                LMoutput = model(input_embeddings_batch, input_masks_batch, target_ids_batch)

                logits = LMoutput.logits # 8*48*50265
                # logits = logits.permute(0,2,1) # 8*50265*48

                # print('target ids[1]:',target_ids_batch[1])
                '''loss'''
                # loss = criterion(logits, target_ids_batch) # calculate cross entropy loss only on encoded target parts
                loss = LMoutput.loss

                """check prediction, instance 0 of each batch"""
                print('target size:', target_ids_batch.size(), ',original logits size:', logits.size())
                # logits = logits.permute(0,2,1)
                for idx in [0,1,2,3,4,5]:
                    print(f'-- instance {idx} --')
                    # print('permuted logits size:', logits.size())
                    probs = logits[idx].softmax(dim = 1)
                    # print('probs size:', probs.size())
                    values, predictions = probs.topk(1)
                    # print('predictions before squeeze:',predictions.size())
                    predictions = torch.squeeze(predictions)
                    # print('predictions:',predictions)
                    # print('target mask:', target_mask_batch[idx])
                    print('[DEBUG]target tokens:',tokenizer.decode(target_ids_batch_copy[idx]))
                    print('[DEBUG]target ids:',target_ids_batch[idx])
                    print('[DEBUG]predicted tokens:',tokenizer.decode(predictions))

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * input_embeddings_batch.size()[0] # batch loss
                print('[DEBUG]loss:',loss.item())
                print('#################################')
                

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
    batch_size = 32
    # print('![Debug] using train batch size 1')
    save_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints' 
    output_checkpoint_name = save_path + '/finetune_bertLMhead_EEG_feature_as_embedding_direct.pt'
    
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
    config = BertConfig.from_pretrained("bert-base-cased")
    config.is_decoder = True
    
    ''' set up dataloader '''
    dataset_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset-with-tokens_6-25.pickle' 
    with open(dataset_path, 'rb') as handle:
        whole_dataset_dict = pickle.load(handle)
    # train dataset
    train_set = ZuCo_dataset(whole_dataset_dict, 'train', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice)
    # dev dataset
    dev_set = ZuCo_dataset(whole_dataset_dict, 'dev', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice)
    # test dataset
    test_set = ZuCo_dataset(whole_dataset_dict, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice)

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
    pretrained_bert = BertLMHeadModel.from_pretrained('bert-base-cased', config=config)
    model = BrainTranslatorBert(pretrained_bert, in_feature = 105*len(bands_choice), hidden_size = 768)
    model.to(device)
    
    ''' training loop '''
    ''' set up optimizer and scheduler'''
    optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    ''' set up loss function '''
    criterion = None

    print('=== start training ... ===')
    # return best loss model from step1 training
    model = train_model(dataloaders, device, model, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epoch, checkpoint_path = output_checkpoint_name)
