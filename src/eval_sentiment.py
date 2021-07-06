import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils.rnn import pack_padded_sequence 
import pickle
import json
import matplotlib.pyplot as plt
from glob import glob
import time
import copy
from tqdm import tqdm

from transformers import BertTokenizer, BertLMHeadModel, BertConfig
from data import ZuCo_dataset
from model_sentiment import BaselineMLPSentence, BaselineLSTM

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    # preds: numpy array: N * 3 
    # labels: numpy array: N 
    pred_flat = np.argmax(preds, axis=1).flatten()  
    
    labels_flat = labels.flatten()
    
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def flat_accuracy_top_k(preds, labels,k):
    topk_preds = []
    for pred in preds:
        topk = pred.argsort()[-k:][::-1]
        topk_preds.append(list(topk))
    # print(topk_preds)
    topk_preds = list(topk_preds)
    right_count = 0
    # print(len(labels))
    for i in range(len(labels)):
        l = labels[i][0]
        if l in topk_preds[i]:
            right_count+=1
    return right_count/len(labels)

def eval_model(dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=25):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    since = time.time()
      
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000000
    best_acc = 0.0
    

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['test']:
            total_accuracy = 0.0
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for input_word_eeg_features, seq_lens, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG in tqdm(dataloaders[phase]):
                
                input_word_eeg_features = input_word_eeg_features.to(device).float()
                sent_level_EEG = sent_level_EEG.to(device)
                sentiment_labels = sentiment_labels.to(device)


                if isinstance(model, BaselineMLPSentence):
                    # forward
                    logits = model(sent_level_EEG) # before softmax

                elif isinstance(model, BaselineLSTM):
                    # forward LSTM baseline
                    # print(seq_lens)
                    # print(input_word_eeg_features.size())
                    x_packed = pack_padded_sequence(input_word_eeg_features, seq_lens, batch_first=True, enforce_sorted=False)
                    logits = model(x_packed)
                    # print(logits.size())
                    # print('successful forward')
                    # quit()

                # calculate loss
                loss = criterion(logits, sentiment_labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    # with torch.autograd.detect_anomaly():
                    loss.backward()
                    optimizer.step()

                # calculate accuracy
                preds_cpu = logits.detach().cpu().numpy()
                label_cpu = sentiment_labels.cpu().numpy()

                total_accuracy += flat_accuracy(preds_cpu, label_cpu)

                # statistics
                running_loss += loss.item() * sent_level_EEG.size()[0] # batch loss
                # print('[DEBUG]loss:',loss.item())
                # print('#################################')
                

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = total_accuracy / len(dataloaders[phase])
            print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            print('{} Acc: {:.4f}'.format(phase, epoch_acc))

            # deep copy the model
            if phase == 'test' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_acc = epoch_acc
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best test loss: {:4f}'.format(best_loss))
    print('Best test acc: {:4f}'.format(best_acc))


if __name__ == '__main__':
    
    ''' config param'''
    num_epochs = 1

    dataset_setting = 'unique_subj'
    checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_sentiment/best/Baseline_LSTM_h256_0.01_b32_unique_subj_7-6.pt'
    # dataset_setting = 'unique_sent'
    model_name = 'LSTM_h256'
    batch_size = 1


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


    ''' load pickle'''
    # dataset_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset_skip_zerofixation.pickle' 
    dataset_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset-with-tokens_6-25.pickle' 
    with open(dataset_path, 'rb') as handle:
        whole_dataset_dict = pickle.load(handle)
    
    '''set up tokenizer'''
    # tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    ''' set up dataloader '''
    # train dataset
    # train_set = ZuCo_dataset(whole_dataset_dict, 'train', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
    # dev dataset
    # dev_set = ZuCo_dataset(whole_dataset_dict, 'dev', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
  
    # test dataset
    test_set = ZuCo_dataset(whole_dataset_dict, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice)

    dataset_sizes = {'test': len(test_set)}
    # print('[INFO]train_set size: ', len(train_set))
    print('[INFO]test_set size: ', len(test_set))
    
    test_dataloader = DataLoader(test_set, batch_size = 1, shuffle=False, num_workers=4)
    # dataloaders
    dataloaders = {'test':test_dataloader}

    ''' set up model '''
    # print('[INFO]Model: BaselineMLP')
    # model = BaselineMLPSentence(input_dim = 840, hidden_dim = 128, output_dim = 3)
    print('[INFO]Model: BaselineLSTM')
    model = BaselineLSTM(input_dim = 840, hidden_dim = 256, output_dim = 3)
    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    
    ''' set up optimizer and scheduler'''
    optimizer_step1 = None
    exp_lr_scheduler_step1 = None

    ''' set up loss function '''
    criterion = nn.CrossEntropyLoss()

    print('=== start training ... ===')
    # return best loss model from step1 training
    model = eval_model(dataloaders, device, model, criterion, optimizer_step1, exp_lr_scheduler_step1, num_epochs=num_epochs)
