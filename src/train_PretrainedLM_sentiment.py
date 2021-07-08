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

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, BartForSequenceClassification, BertTokenizer, BertConfig, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
from data import ZuCo_dataset
from model_sentiment import FineTunePretrainedTwoStep

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

def train_model(dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=25, checkpoint_path_best = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints/best/test.pt', checkpoint_path_last = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints/last/test.pt', best_loss = 100000000000, best_acc = 0.0):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    since = time.time()
      
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'dev']:
            total_accuracy = 0.0
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for input_word_eeg_features, seq_lens, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG in tqdm(dataloaders[phase]):
                
                input_word_eeg_features = input_word_eeg_features.to(device).float()
                input_masks = input_masks.to(device)
                input_mask_invert = input_mask_invert.to(device)
                sentiment_labels = sentiment_labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                if isinstance(model, FineTunePretrainedTwoStep):
                    output = model(input_word_eeg_features, input_masks, input_mask_invert, sentiment_labels)
                    logits = output.logits
                    loss = output.loss


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
            if phase == 'dev' and (epoch_acc > best_acc):
                best_loss = epoch_loss
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                '''save checkpoint'''
                torch.save(model.state_dict(), checkpoint_path_best)
                print(f'update best on dev checkpoint: {checkpoint_path_best}')
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    print('Best val acc: {:4f}'.format(best_acc))
    torch.save(model.state_dict(), checkpoint_path_last)
    print(f'update last checkpoint: {checkpoint_path_last}')
    
    # write to log
    with open(output_log_file_name, 'w') as outlog:
        outlog.write(f'best val loss: {best_loss}\n')
        outlog.write('Best val acc: {:4f}'.format(best_acc))
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_acc, best_loss


if __name__ == '__main__':
    
    ''' config param'''
    num_epochs_step1 = 20
    num_epochs_step2 = 15
    step1_lr = 1e-3
    step2_lr = 1e-3

    # dataset_setting = 'unique_subj'
    dataset_setting = 'unique_sent'
    subject_choice = 'ALL'
    # subject_choice = 'ZAB'
    print(f'![Debug]using {subject_choice}')
    eeg_type_choice = 'GD'
    print(f'[INFO]eeg type {eeg_type_choice}')
    # bands_choice = ['_t1'] 
    bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] 
    print(f'[INFO]using bands {bands_choice}')

    batch_size = 32

    # model_name = 'finetune_Bert'
    model_name = 'finetune_RoBerta'
    
    save_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_sentiment' 
    save_name = f'Sentitment_{model_name}_subj-{subject_choice}_2steptraining_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}_{eeg_type_choice}_7-8_no_PE_add_srcmask'
    output_checkpoint_name_best = save_path + f'/best/{save_name}.pt' 
    output_checkpoint_name_last = save_path + f'/last/{save_name}.pt' 
    output_log_file_name = f'/shared/nas/data/m1/wangz3/SAO_project/SAO/log/sentiment/{save_name}.txt'


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


    ''' load pickle '''
    # dataset_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset_skip_zerofixation.pickle' 
    dataset_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset-with-tokens_6-25.pickle' 
    with open(dataset_path, 'rb') as handle:
        whole_dataset_dict = pickle.load(handle)
    
    '''tokenizer'''
    if model_name == 'finetune_Bert':
        print('[INFO]pretrained checkpoint: bert-base-cased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    
    elif model_name == 'finetune_RoBerta':
        print('[INFO]pretrained checkpoint: roberta-base')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


    ''' set up dataloader '''
    # train dataset
    train_set = ZuCo_dataset(whole_dataset_dict, 'train', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
    # dev dataset
    dev_set = ZuCo_dataset(whole_dataset_dict, 'dev', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
    # test dataset
    # test_set = ZuCo_dataset(whole_dataset_dict, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice)

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
    # pretrained_bart = BartForSequenceClassification.from_pretrained('facebook/bart-base', num_labels = 3)
    if model_name == 'finetune_Bert':
        pretrained = BertForSequenceClassification.from_pretrained('bert-base-cased',num_labels=3)
    elif model_name == 'finetune_RoBerta':
        pretrained = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)

    model = FineTunePretrainedTwoStep(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 768, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)
    model.to(device)
    
    ''' training loop '''
    
    ######################################################
    '''step one trainig: freeze most of BART params'''
    ######################################################
    # closely follow BART paper
    for name, param in model.named_parameters():
        if param.requires_grad and 'pretrained_BART' in name:
            if ('shared' in name) or ('embed_positions' in name) or ('encoder.layers.0' in name) or ('classification_head' in name):
                continue
            else:
                param.requires_grad = False


    ''' set up optimizer and scheduler'''
    optimizer_step1 = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=step1_lr, momentum=0.9)

    exp_lr_scheduler_step1 = lr_scheduler.StepLR(optimizer_step1, step_size=10, gamma=0.1)

    # TODO: rethink about the loss function
    ''' set up loss function '''
    criterion = nn.CrossEntropyLoss()

    print('=== start Step1 training ... ===')
    # return best loss model from step1 training
    model, best_acc_step1, best_loss_step1 = train_model(dataloaders, device, model, criterion, optimizer_step1, exp_lr_scheduler_step1, num_epochs=num_epochs_step1, checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last)
    
    ######################################################
    '''step two trainig: update whole model for a few iterations'''
    ######################################################
    for name, param in model.named_parameters():
        param.requires_grad = True

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)

    ''' set up optimizer and scheduler'''
    optimizer_step2 = optim.SGD(model.parameters(), lr=step2_lr, momentum=0.9)

    exp_lr_scheduler_step2 = lr_scheduler.StepLR(optimizer_step2, step_size=10, gamma=0.1)

    ''' set up loss function '''
    criterion = nn.CrossEntropyLoss()
    
    print()
    print('=== start Step2 training ... ===')
    trained_model = train_model(dataloaders, device, model, criterion, optimizer_step2, exp_lr_scheduler_step2, num_epochs=num_epochs_step2, checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last, best_acc = best_acc_step1, best_loss = best_loss_step1)

    # '''save checkpoint'''
    # torch.save(trained_model.state_dict(), os.path.join(save_path,output_checkpoint_name))