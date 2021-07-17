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

from transformers import BertLMHeadModel, BartTokenizer, BartForConditionalGeneration, BartConfig, BartForSequenceClassification, BertTokenizer, BertConfig, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification

from data import ZuCo_dataset
from model_generation import BrainTranslator, BrainTranslatorNaive
 

def train_model(dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=25, checkpoint_path_best = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints/best/test.pt', checkpoint_path_last = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints/last/test.pt'):
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
            for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG in tqdm(dataloaders[phase]):
                
                # load in batch
                input_embeddings_batch = input_embeddings.to(device).float()
                input_masks_batch = input_masks.to(device)
                input_mask_invert_batch = input_mask_invert.to(device)
                target_ids_batch = target_ids.to(device)
                """replace padding ids in target_ids with -100"""
                target_ids_batch[target_ids_batch == tokenizer.pad_token_id] = -100 
              
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                seq2seqLMoutput = model(input_embeddings_batch, input_masks_batch, input_mask_invert_batch, target_ids_batch)

                """calculate loss"""
                # logits = seq2seqLMoutput.logits # 8*48*50265
                # logits = logits.permute(0,2,1) # 8*50265*48

                # loss = criterion(logits, target_ids_batch_label) # calculate cross entropy loss only on encoded target parts
                # NOTE: my criterion not used
                loss = seq2seqLMoutput.loss # use the BART language modeling loss

                # """check prediction, instance 0 of each batch"""
                # print('target size:', target_ids_batch.size(), ',original logits size:', logits.size(), ',target_mask size', target_mask_batch.size())
                # logits = logits.permute(0,2,1)
                # for idx in [0]:
                #     print(f'-- instance {idx} --')
                #     # print('permuted logits size:', logits.size())
                #     probs = logits[idx].softmax(dim = 1)
                #     # print('probs size:', probs.size())
                #     values, predictions = probs.topk(1)
                #     # print('predictions before squeeze:',predictions.size())
                #     predictions = torch.squeeze(predictions)
                #     # print('predictions:',predictions)
                #     # print('target mask:', target_mask_batch[idx])
                #     # print('[DEBUG]target tokens:',tokenizer.decode(target_ids_batch_copy[idx]))
                #     print('[DEBUG]predicted tokens:',tokenizer.decode(predictions))
                
                # backward + optimize only if in training phase
                if phase == 'train':
                    # with torch.autograd.detect_anomaly():
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.item() * input_embeddings_batch.size()[0] # batch loss
                # print('[DEBUG]loss:',loss.item())
                # print('#################################')
                

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
                print(f'update best on dev checkpoint: {checkpoint_path_best}')
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))
    torch.save(model.state_dict(), checkpoint_path_last)
    print(f'update last checkpoint: {checkpoint_path_last}')
    
    # write to log
    with open(output_log_file_name, 'w') as outlog:
        outlog.write(f'best val loss: {best_loss}\n')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def show_require_grad_layers(model):
    print()
    print(' require_grad layers:')
    # sanity check
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(' ', name)

if __name__ == '__main__':
    
    ''' config param'''
    num_epochs_step1 = 20
    num_epochs_step2 = 30
    step1_lr = 5*1e-5
    step2_lr = 5*1e-7
    dataset_setting = 'unique_sent'

    batch_size = 32
    # print('![Debug] using train batch size 1')
    save_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_generation' 
    
    # model_name = 'NaiveBartGeneration' # with no additional transformers
    model_name = 'BartGeneration' 
    # model_name = 'BertGeneration'
    
    # task_name = 'task1'
    # task_name = 'task1_task2'
    # task_name = 'task1_task2_task3'
    task_name = 'task1_task2_taskNRv2'


    # skip_step_one = False
    skip_step_one = True
    
    load_step1_checkpoint = True

    print(f'[INFO]using pretrained {model_name}')
    
    if skip_step_one:
        save_name = f'{task_name}_finetune_{model_name}_skipstep1_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}_setting_7-13_no_PE_add_srcmask'
    else:
        save_name = f'{task_name}_finetune_{model_name}_2steptraining_b{batch_size}_{num_epochs_step1}_{num_epochs_step2}_{step1_lr}_{step2_lr}_{dataset_setting}_setting_7-13_no_PE_add_srcmask'
    output_checkpoint_name_best = save_path + f'/best/{save_name}.pt' 
    output_checkpoint_name_last = save_path + f'/last/{save_name}.pt' 
    output_log_file_name = f'/shared/nas/data/m1/wangz3/SAO_project/SAO/log/generation/{save_name}.txt'

    subject_choice = 'ALL'
    print(f'![Debug]using {subject_choice}')
    eeg_type_choice = 'GD'
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
        dev = "cuda:2" 
    else:  
        dev = "cpu"
    # CUDA_VISIBLE_DEVICES=0,1,2,3  
    device = torch.device(dev)
    print(f'[INFO]using device {dev}')
    print()


    ''' set up dataloader '''
    whole_dataset_dicts = []
    if 'task1' in task_name:
        dataset_path_task1 = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset-with-tokens_6-25.pickle' 
        with open(dataset_path_task1, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task2' in task_name:
        dataset_path_task2 = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task2-NR/pickle/task2-NR-dataset-with-tokens_7-10.pickle' 
        with open(dataset_path_task2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'task3' in task_name:
        dataset_path_task3 = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task3-TSR/pickle/task3-TSR-dataset-with-tokens_7-10.pickle' 
        with open(dataset_path_task3, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))
    if 'taskNRv2' in task_name:
        dataset_path_taskNRv2 = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task2-NR-2.0/pickle/task2-NR-2.0-dataset-with-tokens_7-15.pickle' 
        with open(dataset_path_taskNRv2, 'rb') as handle:
            whole_dataset_dicts.append(pickle.load(handle))

    print()
    
    if model_name in ['BartGeneration','NaiveBartGeneration']:
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    elif model_name == 'BertGeneration':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        config = BertConfig.from_pretrained("bert-base-cased")
        config.is_decoder = True

    # train dataset
    train_set = ZuCo_dataset(whole_dataset_dicts, 'train', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
    # dev dataset
    dev_set = ZuCo_dataset(whole_dataset_dicts, 'dev', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)
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
    if model_name == 'BartGeneration':
        pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        model = BrainTranslator(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)
    elif model_name == 'BertGeneration':
        pretrained = BertLMHeadModel.from_pretrained('bert-base-cased', config=config)
        model = BrainTranslator(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 768, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)
    elif model_name == 'NaiveBartGeneration':
        pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        model = BrainTranslatorNaive(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)

    model.to(device)
    
    ''' training loop '''

    ######################################################
    '''step one trainig: freeze most of BART params'''
    ######################################################

    # closely follow BART paper
    if model_name in ['BartGeneration','NaiveBartGeneration']:
        for name, param in model.named_parameters():
            if param.requires_grad and 'pretrained' in name:
                if ('shared' in name) or ('embed_positions' in name) or ('encoder.layers.0' in name):
                    continue
                else:
                    param.requires_grad = False
    elif model_name == 'BertGeneration':
        for name, param in model.named_parameters():
            if param.requires_grad and 'pretrained' in name:
                if ('embeddings' in name) or ('encoder.layer.0' in name):
                    continue
                else:
                    param.requires_grad = False
 

    if skip_step_one:
        if load_step1_checkpoint:
            # stepone_checkpoint = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_generation/best/task1_task2_finetune_BartGeneration_2steptraining_b32_30_30_5e-05_5e-07_unique_sent_setting_7-10_no_PE_add_srcmask.pt'
            # stepone_checkpoint = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_generation/best/task1_task2_task3_finetune_BartGeneration_2steptraining_b32_30_40_5e-05_1e-06_unique_sent_setting_7-10_no_PE_add_srcmask.pt'
            # stepone_checkpoint = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_generation/best/cutstep2_task1_task2_task3_finetune_BartGeneration_2steptraining_b32_30_40_5e-05_1e-06_unique_sent_setting_7-10_no_PE_add_srcmask.pt'
            
            stepone_checkpoint = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_generation/best/task1_task2_taskNRv2_finetune_BartGeneration_2steptraining_b32_20_30_5e-05_5e-07_unique_sent_setting_7-13_no_PE_add_srcmask.pt'
            print(f'skip step one, load checkpoint: {stepone_checkpoint}')
            model.load_state_dict(torch.load(stepone_checkpoint))
        else:
            print('skip step one, start from scratch at step two')
    else:

        ''' set up optimizer and scheduler'''
        optimizer_step1 = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=step1_lr, momentum=0.9)

        exp_lr_scheduler_step1 = lr_scheduler.StepLR(optimizer_step1, step_size=20, gamma=0.1)

        ''' set up loss function '''
        criterion = nn.CrossEntropyLoss()

        print('=== start Step1 training ... ===')
        # print training layers
        show_require_grad_layers(model)
        # return best loss model from step1 training
        model = train_model(dataloaders, device, model, criterion, optimizer_step1, exp_lr_scheduler_step1, num_epochs=num_epochs_step1, checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last)

    ######################################################
    '''step two trainig: update whole model for a few iterations'''
    ######################################################
    for name, param in model.named_parameters():
        param.requires_grad = True

    ''' set up optimizer and scheduler'''
    optimizer_step2 = optim.SGD(model.parameters(), lr=step2_lr, momentum=0.9)

    exp_lr_scheduler_step2 = lr_scheduler.StepLR(optimizer_step2, step_size=30, gamma=0.1)

    ''' set up loss function '''
    criterion = nn.CrossEntropyLoss()
    
    print()
    print('=== start Step2 training ... ===')
    # print training layers
    show_require_grad_layers(model)
    
    '''main loop'''
    trained_model = train_model(dataloaders, device, model, criterion, optimizer_step2, exp_lr_scheduler_step2, num_epochs=num_epochs_step2, checkpoint_path_best = output_checkpoint_name_best, checkpoint_path_last = output_checkpoint_name_last)

    # '''save checkpoint'''
    # torch.save(trained_model.state_dict(), os.path.join(save_path,output_checkpoint_name))