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
from model_generation import BrainTranslator, BrainTranslatorNaive
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu
from rouge import Rouge


def eval_model(dataloaders, device, tokenizer, criterion, model, output_all_results_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/results/temp.txt' ):
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

    model.eval()   # Set model to evaluate mode
    running_loss = 0.0

    # Iterate over data.
    sample_count = 0
    
    target_tokens_list = []
    target_string_list = []
    pred_tokens_list = []
    pred_string_list = []
    with open(output_all_results_path,'w') as f:
        for input_embeddings, seq_len, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG in dataloaders['test']:
            # load in batch
            input_embeddings_batch = input_embeddings.to(device).float()
            input_masks_batch = input_masks.to(device)
            target_ids_batch = target_ids.to(device)
            input_mask_invert_batch = input_mask_invert.to(device)
            
            target_tokens = tokenizer.convert_ids_to_tokens(target_ids_batch[0].tolist(), skip_special_tokens = True)
            target_string = tokenizer.decode(target_ids_batch[0], skip_special_tokens = True)
            # print('target ids tensor:',target_ids_batch[0])
            # print('target ids:',target_ids_batch[0].tolist())
            # print('target tokens:',target_tokens)
            # print('target string:',target_string)
            f.write(f'target string: {target_string}\n')

            # add to list for later calculate bleu metric
            target_tokens_list.append([target_tokens])
            target_string_list.append(target_string)
            
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


            # get predicted tokens
            # print('target size:', target_ids_batch.size(), ',original logits size:', logits.size())
            logits = seq2seqLMoutput.logits # 8*48*50265
            # logits = logits.permute(0,2,1)
            # print('permuted logits size:', logits.size())
            probs = logits[0].softmax(dim = 1)
            # print('probs size:', probs.size())
            values, predictions = probs.topk(1)
            # print('predictions before squeeze:',predictions.size())
            predictions = torch.squeeze(predictions)
            predicted_string = tokenizer.decode(predictions).split('</s></s>')[0].replace('<s>','')
            # print('predicted string:',predicted_string)
            f.write(f'predicted string: {predicted_string}\n')
            f.write(f'################################################\n\n\n')

            # convert to int list
            predictions = predictions.tolist()
            truncated_prediction = []
            for t in predictions:
                if t != tokenizer.eos_token_id:
                    truncated_prediction.append(t)
                else:
                    break
            pred_tokens = tokenizer.convert_ids_to_tokens(truncated_prediction, skip_special_tokens = True)
            # print('predicted tokens:',pred_tokens)
            pred_tokens_list.append(pred_tokens)
            pred_string_list.append(predicted_string)
            # print('################################################')
            # print()

            sample_count += 1
            # statistics
            running_loss += loss.item() * input_embeddings_batch.size()[0] # batch loss
            # print('[DEBUG]loss:',loss.item())
            # print('#################################')


    epoch_loss = running_loss / dataset_sizes['test_set']
    print('test loss: {:4f}'.format(epoch_loss))

    """ calculate corpus bleu score """
    weights_list = [(1.0,),(0.5,0.5),(1./3.,1./3.,1./3.),(0.25,0.25,0.25,0.25)]
    for weight in weights_list:
        # print('weight:',weight)
        corpus_bleu_score = corpus_bleu(target_tokens_list, pred_tokens_list, weights = weight)
        print(f'corpus BLEU-{len(list(weight))} score:', corpus_bleu_score)

    print()
    """ calculate rouge score """
    rouge = Rouge()
    rouge_scores = rouge.get_scores(pred_string_list,target_string_list, avg = True)
    print(rouge_scores)


if __name__ == '__main__':    
    ''' config param'''
    batch_size = 1
    
    subject_choice = 'ALL'
    # subject_choice = 'ZAB'
    # print(f'![Debug]using {subject_choice}')
    # print(f'![Debug]using ZPH')
    eeg_type_choice = 'GD'
    print(f'[INFO]eeg type {eeg_type_choice}')
    # bands_choice = ['_t1'] 
    bands_choice = ['_t1','_t2','_a1','_a2','_b1','_b2','_g1','_g2'] 
    print(f'[INFO]using bands {bands_choice}')
    
    dataset_setting = 'unique_sent'

    task_name = 'task1'
    # task_name = 'task1_task2'
    # task_name = 'task1_task2_task3'
    # task_name = 'task1_task2_taskNRv2'
    
    # model_name = 'BrainTranslator'
    # model_name = 'BrainTranslatorNaive'
    model_name = 'BrainTranslator_skipstep1'

    output_all_results_path = f'/shared/nas/data/m1/wangz3/SAO_project/SAO/results/{task_name}-{model_name}_modified_config_all_generation_results-8_15.txt'
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
    
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')

    # test dataset
    test_set = ZuCo_dataset(whole_dataset_dicts, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = dataset_setting)

    dataset_sizes = {"test_set":len(test_set)}
    print('[INFO]test_set size: ', len(test_set))
    
    # dataloaders
    test_dataloader = DataLoader(test_set, batch_size = 1, shuffle=False, num_workers=4)

    dataloaders = {'test':test_dataloader}

    ''' set up model '''
    if task_name == 'task1':
        if model_name == 'BrainTranslator':
            checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_generation/best/task1_finetune_BartGeneration_2steptraining_b32_30_30_5e-05_5e-07_unique_sent_setting_7-8_no_PE_add_srcmask.pt'
        elif model_name == 'BrainTranslatorNaive':
            checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_generation/best/task1_finetune_NaiveBartGeneration_2steptraining_b32_20_20_5e-05_5e-07_unique_sent_setting_7-12_no_PE_add_srcmask.pt'
        elif model_name == 'BrainTranslator_skipstep1':
            checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_generation/best/task1_finetune_BartGeneration_skipstep1_b32_20_30_5e-05_5e-07_unique_sent_setting_7-12_no_PE_add_srcmask.pt'
    elif task_name == 'task1_task2':
        if model_name == 'BrainTranslator':
            checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_generation/best/task1_task2_finetune_BartGeneration_2steptraining_b32_30_30_5e-05_5e-07_unique_sent_setting_7-10_no_PE_add_srcmask.pt'
        elif model_name == 'BrainTranslator_skipstep1':
            checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_generation/best/task1_task2_finetune_BartGeneration_skipstep1_b32_20_30_5e-05_5e-07_unique_sent_setting_7-13_no_PE_add_srcmask.pt'
    elif task_name == 'task1_task2_task3':
        if model_name == 'BrainTranslator':
            checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_generation/best/task1_task2_task3_finetune_BartGeneration_2steptraining_b32_30_40_5e-05_1e-07_unique_sent_setting_7-10_no_PE_add_srcmask.pt'
        elif model_name == 'BrainTranslator_skipstep1':
            checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_generation/best/task1_task2_task3_finetune_BartGeneration_skipstep1_b32_20_30_5e-05_5e-07_unique_sent_setting_7-13_no_PE_add_srcmask.pt'
    elif task_name == 'task1_task2_taskNRv2':
        if model_name == 'BrainTranslator':
            checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_generation/best/task1_task2_taskNRv2_finetune_BartGeneration_2steptraining_b32_20_30_5e-05_5e-07_unique_sent_setting_7-16_no_PE_add_srcmask.pt'
        elif model_name == 'BrainTranslator_skipstep1':
            # pretrained:
            checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_generation/best/task1_task2_taskNRv2_finetune_BartGeneration_skipstep1_b32_20_40_5e-05_5e-07_unique_sent_setting_7-13_no_PE_add_srcmask.pt'
            
            # random init:
            # checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_generation/best/randinit_task1_task2_taskNRv2_finetune_BartGeneration_skipstep1_b32_20_50_5e-05_0.0005_unique_sent_setting_7-21_no_PE_add_srcmask.pt'
        elif model_name == 'BrainTranslatorNaive':
            checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_generation/best/task1_task2_taskNRv2_finetune_NaiveBartGeneration_skipstep1_b32_20_30_5e-05_5e-07_unique_sent_setting_7-21_no_PE_add_srcmask.pt'


    '''set up model'''
    num_beams = 5 # default = 1
    repetition_penalty = 2 # default = 1, no penalty
    print(f'[INFO]num_beams = {num_beams}\n[INFO]repetition_penalty = {repetition_penalty}')
    pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large',
        num_beams = 5,
        repetition_penalty = 2
    )
    
    if model_name in ['BrainTranslator','BrainTranslator_skipstep1']:
        model = BrainTranslator(pretrained_bart, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)
    elif model_name == 'BrainTranslatorNaive':
        model = BrainTranslatorNaive(pretrained_bart, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)

    model.load_state_dict(torch.load(checkpoint_path))
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    ''' eval '''
    eval_model(dataloaders, device, tokenizer, criterion, model, output_all_results_path = output_all_results_path)

    print(f'[INFO]loaded checkpoint: {checkpoint_path}')
    print(f'[INFO]the score is on task: {task_name}, model: {model_name}')