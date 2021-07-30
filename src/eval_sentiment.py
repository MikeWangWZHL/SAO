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

from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, BartForSequenceClassification, BertTokenizer, BertConfig, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
from data import ZuCo_dataset
from model_sentiment import BaselineMLPSentence, BaselineLSTM, FineTunePretrainedTwoStep, ZeroShotSentimentDiscovery, JointBrainTranslatorSentimentClassifier
from model_generation import BrainTranslator, BrainTranslatorNaive
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score

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

def eval_model(dataloaders, device, model, criterion, optimizer, scheduler, num_epochs=25, tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')):

    def logits2PredString(logits, tokenizer):
        probs = logits[0].softmax(dim = 1)
        # print('probs size:', probs.size())
        values, predictions = probs.topk(1)
        # print('predictions before squeeze:',predictions.size())
        predictions = torch.squeeze(predictions)
        predict_string = tokenizer.decode(predictions)
        return predict_string
    
    # modified from: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html
    since = time.time()
      
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 100000000000
    best_acc = 0.0
    
    total_pred_labels = np.array([])
    total_true_labels = np.array([])

    for epoch in range(1):
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
            for input_word_eeg_features, seq_lens, input_masks, input_mask_invert, target_ids, target_mask, sentiment_labels, sent_level_EEG in dataloaders[phase]:
                
                input_word_eeg_features = input_word_eeg_features.to(device).float()
                input_masks = input_masks.to(device)
                input_mask_invert = input_mask_invert.to(device)
 
                sent_level_EEG = sent_level_EEG.to(device)
                sentiment_labels = sentiment_labels.to(device)

                target_ids = target_ids.to(device)
                target_mask = target_mask.to(device)

                ## forward ###################
                if isinstance(model, BaselineMLPSentence):
                    logits = model(sent_level_EEG) # before softmax
                    # calculate loss
                    loss = criterion(logits, sentiment_labels)

                elif isinstance(model, BaselineLSTM):
                    x_packed = pack_padded_sequence(input_word_eeg_features, seq_lens, batch_first=True, enforce_sorted=False)
                    logits = model(x_packed)
                    # calculate loss
                    loss = criterion(logits, sentiment_labels)

                elif isinstance(model, BertForSequenceClassification) or isinstance(model, RobertaForSequenceClassification) or isinstance(model, BartForSequenceClassification):
                    output = model(input_ids = target_ids, attention_mask = target_mask, return_dict = True, labels = sentiment_labels)
                    logits = output.logits
                    loss = output.loss
                
                elif isinstance(model, FineTunePretrainedTwoStep):
                    output = model(input_word_eeg_features, input_masks, input_mask_invert, sentiment_labels)
                    logits = output.logits
                    loss = output.loss

                elif isinstance(model, ZeroShotSentimentDiscovery):    
                    print()
                    print('target string:',tokenizer.decode(target_ids[0]).replace('<pad>','').split('</s>')[0]) 

                    """replace padding ids in target_ids with -100"""
                    target_ids[target_ids == tokenizer.pad_token_id] = -100 

                    output = model(input_word_eeg_features, input_masks, input_mask_invert, target_ids, sentiment_labels)
                    logits = output.logits
                    loss = output.loss
                
                elif isinstance(model, JointBrainTranslatorSentimentClassifier):

                    print()
                    print('target string:',tokenizer.decode(target_ids[0]).replace('<pad>','').split('</s>')[0]) 

                    """replace padding ids in target_ids with -100"""
                    target_ids[target_ids == tokenizer.pad_token_id] = -100 

                    LM_output, classification_output = model(input_word_eeg_features, input_masks, input_mask_invert, target_ids, sentiment_labels)
                    LM_logits = LM_output.logits
                    print('pred string:', logits2PredString(LM_logits, tokenizer).split('</s></s>')[0].replace('<s>',''))
                    classification_loss = classification_output['loss']
                    logits = classification_output['logits']
                    loss = classification_loss 
                ###############################

                # backward + optimize only if in training phase
                if phase == 'train':
                    # with torch.autograd.detect_anomaly():
                    loss.backward()
                    optimizer.step()

                # calculate accuracy
                preds_cpu = logits.detach().cpu().numpy()
                label_cpu = sentiment_labels.cpu().numpy()

                total_accuracy += flat_accuracy(preds_cpu, label_cpu)
                
                # add to total pred and label array, for cal F1, precision, recall
                pred_flat = np.argmax(preds_cpu, axis=1).flatten()
                labels_flat = label_cpu.flatten()

                total_pred_labels = np.concatenate((total_pred_labels,pred_flat))
                total_true_labels = np.concatenate((total_true_labels,labels_flat))
                

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
    print()
    print('test sample num:', len(total_pred_labels))
    print('total preds:',total_pred_labels)
    print('total truth:',total_true_labels)
    print('sklearn macro: precision, recall, F1:')
    print(precision_recall_fscore_support(total_true_labels, total_pred_labels, average='macro'))
    print()
    print('sklearn micro: precision, recall, F1:')
    print(precision_recall_fscore_support(total_true_labels, total_pred_labels, average='micro'))
    print()
    print('sklearn accuracy:')
    print(accuracy_score(total_true_labels,total_pred_labels))
    print()



if __name__ == '__main__':
    
    ''' config param'''
    num_epochs = 1

    dataset_setting = 'unique_sent'
    
    '''model name'''
    # model_name = 'BaselineMLP'
    # model_name = 'BaselineLSTM'
    # model_name = 'NaiveFineTunePretrainedBert'
    # model_name = 'FinetunedBertOnText'
    # model_name = 'FinetunedBertOnText_ext'
    # model_name = 'FinetunedRoBertaOnText'
    # model_name = 'FinetunedRoBertaOnText_ext'
    # model_name = 'FinetunedBartOnText'
    # model_name = 'FinetunedBartOnText_ext'
    # model_name = 'FineTunePretrainedBertTwoStep'
    # model_name = 'FineTunePretrainedRoBertaTwoStep'
    # model_name = 'JointBrainTranslatorSentimentClassifier'

    model_name = 'ZeroShotSentimentDiscovery'
    
    if model_name == 'ZeroShotSentimentDiscovery':
        '''choose generator'''
        # generator_name = 'BrainTranslator'
        # generator_name = 'BrainTranslatorNaive'
        generator_name = 'BrainTranslator_skipstep1'

        '''choose classifier'''
        # classifier_name = 'Bert'
        # classifier_name = 'Bert_ext'
        # classifier_name = 'Bart'
        classifier_name = 'Bart_ext'
        # classifier_name = 'RoBerta'
        # classifier_name = 'RoBerta_ext'
    
    print(f'[INFO] eval {model_name}')

    '''checkpoint'''
    if model_name == 'BaselineMLP':
        checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_sentiment/last/BaselineMLP_0.001_b32_unique_sent_GD_7-8.pt'  
        print('loading checkpoint:', checkpoint_path)
    
    elif model_name == 'BaselineLSTM':
        # layer num = 1
        # checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_sentiment/best/BaselineLSTM_0.001_b32_unique_sent_GD_7-8.pt'
        # layer num = 4
        checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_sentiment/best/BaselineLSTM_numLayers-4_0.01_b32_unique_sent_GD_7-8.pt'
        print('loading checkpoint:', checkpoint_path)
    
    elif model_name == 'FinetunedBertOnText':
        checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_pretrained/best/Sentitment_pretrain_Bert_b32_20_0.001_unique_sent_GD_7-8.pt'
        print('loading checkpoint:', checkpoint_path)
 
    elif model_name == 'FinetunedBertOnText_ext':
        checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_pretrained/best/On_StanfordSentitmentTreeband_pretrain_Bert_b32_20_0.001_7-19.pt'
        print('loading checkpoint:', checkpoint_path)

    elif model_name == 'FinetunedBartOnText':
        checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_pretrained/best/Sentitment_pretrain_Bart_b32_20_0.001_unique_sent_GD_7-8.pt'
        print('loading checkpoint:', checkpoint_path)

    elif model_name == 'FinetunedBartOnText_ext':
        checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_pretrained/best/On_StanfordSentitmentTreeband_pretrain_Bart_b32_20_0.0001_7-19.pt'
        print('loading checkpoint:', checkpoint_path)

    elif model_name == 'FinetunedRoBertaOnText':
        checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_pretrained/best/Sentitment_pretrain_RoBerta_b32_20_0.001_unique_sent_GD_7-8.pt'
        print('loading checkpoint:', checkpoint_path)

    elif model_name == 'FinetunedRoBertaOnText_ext':
        checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_pretrained/best/On_StanfordSentitmentTreeband_pretrain_RoBerta_b32_20_0.001_7-19.pt'
        print('loading checkpoint:', checkpoint_path)
    
    elif model_name == 'FineTunePretrainedBertTwoStep':
        # checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_sentiment/best/Sentitment_finetune_Bert_subj-ALL_2steptraining_b32_1_10_5e-05_5e-07_unique_sent_GD_7-9_no_PE_add_srcmask.pt'
        '''use checkpoint trained on text as start point'''
        # checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_sentiment/best/UsePretrainedTextCheckpoint_Sentitment_finetune_Bert_subj-ALL_2steptraining_b32_10_10_1e-05_1e-07_unique_sent_GD_7-8_no_PE_add_srcmask.pt'
        '''use random init '''
        checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_sentiment/best/RandomInit_Sentitment_finetune_Bert_subj-ALL_2steptraining_b32_1_10_5e-05_5e-07_unique_sent_GD_7-9_no_PE_add_srcmask.pt'
        print('loading checkpoint:', checkpoint_path)
    elif model_name == 'FineTunePretrainedRoBertaTwoStep':
        checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_sentiment/best/Sentitment_finetune_RoBerta_subj-ALL_2steptraining_b32_20_15_0.001_0.001_unique_sent_GD_7-8_no_PE_add_srcmask.pt'
        print('loading checkpoint:', checkpoint_path)
    elif model_name == 'ZeroShotSentimentDiscovery':
        if generator_name == 'BrainTranslator':
            # trained on task 1:
            brain2text_translator_checkpoint = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_generation/best/task1_finetune_BartGeneration_2steptraining_b32_30_30_5e-05_5e-07_unique_sent_setting_7-8_no_PE_add_srcmask.pt'
            
            # trained on task 2 and task 1:
            # brain2text_translator_checkpoint = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_generation/best/task1_task2_finetune_BartGeneration_2steptraining_b32_30_30_5e-05_5e-07_unique_sent_setting_7-10_no_PE_add_srcmask.pt'
            
            # trained on task 1,2,3:
            # brain2text_translator_checkpoint = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_generation/best/task1_task2_task3_finetune_BartGeneration_2steptraining_b32_30_30_5e-05_5e-07_unique_sent_setting_7-10_no_PE_add_srcmask.pt'
        elif generator_name == 'BrainTranslatorNaive':
            brain2text_translator_checkpoint = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_generation/best/task1_finetune_NaiveBartGeneration_2steptraining_b32_20_20_5e-05_5e-07_unique_sent_setting_7-12_no_PE_add_srcmask.pt'
        elif generator_name == 'BrainTranslator_skipstep1':
            # 30 epoch: current best
            # brain2text_translator_checkpoint = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_generation/best/task1_finetune_BartGeneration_skipstep1_b32_20_30_5e-05_5e-07_unique_sent_setting_7-12_no_PE_add_srcmask.pt'

            # randomly init one:    
            brain2text_translator_checkpoint = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_generation/best/randinit_task1_finetune_BartGeneration_skipstep1_b32_20_30_5e-05_0.0005_unique_sent_setting_7-21_no_PE_add_srcmask.pt'
        
        print('loading translator checkpoint:', brain2text_translator_checkpoint)

        if classifier_name == 'Bert':
            sentiment_classifier_checkpoint = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_pretrained/best/Sentitment_pretrain_Bert_b32_20_0.001_unique_sent_GD_7-8.pt'
        elif classifier_name == 'Bert_ext':
            sentiment_classifier_checkpoint = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_pretrained/best/On_StanfordSentitmentTreeband_pretrain_Bert_b32_20_0.001_7-19.pt'
        elif classifier_name == 'Bart':
            sentiment_classifier_checkpoint = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_pretrained/best/Sentitment_pretrain_Bart_b32_20_0.0001_unique_sent_GD_7-8.pt'
        elif classifier_name == 'Bart_ext':
            sentiment_classifier_checkpoint = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_pretrained/best/On_StanfordSentitmentTreeband_pretrain_Bart_b32_20_0.0001_7-19.pt'
        elif classifier_name == 'RoBerta':
            sentiment_classifier_checkpoint = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_pretrained/best/Sentitment_pretrain_RoBerta_b32_20_0.001_unique_sent_GD_7-8.pt'
        elif classifier_name == 'RoBerta_ext':
            sentiment_classifier_checkpoint = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_pretrained/best/On_StanfordSentitmentTreeband_pretrain_RoBerta_b32_20_0.001_7-19.pt'
        print('loading classifier checkpoint:', sentiment_classifier_checkpoint)
        print()
    elif model_name == 'JointBrainTranslatorSentimentClassifier':
        checkpoint_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/checkpoints_sentiment/best/Sentitment_JointBrainTranslatorSentimentClassifier_subj-ALL_2steptraining_b32_20_15_5e-05_5e-07_unique_sent_GD_7-9_no_PE_add_srcmask.pt'
        print('loading classifier checkpoint:',checkpoint_path) 
        print()

    batch_size = 1


    subject_choice = 'ALL'
    # subject_choice = 'ZAB'
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


    ''' load pickle'''
    # dataset_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset_skip_zerofixation.pickle' 
    dataset_path = '/shared/nas/data/m1/wangz3/SAO_project/SAO/dataset/ZuCo/task1-SR/pickle/task1-SR-dataset-with-tokens_6-25.pickle' 
    with open(dataset_path, 'rb') as handle:
        whole_dataset_dict = pickle.load(handle)
    
    '''set up tokenizer'''
    if model_name in ['FinetunedBertOnText','FinetunedBertOnText_ext','BaselineMLP','BaselineLSTM', 'FineTunePretrainedBertTwoStep']:
        print('[INFO]pretrained checkpoint: bert-base-cased')
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    elif model_name in ['FinetunedRoBertaOnText','FinetunedRoBertaOnText_ext','FineTunePretrainedRoBertaTwoStep']:
        print('[INFO]pretrained checkpoint: roberta-base')
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    elif model_name in ['ZeroShotSentimentDiscovery', 'JointBrainTranslatorSentimentClassifier', 'FinetunedBartOnText','FinetunedBartOnText_ext']:
        translation_tokenizer = BartTokenizer.from_pretrained('facebook/bart-large') # Bart
        tokenizer = translation_tokenizer
        if model_name == 'ZeroShotSentimentDiscovery':
            if classifier_name in ['Bert','Bert_ext']:
                sentiment_tokenizer = BertTokenizer.from_pretrained('bert-base-cased') # Bert
            elif classifier_name in ['Bart','Bart_ext']:
                sentiment_tokenizer = translation_tokenizer
            elif classifier_name in ['RoBerta','RoBerta_ext']:
                sentiment_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    ''' set up model '''
    if model_name == 'BaselineMLP':
        print('[INFO]Model: BaselineMLP')
        model = BaselineMLPSentence(input_dim = 840, hidden_dim = 128, output_dim = 3)
    elif model_name == 'BaselineLSTM':
        print('[INFO]Model: BaselineLSTM')
        # model = BaselineLSTM(input_dim = 840, hidden_dim = 256, output_dim = 3, num_layers = 1)
        model = BaselineLSTM(input_dim = 840, hidden_dim = 256, output_dim = 3, num_layers = 4)
    elif model_name in ['FinetunedBertOnText','FinetunedBertOnText_ext']:
        print('[INFO]Model: FinetunedBertOnText/FinetunedBertOnText_ext')
        model = BertForSequenceClassification.from_pretrained('bert-base-cased',num_labels=3)
    elif model_name in ['FinetunedRoBertaOnText','FinetunedRoBertaOnText_ext']:
        print('[INFO]Model: FinetunedRoBertaOnText/FinetunedRoBertaOnText_ext')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
    elif model_name in ['FinetunedBartOnText','FinetunedBartOnText_ext']:
        print('[INFO]Model: FinetunedBartOnText/FinetunedBartOnText_ext')
        model = BartForSequenceClassification.from_pretrained('facebook/bart-large', num_labels=3)
    elif model_name == 'FineTunePretrainedBertTwoStep':
        print('[INFO]Model: FineTunePretrainedBertTwoStep')
        pretrained = BertForSequenceClassification.from_pretrained('bert-base-cased',num_labels=3)
        model = FineTunePretrainedTwoStep(pretrained, in_feature = 105*len(bands_choice), d_model = 768, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)    
    elif model_name == 'FineTunePretrainedRoBertaTwoStep':
        print('[INFO]Model: FineTunePretrainedRoBertaTwoStep')
        pretrained = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
        model = FineTunePretrainedTwoStep(pretrained, in_feature = 105*len(bands_choice), d_model = 768, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)
    elif model_name == 'ZeroShotSentimentDiscovery':
        print(f'[INFO]Model: ZeroShotSentimentDiscovery, using classifer:{classifier_name}, using generator: {generator_name}')
        pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        if generator_name in ['BrainTranslator','BrainTranslator_skipstep1']:
            brain2text_translator = BrainTranslator(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)
        elif generator_name == 'BrainTranslatorNaive':
            brain2text_translator = BrainTranslatorNaive(pretrained, in_feature = 105*len(bands_choice), decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048)

        brain2text_translator.load_state_dict(torch.load(brain2text_translator_checkpoint))
        
        if classifier_name in ['Bert','Bert_ext']:
            sentiment_classifier = BertForSequenceClassification.from_pretrained('bert-base-cased',num_labels=3)
        elif classifier_name in ['Bart','Bart_ext']:
            sentiment_classifier = BartForSequenceClassification.from_pretrained('facebook/bart-large', num_labels=3)
        elif classifier_name in ['RoBerta','RoBerta_ext']:
            sentiment_classifier = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)

        sentiment_classifier.load_state_dict(torch.load(sentiment_classifier_checkpoint))

        model = ZeroShotSentimentDiscovery(brain2text_translator, sentiment_classifier, translation_tokenizer, sentiment_tokenizer, device = device)
        model.to(device)

    elif model_name == 'JointBrainTranslatorSentimentClassifier':
        print('[INFO]Model: JointBrainTranslatorSentimentClassifier')
        # not working, it is really hard to converge using only EEG data
        pretrained = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        model = JointBrainTranslatorSentimentClassifier(pretrained, in_feature = 105*len(bands_choice), d_model = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048, num_labels = 3)
    

    if model_name != 'ZeroShotSentimentDiscovery':
        # load model and send to device
        model.load_state_dict(torch.load(checkpoint_path))
        model.to(device)

    ''' set up dataloader '''
    # test dataset
    test_set = ZuCo_dataset(whole_dataset_dict, 'test', tokenizer, subject = subject_choice, eeg_type = eeg_type_choice, bands = bands_choice, setting = 'unique_sent')

    dataset_sizes = {'test': len(test_set)}
    # print('[INFO]train_set size: ', len(train_set))
    print('[INFO]test_set size: ', len(test_set))
    
    test_dataloader = DataLoader(test_set, batch_size = 1, shuffle=False, num_workers=4)
    # dataloaders
    dataloaders = {'test':test_dataloader}
    
    ''' set up optimizer and scheduler'''
    optimizer_step1 = None
    exp_lr_scheduler_step1 = None

    ''' set up loss function '''
    criterion = nn.CrossEntropyLoss()

    print('=== start training ... ===')
    # return best loss model from step1 training
    model = eval_model(dataloaders, device, model, criterion, optimizer_step1, exp_lr_scheduler_step1, num_epochs=num_epochs, tokenizer = tokenizer)
