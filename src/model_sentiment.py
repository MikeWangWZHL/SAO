import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
import math
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# using sent level EEG, MLP baseline for sentiment
class BaselineMLPSentence(nn.Module):
    def __init__(self, input_dim = 840, hidden_dim = 128, output_dim = 3):
        super(BaselineMLPSentence, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dim, output_dim) # positive, negative, neutral  
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out


class BaselineLSTM(nn.Module):
    def __init__(self, input_dim = 840, hidden_dim = 256, output_dim = 3):
        super(BaselineLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers = 1, batch_first = True, bidirectional = True)

        self.hidden2sentiment = nn.Linear(hidden_dim*2, output_dim)

    def forward(self, x_packed):
        # input: (N,seq_len,input_dim)
        # print(x_packed.data.size())
        lstm_out, _ = self.lstm(x_packed)
        last_hidden_state = pad_packed_sequence(lstm_out, batch_first = True)[0][:,-1,:]
        # print(last_hidden_state.size())
        out = self.hidden2sentiment(last_hidden_state)
        return out


# modified from BertPooler
class Pooler(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

# from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # print('[DEBUG] input size:', x.size())
        # print('[DEBUG] positional embedding size:', self.pe.size())
        x = x + self.pe[:x.size(0), :]
        # print('[DEBUG] output x with pe size:', x.size())
        return self.dropout(x)

# class BaselineTransformerWord(nn.Module):
#     def __init__(self, input_dim = 840, hidden_dim = 128, output_dim = 3):
#         super(BaselineTransRNN, self).__init__()
    
#     def forward(self, x):


# class FineTunePretrainedBertWord(nn.Module):
#     def __init__(self, input_dim = 840, hidden_dim = 128, output_dim = 3):
#         super(FineTunePretrainedBertWord, self).__init__()
    
#     def forward(self, x):
