import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

# NOT WORKING: nan loss
# class BrainTranslator_Naive(nn.Module):
#     def __init__(self, pretrained_layers, in_feature = 840, hidden_size = 1024):
#         super(BrainTranslator, self).__init__()
#         self.pretrained_BART = pretrained_layers
#         # using a simple linear projection for now, TODO: try other methods
#         self.fc1 = nn.Linear(in_feature, hidden_size)

#     def forward(self, input_embeddings_batch, input_masks_batch, target_ids_batch):
#         # input_embeddings_batch: batch_size*Seq_len*840
#         # print(input_embeddings_batch.size())
#         embedding = self.fc1(input_embeddings_batch) # batch_size*Seq_len*1024
#         # embedding = F.relu(self.fc1(input_embeddings_batch)) # batch_size*Seq_len*1024

#         # print(embedding.size())
#         out = self.pretrained_BART(inputs_embeds = embedding, attention_mask = input_masks_batch, decoder_input_ids = target_ids_batch, labels = target_ids_batch, return_dict = True)          
#         return out

class BrainTranslator(nn.Module):
    def __init__(self, pretrained_layers, in_feature = 840, decoder_embedding_size = 1024, additional_encoder_nhead=8, additional_encoder_dim_feedforward = 2048):
        super(BrainTranslator, self).__init__()
        
        self.pretrained_BART = pretrained_layers
        # additional transformer encoder, following BART paper about 
        self.additional_encoder_layer = nn.TransformerEncoderLayer(d_model=in_feature, nhead=additional_encoder_nhead,  dim_feedforward = additional_encoder_dim_feedforward, batch_first=True)
        self.additional_encoder = nn.TransformerEncoder(self.additional_encoder_layer, num_layers=6)
        self.fc1 = nn.Linear(in_feature, decoder_embedding_size)

    def forward(self, input_embeddings_batch, input_masks_batch, target_ids_batch):
        # input_embeddings_batch: batch_size*Seq_len*840
        encoded_embedding = self.additional_encoder(input_embeddings_batch)
        encoded_embedding = F.relu(self.fc1(input_embeddings_batch))
        
        out = self.pretrained_BART(inputs_embeds = encoded_embedding, attention_mask = input_masks_batch, decoder_input_ids = target_ids_batch, return_dict = True)          
        # out = self.pretrained_BART(inputs_embeds = encoded_embedding, attention_mask = input_masks_batch, decoder_input_ids = target_ids_batch, labels = target_ids_batch, return_dict = True)          
        return out


class BrainTranslatorBert(nn.Module):
    def __init__(self, pretrained_layers, in_feature = 840, hidden_size = 768):
        super(BrainTranslatorBert, self).__init__()

        self.pretrained_Bert = pretrained_layers
        self.fc1 = nn.Linear(in_feature, hidden_size)

    def forward(self, input_embeddings_batch, input_masks_batch, target_ids_batch):
        embedding = F.relu(self.fc1(input_embeddings_batch))
        out = self.pretrained_Bert(inputs_embeds = embedding, attention_mask = input_masks_batch, labels = target_ids_batch, return_dict = True)
        return out



class EEG2BertMapping(nn.Module):
    def __init__(self, in_feature = 840, hidden_size = 512, out_feature = 768):
        super(EEG2BertMapping, self).__init__()
        self.fc1 = nn.Linear(in_feature, hidden_size)
        self.fc2 = nn.Linear(hidden_size, out_feature)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out
