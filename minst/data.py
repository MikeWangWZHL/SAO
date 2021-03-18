import torch
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
# import pandas as pd

class MINSTBrainDataset(Dataset):
    def __init__(self, sample_dict):
        self.data = [sample for sample in sample_dict.values()]
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        channel_tensors = [torch.Tensor(ch) for ch in sample['channels'].values()]
        # concat_signal = torch.stack(channel_tensors)
        size_tensor = torch.tensor(sample['size'], dtype = torch.float64)
        label_tensor = torch.tensor(sample['label'])

        return channel_tensors, size_tensor, label_tensor
        
def load_minst_brain_txt(txt_path):
    
    def parse_signal_str(signal_str):
        signal_str = signal_str.strip()
        parsed_signal_str = signal_str.split(',')
        signal_list = [float(sig) for sig in parsed_signal_str]
        return np.array(signal_list)
        # return signal_list
    
    max_length = 0
    min_length = 1e10
    
    sample_dict = {}
    with open(txt_path) as f:
        for line in f:
            parsed_line = line.split('\t')
            event_id = parsed_line[1]
            label = int(parsed_line[4])
            if label == -1:
                continue
            device = parsed_line[2]
            channel = parsed_line[3]
            size = int(parsed_line[5])
            signal = parse_signal_str(parsed_line[6])

            if len(signal) < min_length:
                min_length = len(signal)
            elif len(signal) > max_length:
                max_length = len(signal)

            if event_id not in sample_dict:
                sample_dict[event_id] = {'channels':{},'label':label,'device':device, 'size':size}
            sample_dict[event_id]['channels'][channel] = signal
    
    return sample_dict, min_length, max_length



if __name__ == "__main__":
    minst_dataset_IN_path = './dataset/IN.txt'
    sample_dict,min_length,max_length = load_minst_brain_txt(minst_dataset_IN_path)
    # print(min_length,max_length)
    # print(sample_dict['173652'])
    # print(isinstance(sample_dict['173652']['channels']['AF3']['signal'][0],float))
    print(MINSTBrainDataset(sample_dict)[0])

