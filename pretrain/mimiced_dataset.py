import torch
from torch.utils.data import Dataset

import os
import pandas as pd
import numpy as np
import random
import pickle

from sklearn.model_selection import train_test_split

COLS_NAME = ['eventval', 'time_arrive']
DATA_DIR = '/home/mkeoliya/projects/CAREBench/data/mimiciv/3.1/ed'

class MIMICEDDataset(Dataset):
    def __init__(self, data):
        self.samples = data
        self.index_map = list(range(len(self.samples)))
        random.shuffle(self.index_map)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples.iloc[self.index_map[idx]] 
        event = [sample['Gender_str'], sample['Race_str']] + list(sample['eventval'])
        time = list(sample['time_arrive'])
        return event, time

    @staticmethod
    def collate_fn(batch):
        events = [item[0] for item in batch]
        times = [item[1] for item in batch]
        return events, times

def mimiced_loader(batch_size, context_length = 1024, seed=1234):
    def sample_list_preserve_order(lst):
        if len(lst) > context_length:
            indices = sorted(random.sample(range(len(lst)), context_length))
            return [lst[i] for i in indices]
        else:
            return lst

    demo_df = pd.read_csv(os.path.join(DATA_DIR, 'demo.csv'), usecols=['subject_id', 'Gender_str', 'Race_str'])  
    med_df = pd.read_csv(os.path.join(DATA_DIR, 'med.csv'), usecols=['subject_id', 'stay_id', 'eventval', 'time_arrive'])   
    numerics_df = pd.read_csv(os.path.join(DATA_DIR, 'numerics.csv'), usecols=['subject_id', 'stay_id', 'eventval', 'time_arrive'])

    df = pd.concat([numerics_df, med_df], ignore_index=True)
    df = df.sort_values(['stay_id', 'time_arrive']).groupby(['subject_id', 'stay_id']).agg(list).reset_index()
    df = df.merge(demo_df, on='subject_id', how='left')
    df['eventval'] = df['eventval'].apply(sample_list_preserve_order)
    
    train_inputs, test_inputs = train_test_split(
        df, test_size=0.2, random_state=seed
    )

    train = MIMICEDDataset(train_inputs)
    train_loader = torch.utils.data.DataLoader(train, collate_fn=MIMICEDDataset.collate_fn, batch_size=batch_size, shuffle=True)

    test = MIMICEDDataset(test_inputs)
    test_loader =torch.utils.data.DataLoader(test, collate_fn=MIMICEDDataset.collate_fn, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader