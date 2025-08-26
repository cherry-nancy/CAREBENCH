import torch
from torch.utils.data import Dataset

import os
import pandas as pd
import numpy as np
import random
import pickle

from sklearn.model_selection import train_test_split

COLS_NAME = ['eventval', 'time_arrive']
DATA_DIR = '/home/mkeoliya/projects/CAREBench/data/mcmed'

class MCMEDDataset(Dataset):
    def __init__(self, data):
        self.samples = data
        self.index_map = list(range(len(self.samples)))
        random.shuffle(self.index_map)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples.iloc[self.index_map[idx]] 
        event = [sample['Gender_str'], sample['Race_str'], sample['Ethnicity_str'], sample['Age_str']] + list(sample['eventval'])
        time = list(sample['time_arrive'])
        return event, time

    @staticmethod
    def collate_fn(batch):
        events = [item[0] for item in batch]
        times = [item[1] for item in batch]
        return events, times

def mcmed_loader(batch_size, context_length = 1024, seed=1234):
    def sample_list_preserve_order(lst):
        if len(lst) > context_length:
            indices = sorted(random.sample(range(len(lst)), context_length))
            return [lst[i] for i in indices]
        else:
            return lst

    demo_df = pd.read_csv(os.path.join(DATA_DIR, 'demo.csv'), usecols=['CSN', 'Gender_str', 'Race_str', 'Ethnicity_str', 'Age_str'])  
    labs_df = pd.read_csv(os.path.join(DATA_DIR, 'labs.csv'), usecols=['CSN', 'eventval', 'time_arrive'])   
    numerics_df = pd.read_csv(os.path.join(DATA_DIR, 'numerics.csv'), usecols=['CSN', 'eventval', 'time_arrive'])
    orders_df = pd.read_csv(os.path.join(DATA_DIR, 'orders.csv'), usecols=['CSN', 'eventval', 'time_arrive'])  

    df = pd.concat([labs_df, numerics_df, orders_df], ignore_index=True)
    df = df.sort_values(['CSN', 'time_arrive']).groupby('CSN').agg(list).reset_index()
    df = df.merge(demo_df, on='CSN', how='left')
    df['eventval'] = df['eventval'].apply(sample_list_preserve_order)
    
    train_inputs, test_inputs = train_test_split(
        df, test_size=0.2, random_state=seed
    )

    train = MCMEDDataset(train_inputs)
    train_loader = torch.utils.data.DataLoader(train, collate_fn=MCMEDDataset.collate_fn, batch_size=batch_size, shuffle=True)

    test = MCMEDDataset(test_inputs)
    test_loader =torch.utils.data.DataLoader(test, collate_fn=MCMEDDataset.collate_fn, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader