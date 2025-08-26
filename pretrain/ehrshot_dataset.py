import torch
import os
import pandas as pd
import random
from torch.utils.data import Dataset

from sklearn.model_selection import train_test_split
# from hf_ehr.data.tokenization import CLMBRTokenizer
# from hf_ehr.config import Event

DATA_DIR = '/home/mkeoliya/projects/CAREBench/data/ehrshot'

class EHRShotDataset(Dataset):
    def __init__(self, data):
        self.samples = data[['eventval', 'time_arrive']]
        self.index_map = list(range(len(self.samples)))
        random.shuffle(self.index_map)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples.iloc[self.index_map[idx]] 
        return list(sample['eventval']), sample['time_arrive']

    @staticmethod
    def collate_fn(batch):
        events = [item[0] for item in batch]
        times = [item[1] for item in batch]
        return events, times


def ehrshot_loader(batch_size, context_length=1024, seed=1234):
    def sample_list_preserve_order(lst):
        if len(lst) > context_length:
            indices = sorted(random.sample(range(len(lst)), context_length))
            return [lst[i] for i in indices]
        else:
            return lst
    med_df = pd.read_csv(os.path.join(DATA_DIR, 'med.csv'), usecols=['CSN', 'time_arrive', 'eventval', 'group'])
    num_df = pd.read_csv(os.path.join(DATA_DIR, 'numerics.csv'), usecols=['CSN', 'time_arrive', 'eventval', 'group'])
    df = pd.concat([med_df, num_df], ignore_index=True)
    
    df = df.sort_values(['CSN', 'time_arrive']).groupby(['CSN', 'group']).agg(list).reset_index()    
    df = df.dropna()
    df = df[df['eventval'].map(len) > 2]

    demo_df = pd.read_csv(os.path.join(DATA_DIR, 'demo.csv'), usecols=['CSN', 'Ethnicity_str', 'Race_str', 'Gender_str', 'Age_str'])
    df = df.merge(demo_df, on='CSN', how='left')
    df['eventval'] = df['eventval'].apply(sample_list_preserve_order)

    train_inputs, test_inputs = train_test_split(
        df, test_size=0.2, random_state=seed
    )
    
    train = EHRShotDataset(train_inputs)
    test = EHRShotDataset(test_inputs)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, collate_fn=EHRShotDataset.collate_fn, shuffle=True)
    test_loader =torch.utils.data.DataLoader(test, batch_size=batch_size, collate_fn=EHRShotDataset.collate_fn, shuffle=True)

    return train_loader, test_loader