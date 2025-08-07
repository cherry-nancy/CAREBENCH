import torch
from torch.utils.data import Dataset

import pandas as pd
import random
from sklearn.model_selection import train_test_split

TIME = 6.0

class LabDataset(Dataset):
    def __init__(self, data):
        self.samples, labels = [], []
        data = data[['eventval', 'charttime', 'Label', 'demo_str']]
        for _, d in data.iterrows():
            if d['Label'] == 1: 
                self.samples.append(d)
                labels.append(d['Label'])
            else: 
                if random.random() < 1.0: 
                    self.samples.append(d)
                    labels.append(d['Label'])
        
        self.index_map = list(range(len(self.samples)))
        random.shuffle(self.index_map)

        _, counts = torch.unique(torch.tensor(labels), return_counts=True)
        print(f"Original: {counts}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[self.index_map[idx]] 
        label = float(sample['Label'])
        event = list(sample['demo_str']) + list(sample['eventval'])
        time = list(sample['charttime'])
        return event, time, label

    @staticmethod
    def collate_fn(batch):
        events = [item[0] for item in batch]
        times = [item[1] for item in batch]
        labels = [item[2] for item in batch]
        labels = torch.tensor(labels)
        return events, times, labels

class LabDataset(Dataset):
    def __init__(self, data):
        self.samples, labels = [], []
        data = data[['eventval', 'charttime', 'Label', 'demo_str']]
        for _, d in data.iterrows():
            if d['Label'] == 1: 
                self.samples.append(d)
                labels.append(d['Label'])
            else: 
                if random.random() < 1.0: 
                    self.samples.append(d)
                    labels.append(d['Label'])
        
        self.index_map = list(range(len(self.samples)))
        random.shuffle(self.index_map)

        _, counts = torch.unique(torch.tensor(labels), return_counts=True)
        print(f"Original: {counts}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[self.index_map[idx]] 
        label = float(sample['Label'])
        event = list(sample['eventval'])
        time = list(sample['charttime'])
        demo = list(sample['demo_str'])
        return event, demo, time, label

    @staticmethod
    def collate_fn(batch):
        events = [item[0] for item in batch]
        demos = [item[1] for item in batch]
        times = [item[2] for item in batch]
        labels = [item[3] for item in batch]
        labels = torch.tensor(labels)
        return events, demos, times, labels


def create_df_limit(df, label_df, demo_df, context_length):
    df = df.merge(label_df[['stay_id', 'Time', 'intime']], on='stay_id')
    df['charttime'] = pd.to_datetime(df['charttime'])
    df['intime'] = pd.to_datetime(df['intime'])

    df['charttime'] = (df['charttime'] - df['intime']).dt.total_seconds() / 3600

    df[df['Time'] >= TIME]
    df = df[df['charttime'] <= (df['Time'] - TIME)]
    df = df.drop(columns=['Time', 'intime'])
    df = df.sort_values(['stay_id', 'charttime']).groupby('stay_id').agg(list).reset_index()

    df['eventval'] = df['eventval'].apply(lambda x: x[-context_length:] if len(x) > context_length else x)
    df['charttime'] = df['charttime'].apply(lambda x: x[-context_length:] if len(x) > context_length else x)

    df = df.merge(label_df[['stay_id', 'Label']], on='stay_id', how='left')
    df = df.merge(demo_df, on='stay_id')

    return df

def icu_loader(batch_size, SickDataset, context_length=1000, seed=1234):
    med_df = pd.read_csv(f"data/transfer/med.csv", usecols=['stay_id', 'charttime', 'eventval'])
    num_df = pd.read_csv(f"data/transfer/numerics.csv", usecols=['stay_id', 'charttime', 'eventval'])
    df = pd.concat([med_df, num_df]).sort_values(['stay_id', 'charttime'])
    
    label_df = pd.read_csv("data/transfer/label.csv", usecols=['stay_id', 'Label', 'intime', 'Time'])
    demo_df = pd.read_csv('data/transfer/demo.csv', usecols=['stay_id', 'gender_str', 'race_str', 'age_str'])
    demo_df['demo_str'] = demo_df.apply(lambda x: [x['race_str'], x['gender_str'], x['age_str']], axis=1)
    demo_df = demo_df.drop(columns=['gender_str', 'race_str', 'age_str'])
    
    df = create_df_limit(num_df, label_df, demo_df, context_length)

    train_inputs, df_temp = train_test_split(df, test_size=0.4, random_state=seed)
    test_inputs, val_inputs = train_test_split(df_temp, test_size=0.5, random_state=seed)

    train = SickDataset(train_inputs)
    train_loader = torch.utils.data.DataLoader(train, collate_fn=SickDataset.collate_fn, batch_size=batch_size, shuffle=True)

    val = SickDataset(val_inputs)
    val_loader = torch.utils.data.DataLoader(val, collate_fn=SickDataset.collate_fn, batch_size=batch_size, shuffle=True)

    test = SickDataset(test_inputs)
    test_loader =torch.utils.data.DataLoader(test, collate_fn=SickDataset.collate_fn, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader