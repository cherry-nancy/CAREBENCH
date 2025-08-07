import torch
from torch.utils.data import Dataset

import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

DATA_DIR = '/project/ClinicalAI/working_data/seewon/mc-med/'
TIME = 1.5

class SickDataset(Dataset):
    def __init__(self, data, backbone='gpt', sample=False, oversample=False):
        data = data[['eventval', 'time', 'Label', 'Age_str', 'Gender_str', 'Race_str', 'Ethnicity_str']]
        if backbone == 'gpt':
            model = GPT2Model.from_pretrained("gpt2")
        elif backbone == 'mamba':
            # model = AutoModelForCausalLM.from_pretrained("StanfordShahLab/mamba-tiny-16384-clmbr")
            model = AutoModelForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
    
        events, times, labels = [], [], []
        for _, r in data.iterrows():
            event = [r['Age_str'], r['Gender_str'], r['Race_str'], r['Ethnicity_str']] + list(r['eventval'])
            time = [0, 0, 0, 0] + r['time']
            label = float(r['Label'])

            if sample:
                new_event, new_time = [], []
                if label == 1:
                    for e, t in zip(event, time):
                        if random.random() <= 0.80:
                            new_event.append(e)
                            new_time.append(t)
                    event = new_event
                    time = new_time
            assert(len(event) == len(time))

            events.append(event)
            times.append(time)
            labels.append(label)

        tokens = tokenizer(events, return_tensors="pt", is_split_into_words=True, padding=True, return_attention_mask=True)
        inputs = tokens["input_ids"] 

        if backbone == 'gpt':
            inputs = model.wte(inputs)
        elif backbone == 'mamba':
            inputs = model.backbone.embeddings(inputs)

        max_len = inputs.shape[1]
        times = [F.pad(torch.tensor(r, dtype=torch.float32), (0, max_len - len(r))).round(decimals=4)  for r in times]
        times = torch.stack(times, dim=0) 

        labels = torch.tensor(labels)

        _, counts = torch.unique(labels, return_counts=True)
        print(f"Original: {counts}")

        if oversample == True:
            data = torch.concat((inputs, times.unsqueeze(-1)), dim=-1)
            smote = SMOTE(random_state=1234, sampling_strategy='auto')
            X_balanced, y_balanced = smote.fit_resample(data.flatten(1).detach().numpy(), labels.detach().numpy())
            print(X_balanced.shape)
            X_balanced = X_balanced.reshape(len(y_balanced), -1, 769)
            
            self.events = torch.tensor(X_balanced[:, :, :-1])
            self.times = torch.tensor(X_balanced[:, :, -1])
            self.labels = torch.tensor(y_balanced)
        else:
            self.events = torch.tensor(inputs.detach().numpy())
            self.labels = torch.tensor(labels.detach().numpy())
            self.times = torch.tensor(times.detach().numpy())

        _, counts = torch.unique(self.labels, return_counts=True)
        print(f"New: {counts}")

        self.index_map = list(range(len(self.labels)))
        random.shuffle(self.index_map)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        event = self.events[self.index_map[idx]]
        time = self.times[self.index_map[idx]]
        label = self.labels[self.index_map[idx]]
        return event, time, label

    @staticmethod
    def collate_fn(batch):
        events = torch.stack([item[0] for item in batch], dim=0)
        times = torch.stack([item[1] for item in batch], dim=0)
        labels = torch.stack([item[2] for item in batch], dim=0)
        return events, times, labels

class SickDataset(Dataset):
    def __init__(self, data, sample=False, backbone=None):
        """
        Args:
            ecg_data: numpy array of shape (total_samples, n_leads)
            labels: numpy array with timestamps of abnormal events
            window_size: number of samples in each snippet
        """
        self.samples, labels = [], []
        data = data[['eventval', 'time', 'Label', 'Age_str', 'Gender_str', 'Race_str', 'Ethnicity_str']]
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
        time = list(sample['time'])
        demo = [sample['Age_str'], sample['Gender_str'], sample['Race_str'], sample['Ethnicity_str']]
        return event, demo, time, label

    @staticmethod
    def collate_fn(batch):
        events = [item[0] for item in batch]
        # demos = [item[1] for item in batch]
        demos = None
        times = [item[2] for item in batch]
        labels = [item[3] for item in batch]
        labels = torch.tensor(labels)
        return events, demos, times, labels

def create_df_limit(df, context_length):
    label_df = pd.read_parquet(os.path.join(DATA_DIR, "data/eSOFA.parquet"))
    demo_df = pd.read_parquet(os.path.join(DATA_DIR, "data/eSOFA_demo.parquet"),
                              columns=['CSN', 'Age_str', 'Race_str', 'Ethnicity_str', 'Gender_str'])

    label_df['Time'] = label_df.apply(lambda x: x['Trigger_time'] if x['Label'] else x['Sample_time'], axis=1)
    label_df = label_df[(label_df['Time'] >= TIME)]
    df = df.merge(label_df[['CSN', 'Time']], on='CSN')

    df = df[df['time'] <= (df['Time'] - TIME)]
    df = df.drop(columns=['Time'])
    df = df.sort_values(['CSN', 'time']).groupby('CSN').agg(list).reset_index()

    df['value'] = df['value'].apply(lambda x: x[-context_length:] if len(x) > context_length else x)
    df['event'] = df['event'].apply(lambda x: x[-context_length:] if len(x) > context_length else x)
    df['eventval'] = df['eventval'].apply(lambda x: x[-context_length:] if len(x) > context_length else x)
    df['time'] = df['time'].apply(lambda x: x[-context_length:] if len(x) > context_length else x)

    df = df.merge(label_df[['CSN', 'Label']], on='CSN', how='left')
    df = df.merge(demo_df, on='CSN')

    return df


def sepsis_loader(time, batch_size, SickDataset, model, context_length=1000, seed=1234, plot=False):
    df = pd.read_parquet(os.path.join(DATA_DIR, "data/eSOFA_data.parquet"))
    df = create_df_limit(df, context_length)

    train_inputs, df_temp = train_test_split(df, test_size=0.4, random_state=seed)
    test_inputs, val_inputs = train_test_split(df_temp, test_size=0.5, random_state=seed)

    train = SickDataset(train_inputs, sample=True, backbone=model)
    train_loader = torch.utils.data.DataLoader(train, collate_fn=SickDataset.collate_fn, batch_size=batch_size, shuffle=True)

    val = SickDataset(val_inputs, backbone=model)
    val_loader = torch.utils.data.DataLoader(val, collate_fn=SickDataset.collate_fn, batch_size=batch_size, shuffle=True)

    test = SickDataset(test_inputs, backbone=model)
    test_loader =torch.utils.data.DataLoader(test, collate_fn=SickDataset.collate_fn, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader