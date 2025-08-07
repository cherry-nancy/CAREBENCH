import torch
from torch.utils.data import Dataset

import random
import pandas as pd
from sklearn.model_selection import train_test_split

TIME = 12.0

class LabDataset(Dataset):
    def __init__(self, data):
        self.samples, labels = [], []
        data = data[['eventval', 'time', 'Label', 'demo_str']]
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

def create_df_limit(label_df, demo_df, context_length, stay=True):
    df = None
    for i in range(0, 37):
        df_i = pd.read_parquet(f'data/mortality/{i}_final.parquet', columns=['subject_id', 'stay_id', 'time', 'eventval'])
        # df_i = df_i.dropna(subset=['time', 'eventval'])
        df_i = df_i.merge(label_df[['subject_id', 'stay_id', 'Time', 'intime']], on='subject_id')
        if stay:
            df_i = df_i[df_i['stay_id_x'] == df_i['stay_id_y']]
            df_i = df_i.drop(columns=['stay_id_x', 'stay_id_y'])
        else:
            df_i = df_i[(df_i['stay_id_x'] == df_i['stay_id_y']) | (df_i['stay_id_x'].isna())]
            df_i = df_i.drop(columns=['stay_id_x', 'stay_id_y'])
        df_i['time'] = pd.to_datetime(df_i['time'])
        df_i['intime'] = pd.to_datetime(df_i['intime'])
        df_i['Time'] = pd.to_datetime(df_i['Time'])

        df_i['time'] = (df_i['time'] - df_i['intime']).dt.total_seconds() / 3600
        df_i['Time'] = (df_i['Time'] - df_i['intime']).dt.total_seconds() / 3600

        df_i[df_i['Time'] >= TIME]
        df_i = df_i[df_i['time'] <= (df_i['Time'] - TIME)]
        df_i = df_i.drop(columns=['Time', 'intime'])
        df_i = df_i.sort_values(['subject_id', 'time']).groupby('subject_id').agg(list).reset_index()

        df_i['eventval'] = df_i['eventval'].apply(lambda x: x[-context_length:] if len(x) > context_length else x)
        df_i['time'] = df_i['time'].apply(lambda x: x[-context_length:] if len(x) > context_length else x)

        df_i = df_i.merge(label_df[['subject_id', 'Label']], on='subject_id', how='left')
        df_i = df_i.merge(demo_df, on='subject_id', how='left')

        if df is None: df = df_i
        else: df = pd.concat([df, df_i])
    df = df.drop(columns='subject_id')
    return df

def mortality_loader(batch_size, SickDataset, context_length=1000, seed=1234):
    label_df = pd.read_csv("data/mortality/mortality_labels.csv", usecols=['subject_id', 'stay_id', 'Label', 'intime', 'Time'])
    demo_df = pd.read_parquet('data/mortality/demographics.parquet', columns=['race_str', 'gender_str', 'ethnicity_str', 'age_str'])
    demo_df['subject_id'] = pd.to_numeric(demo_df.index)
    demo_df['demo_str'] = demo_df.apply(lambda x: [x['race_str'], x['ethnicity_str'], x['gender_str'], x['age_str']], axis=1)
    demo_df = demo_df.drop(columns=['gender_str', 'race_str', 'age_str', 'ethnicity_str'])
    
    df = create_df_limit(label_df, demo_df, context_length)

    train_inputs, df_temp = train_test_split(df, test_size=0.4, random_state=seed)
    test_inputs, val_inputs = train_test_split(df_temp, test_size=0.5, random_state=seed)

    train = SickDataset(train_inputs)
    train_loader = torch.utils.data.DataLoader(train, collate_fn=SickDataset.collate_fn, batch_size=batch_size, shuffle=True)

    val = SickDataset(val_inputs)
    val_loader = torch.utils.data.DataLoader(val, collate_fn=SickDataset.collate_fn, batch_size=batch_size, shuffle=True)

    test = SickDataset(test_inputs)
    test_loader =torch.utils.data.DataLoader(test, collate_fn=SickDataset.collate_fn, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader