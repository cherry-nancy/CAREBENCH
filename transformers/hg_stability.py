from transformers import PreTrainedTokenizerFast

import torch
import torch.nn.functional as F

import os
import pandas as pd
import numpy as np

from models import CustomGPT, CustomMamba
from models_nesy import CustomGPT as CustomGPTLinear

tokenizer = PreTrainedTokenizerFast(tokenizer_file="data/eventval_tokenizer.json")
tokenizer.eos_token = '[EOS]'
tokenizer.sep_token = '[SEP]'
tokenizer.bos_token = '[BOS]'
tokenizer.pad_token = '[PAD]'
tokenizer.cls_token = '[CLS]'
tokenizer.mask_token = '[MASK]'

DATA_DIR = '/project/ClinicalAI/working_data/seewon/hf_ehr/'
TIME = 1.0
HYPOGLYCEMIA = ['SNOMED/33747003', 'LOINC/LP416145-3', 'LOINC/14749-6']

def inference(inputs, demos, times, temperature=1.0):
    next_probabilities = []

    for i in range(4):
        max_len = max(len(i) for i in inputs)
        demo_inputs = [demo + inputs_i for demo, inputs_i in zip(demos, inputs)]
        tokens = tokenizer(demo_inputs, return_tensors="pt", is_split_into_words=True, padding=True, return_attention_mask=True)
        sequences = tokens["input_ids"].to(device)   
        attention_masks = tokens["attention_mask"].to(device) 

        seq_times = [F.pad(torch.tensor(r, dtype=torch.float32), (4, max_len - len(r))).round(decimals=2)  for r in times]
        seq_times = torch.stack(seq_times, dim=0).to(device) 

        next_ids = model(sequences, seq_times, attention_masks)
        
        next_token_logits = next_ids[:, -1, :] / temperature
        if temperature > 0:
            next_token_logits = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(next_token_logits, num_samples=1)
        else:
            # Greedy decoding
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        inputs = [inputs_i[-1019:] + [next_i] for (inputs_i, next_i) in zip(inputs, tokenizer.convert_ids_to_tokens(next_token))]
        times = [t_i[-1019:] + [t_i[-1] + 0.25] for t_i in times]
        next_probabilities.append(F.softmax(next_ids[:, -1, :], dim=-1))
    next_probabilities = torch.stack(next_probabilities, dim=1)
    return next_probabilities

def hg_loader(i):
    TASK = 'hypoglycemia'
    df = pd.read_parquet(os.path.join(DATA_DIR, f"data/lab_{TASK}/{TASK}2.parquet"), columns=['patient_id', 'start', 'code', 'eventval'])
    df = df[~df['code'].isin(HYPOGLYCEMIA)]
    df = df.drop(columns=['code'])
    
    label_df = pd.read_parquet(os.path.join(DATA_DIR, f"data/lab_{TASK}/{TASK}_label2.parquet"), columns=['patient_id', 'value', 'start', 'Time'])
    label_df = label_df.rename(columns={'value': 'Label', 'start': 'start_time'})
    demo_df = pd.read_parquet(os.path.join(DATA_DIR, f"data/lab_{TASK}/{TASK}_demo.parquet"),
                              columns=['patient_id', 'gender_str', 'ethnicity_str', 'race_str', 'age_str'])

    test_ids = pd.read_csv('hg_1234.csv')
    test_ids = test_ids[i*5:(i+1)*5]
    
    label_df = label_df[label_df['patient_id'].isin(test_ids['patient_id'])]
    demo_df = demo_df[demo_df['patient_id'].isin(test_ids['patient_id'])]
    df = df[df['patient_id'].isin(test_ids['patient_id'])]

    df = df.merge(label_df[['patient_id', 'Time', 'start_time']], on='patient_id')

    df['start'] = (df['start'] - df['start_time']).dt.total_seconds() / 3600
    df['Time'] = (df['Time'] - df['start_time']).dt.total_seconds() / 3600

    df['delta'] = df['Time'] - df['start']
    df = df.drop(columns=['Time', 'start_time'])
    return df, label_df, demo_df, test_ids


def get_inputs(inputs, times):
    max_len = max(len(i) for i in inputs) - 4
    tokens = tokenizer(inputs, return_tensors="pt", is_split_into_words=True, padding=True, return_attention_mask=True)
    sequences = tokens["input_ids"].to(device)   
    attention_masks = tokens["attention_mask"].to(device) 

    times = [F.pad(torch.tensor(r, dtype=torch.float32), (4, max_len - len(r))).round(decimals=2)  for r in times]
    times = torch.stack(times, dim=0).to(device)  

    return sequences, times, attention_masks

def demo_inference(inputs, demos, times, temperature=1.0):
        next_probabilities = []

        for i in range(N):
            max_len = max(len(i) for i in inputs)
            demo_inputs = [demo + inputs_i for demo, inputs_i in zip(demos, inputs)]

            tokens = tokenizer(demo_inputs, return_tensors="pt", is_split_into_words=True, padding=True, return_attention_mask=True)
            sequences = tokens["input_ids"].to(device)   
            attention_masks = tokens["attention_mask"].to(device) 

            seq_times = [F.pad(torch.tensor(r, dtype=torch.float32), (4, max_len - len(r))).round(decimals=2)  for r in times]
            seq_times = torch.stack(seq_times, dim=0).to(device) 

            next_ids = model(sequences, seq_times, attention_masks)
            
            next_token_logits = next_ids[:, -1, :] / temperature
            if temperature > 0:
                next_token_logits = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(next_token_logits, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            inputs = [inputs_i[-1019:] + [next_i] for (inputs_i, next_i) in zip(inputs, tokenizer.convert_ids_to_tokens(next_token))]
            times = [t_i[-1019:] + [t_i[-1] + 0.25] for t_i in times]
            next_probabilities.append(next_ids[:, -1, :])
        next_probabilities = torch.stack(next_probabilities, dim=1)
        return next_probabilities

if __name__ == "__main__":
    model_name = 'gpt_linear'
    device =torch.device("cuda:%d" % 0) if torch.cuda.is_available() else torch.device('cpu')
    N = 3
    if model_name == "gpt":
        model = CustomGPT(vocab_size=len(tokenizer)).to(device)
    elif model_name == "mamba":
        model = CustomMamba(vocab_size=len(tokenizer)).to(device)
    elif model_name == "gpt_linear":
        model = CustomGPTLinear(vocab_size=len(tokenizer)).to(device)
        classifier = torch.nn.Linear(len(tokenizer)*4, 1).to(device)
        classifier.load_state_dict(torch.load('checkpoints/hg_classifier_gpt_1234.pth', map_location=device))
    context_length = 1024
    
    lmhead_state_dict = torch.load(f"checkpoints/hypoglycemia_{model_name}_1234.pth", weights_only=True, map_location=device)
    model.load_state_dict(state_dict=lmhead_state_dict, strict=True)

    for i in range(0, 1174, 5):
        # Load data
        df_all, label_df, demo_df, probs_df = hg_loader(i)

        df_i = None
        TIMES = np.arange(0.5, 4.0, 0.1)
        for n, TIME in enumerate(TIMES):
            df = df_all[df_all['delta'] >= TIME]
            if n > 0:
                valid_ids = df_all[(df_all['delta'] > TIMES[n-1]) & df_all['delta'] <= TIME]['patient_id'].unique()
                df = df[df['patient_id'].isin(valid_ids)]

            df = df.drop(columns='delta')

            df = df.sort_values(['patient_id', 'start']).groupby('patient_id').agg(list).reset_index()

            df = df[df['eventval'].apply(len) >= 5]
            df['eventval'] = df['eventval'].apply(lambda x: x[-(context_length-4):] if len(x) > context_length else x)
            df['start'] = df['start'].apply(lambda x: x[-(context_length-4):] if len(x) > context_length else x)

            df = df.merge(label_df[['patient_id', 'Label']], on='patient_id', how='left')
            df = df.merge(demo_df, on='patient_id', how='left')
            
            valid_ids = df['patient_id'].unique()

            df['demo_str'] = df.apply(lambda sample: [sample['age_str'], sample['gender_str'], sample['race_str'], sample['ethnicity_str']], axis=1)
            demos = list((df['demo_str']).values)
            input_ids = list((df['eventval']).values)
            times = list(df['start'].values) 

            if 'task' == 1:
                input_ids, times, attention_masks = get_inputs(input_ids, times)
                output = model(input_ids, times, attention_masks)
            else:
                next_probs = inference(input_ids, demos, times)
                output = classifier(next_probs.flatten(1)).squeeze(-1)
            output = torch.sigmoid(output).squeeze(-1)

            if df_i is None:
                df_i = pd.DataFrame({
                'ids': valid_ids,
                f'{TIME}': output.cpu().detach().numpy()
                })
            else:
                df_temp = pd.DataFrame({
                'ids': valid_ids,
                f'{TIME}': output.cpu().detach().numpy()
                })
                df_i = df_i.merge(df_temp, on='ids', how='outer')
            
        df_i.to_csv(f'hg_stability/{model_name}_prob_{i}.csv', index=False)
