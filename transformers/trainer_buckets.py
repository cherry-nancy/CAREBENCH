from transformers import PreTrainedTokenizerFast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torcheval.metrics import BinaryAUROC, BinaryF1Score, BinaryPrecision, BinaryRecall, BinaryConfusionMatrix, BinaryAUPRC

import random
from time import time
from tqdm import tqdm
import wandb
from argparse import ArgumentParser
from collections import OrderedDict

from models_nesy import CustomGPT
from decomp_dataset import decomp_loader
from sepsis_dataset import sepsis_loader
from hg_dataset import hg_loader
from hk_dataset import hk_loader
from transfer_dataset import icu_loader
from mortality_dataset import mortality_loader

NUM_COLS = ['HR', 'SpO2', 'MAP'] 
CRITERIA = ['HR|106.0667-268.0', 'HR|268.0-', 'MAP|1.0-75.6667', 'SpO2|0.0-94.7907']

class SickDataset(Dataset):
    def __init__(self, data, sample=False, backbone=None):
        self.samples, labels = [], []
        data = data[['event', 'bucket', 'time_arrive', 'Label']]
        
        if sample: threshold = 0.8
        else: threshold = 1.0

        for _, d in data.iterrows():
            if d['Label'] == 1: 
                self.samples.append(d)
                labels.append(d['Label'])
            else: 
                if random.random() < threshold: 
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
        event = list(sample['event'])
        value = list(sample['bucket'].round(2))
        time = list(sample['time_arrive'].round(2))
        return event, value, time, label

    @staticmethod
    def collate_fn(batch):
        events = [item[0] for item in batch]
        values = [item[1] for item in batch]
        times = [item[2] for item in batch]
        labels = [item[3] for item in batch]
        labels = torch.tensor(labels)
        return events, values, times, labels

class Trainer():
    def __init__(self, model, train_loader, test_loader, val_loader, learning_rate, gpu, save_model=False):
        if gpu >= 0 and torch.cuda.is_available(): device = torch.device("cuda:%d" % gpu)
        else: device = torch.device("cpu")
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.save_model = save_model
        self.best_acc = 0
        self.criterion = nn.BCELoss()
        self.N = 3

        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=f"data/{args.tokenizer}_tokenizer.json")
        self.tokenizer.eos_token = '[EOS]'
        self.tokenizer.sep_token = '[SEP]'
        self.tokenizer.bos_token = '[BOS]'
        self.tokenizer.pad_token = '[PAD]'
        self.tokenizer.cls_token = '[CLS]'
        self.tokenizer.mask_token = '[MASK]'

        self.network = model(vocab_size = len(self.tokenizer), full=False).to(self.device)
        if args.model == 'gpt':
            lmhead_state_dict = torch.load(f"checkpoints/gpt_eventval.pth", map_location=self.device, weights_only=True)
            new_state_dict = OrderedDict()
            for key, value in lmhead_state_dict.items():
                if ('wte' not in key) and (('lm_head' not in key)):
                    new_key = key.replace('model.transformer', 'model')
                    new_state_dict[new_key] = value
            self.network.load_state_dict(new_state_dict, strict=False)

        self.optimizer = torch.optim.AdamW(list(filter(lambda p: p.requires_grad, self.network.parameters())), lr=learning_rate)
    
    def program(self, ps):
        output = torch.zeros(len(CRITERIA)).to(self.device)
        for i, sofa_type in enumerate([]):
            criteria = [s[i] for s in CRITERIA]
            ps_i = [p_i[1] for p_i in ps if p_i[0] == sofa_type]
            if len(ps_i) == 0: continue
            
            ps_i = torch.stack(ps_i, dim=-1).unsqueeze(-1)
            for j, (c, p_i) in enumerate(zip(criteria, ps_i)):
                if torch.isnan(torch.tensor(c)): continue
                if ((i == 0 or i == 1) and any(p_i > c)) or ((i == 2 or i == 3) and any(p_i < c)):
                    output[j] = 1
        return output

    def train_epoch(self, epoch):
        train_loss = 0
        total_correct = 0
        num_items = 0
        iter = tqdm(self.train_loader, total=len(self.train_loader))
        self.network.train()
        for (i, (events, demos, values, times, target)) in enumerate(iter):
            self.optimizer.zero_grad()
            target = target.to(self.device)

            next_outputs = self.inference(events, demos, values, times)
            output = self.program(next_outputs)

            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            avg_loss = train_loss / (i + 1)
            
            output = F.sigmoid(output)
            total_correct += (output.round()==target).float().sum() 
            num_items += output.shape[0]
            train_acc = (total_correct / num_items)
            iter.set_description(f"[Train Epoch {epoch}] Loss: {avg_loss:.4f} Acc: {total_correct}/{num_items} {train_acc:.4f}")
        return train_loss / len(self.train_loader)

    def test_epoch(self, epoch):
        total_correct = 0
        num_items = 0
        iter = tqdm(self.test_loader, total=len(self.test_loader))
        self.network.eval()
        
        auroc = BinaryAUROC().to(self.device)
        auprc = BinaryAUPRC().to(self.device)
        f1 = BinaryF1Score().to(self.device)
        precision = BinaryPrecision().to(self.device)
        recall = BinaryRecall().to(self.device)
        confusion_matrix = BinaryConfusionMatrix().to(self.device)
        
        for (i, (events, demos, values, times, target)) in enumerate(iter):
            with torch.no_grad():
                target = target.to(self.device)

                next_tokens = self.inference(events, demos, values, times, temperature=0)
                output = self.program(next_tokens)

                output = F.sigmoid(output)
                preds = output.round()

                total_correct += (preds==target).float().sum() 
                num_items += output.shape[0]
                test_acc = (total_correct / num_items)

                target = target.long()
                f1.update(preds, target)
                precision.update(preds, target)
                recall.update(preds, target)
                auroc.update(output, target)
                auprc.update(output, target)
                confusion_matrix.update(preds, target)

                iter.set_description(f"[Test Epoch {epoch}] Acc: {total_correct}/{num_items} {test_acc:.4f}")
        
        f1 = f1.compute()
        auc = auroc.compute()
        precision = precision.compute()
        recall = recall.compute()
        conf_matrix = confusion_matrix.compute()
        auprc = auprc.compute()

        print("Confusion Matrix:\n", conf_matrix)
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}, AUPRC: {auprc:.4f}")

        return test_acc, auc, f1, precision, recall, auprc

    def inference(self, inputs, demos, values, times, temperature=1.0):
        outputs = []

        sofa_tokens = NUM_COLS
        for i in range(self.N):
            next_inputs, next_values = [], []
            max_len = max(len(i) for i in inputs)
            for j in sofa_tokens:
                max_len_j = max_len + 1
                
                demo_inputs = [demo + inputs_i for demo, inputs_i in zip(demos, inputs)]
                tokens = self.tokenizer(demo_inputs, return_tensors="pt", is_split_into_words=True, padding=True, return_attention_mask=True)
                sequences = tokens["input_ids"].to(self.device)   
                attention_masks = tokens["attention_mask"].to(self.device) 

                times_j = [times_i + [times_i[-1] + 0.5] for times_i in times]
                seq_times = [F.pad(torch.tensor(r, dtype=torch.float32), (4, max_len_j - len(r))).round(decimals=2)  for r in times_j]
                seq_times = torch.stack(seq_times, dim=0).to(self.device) 

                values_j = [values_i + [0.0] for values_i in values]
                seq_values = [F.pad(torch.tensor(r, dtype=torch.float32), (0, max_len_j - len(r)))  for r in values_j]
                seq_values = torch.stack(seq_values, dim=0).to(self.device) 

                next_value = self.network(sequences, seq_values, seq_times, attention_masks)
                
                next_inputs.append(j)
                next_values.append(next_value.squeeze(-1))
            
            next_values = torch.stack(next_values, dim=-1)
            inputs = [input_i + next_inputs for input_i in inputs]
            times = [times_i + [times_i[-1] + 0.5]*len(next_inputs) for times_i in times]
            values = [values_i + list(next_i) for values_i, next_i in zip(values, next_values)]

            outputs += [(e, v) for e, v in zip(sofa_tokens, next_values.t())]
        return outputs

    def train(self, n_epochs):
        for epoch in range(1, n_epochs+1):
            t0 = time()
            train_loss = self.train_epoch(epoch)
            t1 = time()
            test_acc, auc, f1, precision, recall, auprc = self.test_epoch(epoch)
            t2 = time()

            wandb.log({
                "train_loss": train_loss,
                "test_acc": test_acc,
                "train time": t1 - t0,
                "test time": t2 - t1,
                "epoch time": t2 - t0,
                "auc": auc,
                "auprc": auprc,
                "f1": f1,
                "precision": precision,
                "recall": recall,})

            if self.save_model and test_acc > self.best_acc:
                self.best_acc = test_acc
                torch.save(self.network.state_dict(), f"checkpoints/icu_{args.model}_{args.seed}.pth")

if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser("sepsis")
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--model", type=str, choices=["gpt"], default="gpt")
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    # Setup parameters
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.model == "gpt":
        model = CustomGPT

    # Load data
    if args.task == 'hyperkalemia': 
        (train_loader, val_loader, test_loader) = hk_loader(args.batch_size)
    elif args.task == 'hypoglycemia':
        (train_loader, val_loader, test_loader) = hg_loader(args.batch_size)
    elif args.task == 'sepsis':
        (train_loader, val_loader, test_loader) = sepsis_loader(args.batch_size)
    elif args.task == 'decompensation':
        (train_loader, val_loader, test_loader) = decomp_loader(args.batch_size)
    elif args.task == 'mortality':
        (train_loader, val_loader, test_loader) = mortality_loader(args.batch_size)
    elif args.task == 'transfer':
        (train_loader, val_loader, test_loader) = icu_loader(args.batch_size)

    trainer = Trainer(model, train_loader, test_loader, val_loader, args.learning_rate, args.gpu)

    # setup wandb
    config = vars(args)
    run = wandb.init(
        project=f"BUCKETS",
        name=f'{args.task}_{args.model}_{args.seed}',
        config=config)
    print(config)

    # Run
    trainer.train(args.n_epochs)

    run.finish()