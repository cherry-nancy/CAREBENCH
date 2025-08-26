# from hf_ehr.data.tokenization import CLMBRTokenizer
from transformers import PreTrainedTokenizerFast

import torch
import torch.nn.functional as F

import random
from time import time
from tqdm import tqdm

import wandb
from argparse import ArgumentParser

from ehrshot_dataset import ehrshot_loader
from mcmed_dataset import mcmed_loader
from mimiced_dataset import mimiced_loader
from models import CustomGPT, CustomMamba

EPS = 1e-2

class Trainer():
    def __init__(self, model, train_loader, test_loader, learning_rate, step_freq, gpu, tokenizer_file, ignore_seq, save_model=True):
        if gpu >= 0 and torch.cuda.is_available(): device = torch.device("cuda:%d" % gpu)
        else: device = torch.device("cpu")
        self.device = device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.save_model = save_model
        self.best_acc = 0
        self.criterion = F.cross_entropy
        self.ignore_seq = ignore_seq

        # self.tokenizer = CLMBRTokenizer.from_pretrained("StanfordShahLab/mamba-tiny-16384-clmbr")
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)
        self.tokenizer.eos_token = '[EOS]'
        self.tokenizer.sep_token = '[SEP]'
        self.tokenizer.bos_token = '[BOS]'
        self.tokenizer.pad_token = '[PAD]'
        self.tokenizer.cls_token = '[CLS]'
        self.tokenizer.mask_token = '[MASK]'

        self.network = model(vocab_size = len(self.tokenizer)).to(self.device)
        self.step_freq = step_freq
        self.optimizer = torch.optim.AdamW(list(filter(lambda p: p.requires_grad, self.network.parameters())), lr=learning_rate)

    def get_inputs(self, inputs, times):
        max_len = max(len(i) for i in inputs)
        tokens = self.tokenizer(inputs, return_tensors="pt", is_split_into_words=True, padding=True, return_attention_mask=True)
        sequences = tokens["input_ids"].to(self.device)   
        attention_masks = tokens["attention_mask"].to(self.device) 

        times = [F.pad(torch.tensor(r, dtype=torch.float32), (0, max_len - len(r))).round(decimals=2)  for r in times]
        times = torch.stack(times, dim=0).to(self.device)  

        return sequences, times, attention_masks

    def train_epoch(self, epoch):
        train_loss = 0
        total_correct = 0
        num_items = 0
        iter = tqdm(self.train_loader, total=len(self.train_loader))
        self.network.train()
        accumulated_loss = 0

        for i, (events, times) in enumerate(iter):
            input_ids, times, attention_mask = self.get_inputs(events, times)
            output = self.network(input_ids[:, :-1], times[:, :-1], attention_mask[:, :-1])
            output = output[:, self.ignore_seq:].squeeze(-1)
            target = input_ids[:, 1+self.ignore_seq:]
            loss = self.criterion(output.flatten(0, 1), target.flatten())
            
            accumulated_loss += loss
            train_loss += loss.item() 
            
            if (i + 1) % self.step_freq == 0 or i == len(self.train_loader) - 1:
                # Calculate how many batches we actually accumulated
                batches_accumulated = min(self.step_freq, (i % self.step_freq) + 1)
                final_loss = accumulated_loss / batches_accumulated
                final_loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                accumulated_loss = 0
                
            avg_loss = train_loss / (i + 1)
            total_correct += (output.argmax(dim=-1) == target).float().sum()
            num_items += target.numel()
            train_acc = (total_correct / num_items)
            iter.set_description(f"[Train Epoch {epoch}] Loss: {avg_loss:.4f} Acc: {total_correct}/{num_items} {train_acc:.4f}")
        
        return avg_loss

    def test_epoch(self, epoch):
        total_correct = 0
        num_items = 0
        iter = tqdm(self.test_loader, total=len(self.test_loader))
        self.network.eval()
        for i, (events, times) in enumerate(iter):
            with torch.no_grad():
                input_ids, times, attention_mask = self.get_inputs(events, times)

                output = self.network(input_ids[:, :-1], times[:, :-1], attention_mask[:, :-1])
                output = output[:, self.ignore_seq:].squeeze(-1)

                target = input_ids[:, 1+self.ignore_seq:]

                total_correct += (output.argmax(dim=-1) == target).float().sum()
                num_items += target.numel()
                test_acc = (total_correct / num_items)
                iter.set_description(f"[Test Epoch {epoch}] Acc: {total_correct}/{num_items} {test_acc:.4f}")
        return test_acc

    def train(self, n_epochs):
        for epoch in range(1, n_epochs+1):
            t0 = time()
            train_loss = self.train_epoch(epoch)
            t1 = time()
            test_acc = self.test_epoch(epoch)
            t2 = time()

            wandb.log({
                "train_loss": train_loss,
                "test_acc": test_acc,
                "train time": t1 - t0,
                "test time": t2 - t1,
                "epoch time": t2 - t0})
        
            if self.save_model and test_acc > self.best_acc:
                self.best_acc = test_acc
                torch.save(self.network.state_dict(), f"checkpoints/{args.task}-{args.model}.pth")


if __name__ == "__main__":
    # Argument parser
    parser = ArgumentParser("sepsis")
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--step-freq", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--model", type=str, choices=["gpt", "mamba"], default="gpt")
    parser.add_argument("--task", type=str, choices=["ehrshot", "mcmed", "mimic", "mimiced"], default="ehrshot")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--cuda", action="store_true")
    args = parser.parse_args()

    # Setup parameters
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.model == "gpt":
        model = CustomGPT
        CONTEXT = 1024
    elif args.model == "mamba":
        model = CustomMamba
        CONTEXT = 2048

    # Load data
    if args.task == "ehrshot":
        ignore_seq = 3
        (train_loader, test_loader) = ehrshot_loader(args.batch_size, seed=args.seed, context_length=CONTEXT-ignore_seq)
        tokenizer_file = 'data/ehrshot_tokenizer.json'
    elif args.task == "mcmed":
        ignore_seq = 4
        (train_loader, test_loader) = mcmed_loader(args.batch_size, seed=args.seed, context_length=CONTEXT-ignore_seq)
        tokenizer_file = 'data/mcmed_tokenizer.json'
    elif args.task == "mimic":
        ignore_seq = 3
        (train_loader, test_loader) = mimic_loader(args.batch_size, seed=args.seed, context_length=CONTEXT-ignore_seq)
        tokenizer_file = 'data/mimic_tokenizer.json'
    elif args.task == "mimiced":
        ignore_seq = 2
        (train_loader, test_loader) = mimiced_loader(args.batch_size, seed=args.seed, context_length=CONTEXT-ignore_seq)
        tokenizer_file = '/home/mkeoliya/projects/CAREBench/data/mimiced_tokenizer.json'
    print("LOADED")
    trainer = Trainer(model, train_loader, test_loader, args.learning_rate, args.step_freq, args.gpu, tokenizer_file, ignore_seq)

    # setup wandb
    config = vars(args)
    run = wandb.init(
        project=f"pretrain-{args.task}-{args.model}",
        config=config)
    print(config)

    # Run
    trainer.train(args.n_epochs)

    run.finish()