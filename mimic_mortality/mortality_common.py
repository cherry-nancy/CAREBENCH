"""Shared helpers for the MIMIC-IV in-hospital mortality notebooks.

Task: for each ICU stay, use every event recorded up to 12 hours before the end
of the stay (death time for patients who die in hospital, ICU discharge
otherwise) and predict in-hospital mortality.

Produced by 01_prepare_data.ipynb, consumed by the model notebooks:
    data/mimic_mortality/cohort.csv              one row per ICU stay (label, split, demographics)
    data/mimic_mortality/tokenizer.json          word-level tokenizer over bucketed events
    data/mimic_mortality/{train,val,test}.parquet  token-id / time sequences per stay
    data/mimic_mortality/llm_test.parquet        text serialisation of the test stays (Qwen)
"""
import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_fscore_support, accuracy_score, confusion_matrix,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
MIMIC_DIR = Path(os.environ.get('MIMIC_DIR', REPO_ROOT / 'data' / 'mimiciv' / '3.1'))
DATA_DIR = Path(os.environ.get('MORTALITY_DATA_DIR', REPO_ROOT / 'data' / 'mimic_mortality'))
CHECKPOINT_DIR = Path(os.environ.get('MORTALITY_CHECKPOINT_DIR', REPO_ROOT / 'checkpoints' / 'mimic_mortality'))
RESULTS_DIR = Path(os.environ.get('MORTALITY_RESULTS_DIR', REPO_ROOT / 'results' / 'mimic_mortality'))

PREDICTION_GAP_HOURS = 12.0
SPECIAL_TOKENS = ['[UNK]', '[BOS]', '[EOS]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
# Event times are hours since ICU admission; divide before feeding the time embedding
TIME_SCALE = 24.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(gpu=0):
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu}')
    return torch.device('cpu')


def load_tokenizer(path=None):
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(path or DATA_DIR / 'tokenizer.json'))
    tokenizer.unk_token = '[UNK]'
    tokenizer.bos_token = '[BOS]'
    tokenizer.eos_token = '[EOS]'
    tokenizer.sep_token = '[SEP]'
    tokenizer.pad_token = '[PAD]'
    tokenizer.cls_token = '[CLS]'
    tokenizer.mask_token = '[MASK]'
    return tokenizer


# ----------------------------------------------------------------------------- data

class MortalityDataset(Dataset):
    """One sample per ICU stay: demographic token ids, event token ids, event times, label."""

    def __init__(self, df, max_events):
        self.subject_ids = df['subject_id'].to_numpy()
        self.labels = df['label'].to_numpy().astype(np.float32)
        self.demo_ids = [np.asarray(x, dtype=np.int64) for x in df['demo_ids']]
        # keep the most recent events, i.e. the ones closest to the prediction time
        self.input_ids = [np.asarray(x, dtype=np.int64)[-max_events:] for x in df['input_ids']]
        self.times = [np.asarray(x, dtype=np.float32)[-max_events:] for x in df['times']]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'subject_id': int(self.subject_ids[idx]),
            'demo_ids': self.demo_ids[idx],
            'input_ids': self.input_ids[idx],
            'times': self.times[idx],
            'label': self.labels[idx],
        }

    @staticmethod
    def collate_fn(batch):
        return {
            'subject_id': [b['subject_id'] for b in batch],
            'demo_ids': [b['demo_ids'] for b in batch],
            'input_ids': [b['input_ids'] for b in batch],
            'times': [b['times'] for b in batch],
            'label': torch.tensor(np.array([b['label'] for b in batch])),
        }


def load_split(split, data_dir=None):
    return pd.read_parquet(Path(data_dir or DATA_DIR) / f'{split}.parquet')


def make_loaders(batch_size, max_events, data_dir=None, eval_batch_size=None, num_workers=0):
    loaders = {}
    for split in ['train', 'val', 'test']:
        dataset = MortalityDataset(load_split(split, data_dir), max_events)
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size if split == 'train' else (eval_batch_size or batch_size),
            shuffle=(split == 'train'),
            collate_fn=MortalityDataset.collate_fn,
            num_workers=num_workers,
        )
        labels = dataset.labels
        print(f'{split:>5}: {len(dataset):>6} stays, {int(labels.sum()):>5} deaths ({labels.mean():.2%})')
    return loaders


def encode_batch(batch, pad_id, device, time_scale=TIME_SCALE):
    """Right-padded [demographics + events] tensors. Demographic tokens get time 0."""
    seqs = [np.concatenate([d, e]) for d, e in zip(batch['demo_ids'], batch['input_ids'])]
    max_len = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    times = torch.zeros((len(seqs), max_len), dtype=torch.float32)
    attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
    for i, (seq, demo, t) in enumerate(zip(seqs, batch['demo_ids'], batch['times'])):
        input_ids[i, :len(seq)] = torch.from_numpy(seq)
        times[i, len(demo):len(seq)] = torch.from_numpy(t) / time_scale
        attention_mask[i, :len(seq)] = 1
    return input_ids.to(device), times.to(device), attention_mask.to(device)


def pos_weight_from_loader(loader):
    labels = loader.dataset.labels
    return float((labels == 0).sum() / max((labels == 1).sum(), 1))


# ----------------------------------------------------------------------------- models

def pool_hidden_states(strategy, hidden_states, attention_mask=None):
    if attention_mask is None:
        attention_mask = torch.ones(hidden_states.shape[:2], device=hidden_states.device)
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    if strategy == 'mean':
        return (hidden_states * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
    if strategy == 'max':
        return (hidden_states + (1 - mask) * -1e4).max(dim=1).values
    if strategy == 'first':
        return hidden_states[:, 0]
    if strategy == 'last':
        last_idx = attention_mask.sum(1) - 1
        return hidden_states[torch.arange(hidden_states.size(0), device=hidden_states.device), last_idx]
    raise ValueError(f'Unsupported pooling strategy: {strategy}')


def count_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f'trainable params: {trainable:,} / {total:,} ({trainable / total:.2%})')


# ----------------------------------------------------------------------------- metrics

def binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    both_classes = len(np.unique(y_true)) > 1
    return {
        'auroc': roc_auc_score(y_true, y_prob) if both_classes else float('nan'),
        'auprc': average_precision_score(y_true, y_prob) if both_classes else float('nan'),
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'accuracy': accuracy_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
        'n': int(len(y_true)),
        'n_pos': int(y_true.sum()),
    }


def format_metrics(m):
    return ' '.join(f'{k}={m[k]:.4f}' for k in ['auroc', 'auprc', 'f1', 'precision', 'recall', 'accuracy'])


# ----------------------------------------------------------------------------- training

def autocast_context(device, enabled=True):
    if enabled and device.type == 'cuda':
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast('cuda', dtype=dtype)
    return torch.autocast(device.type, enabled=False)


@torch.no_grad()
def predict(model, predict_fn, loader, device, use_amp=True):
    """Returns a DataFrame with subject_id, label and predicted probability."""
    model.eval()
    rows = {'subject_id': [], 'label': [], 'prob': []}
    for batch in tqdm(loader, desc='eval', leave=False):
        with autocast_context(device, use_amp):
            logits = predict_fn(model, batch, train=False)
        rows['subject_id'] += batch['subject_id']
        rows['label'] += batch['label'].tolist()
        rows['prob'] += torch.sigmoid(logits.float()).cpu().tolist()
    return pd.DataFrame(rows)


def train_binary_classifier(
    model, predict_fn, loaders, device, ckpt_path, *,
    learning_rate=5e-5, weight_decay=0.01, n_epochs=10, patience=3, grad_accum=1,
    max_grad_norm=1.0, pos_weight=None, use_amp=True, extra_params=(),
):
    """Generic loop: BCE on predict_fn logits, model selection on validation AUROC,
    final evaluation of the best checkpoint on the test split.

    predict_fn(model, batch, train) -> logits of shape [batch_size]
    """
    ckpt_path = Path(ckpt_path)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    params = [p for p in model.parameters() if p.requires_grad] + list(extra_params)
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    if pos_weight is None:
        pos_weight = pos_weight_from_loader(loaders['train'])
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    print(f'pos_weight={pos_weight:.2f}')

    history, best_auroc, bad_epochs = [], -1.0, 0
    for epoch in range(1, n_epochs + 1):
        model.train()
        running_loss = 0.0
        optimizer.zero_grad()
        progress = tqdm(loaders['train'], desc=f'epoch {epoch}')
        for step, batch in enumerate(progress):
            with autocast_context(device, use_amp):
                logits = predict_fn(model, batch, train=True)
            loss = criterion(logits.float(), batch['label'].to(device))
            (loss / grad_accum).backward()
            if (step + 1) % grad_accum == 0 or step == len(loaders['train']) - 1:
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
            running_loss += loss.item()
            progress.set_postfix(loss=f'{running_loss / (step + 1):.4f}')

        val_pred = predict(model, predict_fn, loaders['val'], device, use_amp)
        val_metrics = binary_metrics(val_pred['label'], val_pred['prob'])
        history.append({'epoch': epoch, 'train_loss': running_loss / len(loaders['train']), **{f'val_{k}': v for k, v in val_metrics.items() if k != 'confusion_matrix'}})
        print(f'[epoch {epoch}] train_loss={history[-1]["train_loss"]:.4f} | val {format_metrics(val_metrics)}')

        if val_metrics['auroc'] > best_auroc:
            best_auroc, bad_epochs = val_metrics['auroc'], 0
            torch.save(model.state_dict(), ckpt_path)
            print(f'  saved best checkpoint -> {ckpt_path}')
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f'early stopping after {epoch} epochs')
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    test_pred = predict(model, predict_fn, loaders['test'], device, use_amp)
    test_metrics = binary_metrics(test_pred['label'], test_pred['prob'])
    print(f'[test] {format_metrics(test_metrics)}')
    print('confusion matrix [[TN, FP], [FN, TP]]:', test_metrics['confusion_matrix'])
    return pd.DataFrame(history), test_metrics, test_pred


def save_results(name, test_metrics, test_pred, history=None, config=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    test_pred.to_csv(RESULTS_DIR / f'{name}_test_predictions.csv', index=False)
    if history is not None:
        history.to_csv(RESULTS_DIR / f'{name}_history.csv', index=False)
    with open(RESULTS_DIR / f'{name}_metrics.json', 'w') as f:
        json.dump({'test': test_metrics, 'config': config or {}}, f, indent=2, default=str)
    print(f'results written to {RESULTS_DIR}')
