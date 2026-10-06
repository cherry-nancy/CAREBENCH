# MIMIC-IV in-hospital mortality: GPT2, Mamba, GPT2-AR, Qwen

Notebook reconstruction of the CAREBench mortality pipeline (`process_data/mimic-mortality.ipynb`,
`transformers/`, `pretrain/`, `llm/`). All models share one cohort, one patient-level split and one tokenizer.

**Task:** one ICU stay per patient. Inputs are all events from ICU admission up to 12 h before the end of the stay
(death time if the patient died in hospital, otherwise ICU discharge). The label is in-hospital death.

| notebook | what it does |
|---|---|
| `01_prepare_data.ipynb` | cohort + labels, 60/20/20 split, event extraction, per-item decile buckets, tokenizer, LLM text |
| `02_gpt2_mamba.ipynb` | GPT2 or Mamba (`MODEL_NAME`) sequence classifier: event + time embeddings, pooled MLP head |
| `03_gpt2_ar.ipynb` | GPT2-AR: next-event pretraining, then a 3-step autoregressive rollout scored by a linear layer over the vocabulary |
| `04_qwen.ipynb` | zero-shot Qwen3 (vLLM or transformers) on the text timeline, parses `Prediction: Yes/No, Risk: x` |

## Run

1. Download MIMIC-IV v3.1 (`hosp/`, `icu/`) from PhysioNet into `data/mimiciv/3.1/`, or point the `MIMIC_DIR`
   environment variable at it. The `.csv.gz` files can stay compressed.
2. Run `01_prepare_data.ipynb` once. It writes to `data/mimic_mortality/`. Set `SUBSAMPLE_PATIENTS = 3000` first for a quick dry run.
3. Run any of `02`–`04`, with the notebook directory as the working directory. Checkpoints go to `checkpoints/mimic_mortality/`.
   Metrics and test predictions go to `results/mimic_mortality/<run>_{metrics.json,test_predictions.csv}`.

Optional overrides: `MORTALITY_DATA_DIR`, `MORTALITY_CHECKPOINT_DIR`, `MORTALITY_RESULTS_DIR`.

## Requirements

`torch`, `transformers>=4.44`, `tokenizers`, `pandas`, `pyarrow`, `scikit-learn`, `tqdm`. Also:
* Mamba: `mamba-ssm` + `causal-conv1d` (CUDA). Without them HF falls back to a slow sequential scan.
* Qwen: `vllm` (Linux + CUDA). Qwen3-32B needs ~80 GB of GPU memory. `BACKEND = 'hf'` works with plain transformers for smaller models.

MIMIC data must not be sent to hosted LLM APIs ([PhysioNet policy](https://physionet.org/news/post/gpt-responsible-use)).

## Google Colab

The notebooks use the GPU automatically. Mixed precision is bf16 on A100/L4 and fp16 with loss scaling on T4.
Paste this as the **first cell** of a notebook, before `mortality_common` is imported:

```python
from google.colab import drive
drive.mount('/content/drive')
!git clone -q https://github.com/cherry-nancy/CAREBENCH.git /content/CAREBENCH
%cd /content/CAREBENCH/mimic_mortality

import os
os.environ['MIMIC_DIR'] = '/content/drive/MyDrive/mimiciv/3.1'                       # only needed by 01
os.environ['MORTALITY_DATA_DIR'] = '/content/drive/MyDrive/mimic_mortality'          # output of 01
os.environ['MORTALITY_CHECKPOINT_DIR'] = '/content/drive/MyDrive/mimic_mortality/checkpoints'
os.environ['MORTALITY_RESULTS_DIR'] = '/content/drive/MyDrive/mimic_mortality/results'
# Mamba: !pip install -q mamba-ssm causal-conv1d      Qwen with vLLM: !pip install -q vllm  (then restart the runtime)
```

* Run `01_prepare_data.ipynb` on your own machine, which has more RAM than a standard Colab runtime. Then upload
  `tokenizer.json`, `cohort.csv`, `{train,val,test}.parquet` and `llm_test.parquet` to the Drive folder. `events_raw/` and `sequences/` aren't needed.
* Keep checkpoints on Drive because Colab runtimes disconnect. GPT2-AR reuses its stage-1 checkpoint if it exists.
* GPU guide: GPT2 / GPT2-AR fit a T4 at batch size 8. Mamba needs the CUDA kernels above. For Qwen, Qwen3-4B fits a
  T4/L4, Qwen3-14B needs an A100 40GB, and Qwen3-32B needs 80 GB.
* Executed notebooks contain patient-level MIMIC text in their outputs (the example timeline in 01, the prompt in 04).
  Clear the outputs before committing them to GitHub or sharing them.

## Differences from the original scripts

* Model selection uses validation AUROC. The original saved the best *test* epoch.
* GPT2-AR reads next-event logits at each sequence's last real token. The original read `[:, -1]`, which is padding for shorter sequences in a batch.
* The new-vocabulary token embeddings are trained. The original froze them.
* Mamba fine-tunes its last 2 blocks instead of LoRA. With the CUDA kernels, LoRA on `x_proj`/`dt_proj`/`out_proj` is bypassed in training.
* Qwen prompts that are too long drop the oldest measurements instead of cutting off the end of the prompt.
* Deaths more than 24 h after ICU discharge are excluded (`MAX_DEATH_AFTER_ICU_HOURS`). Their windows contain only ward labs, which is a label shortcut.
