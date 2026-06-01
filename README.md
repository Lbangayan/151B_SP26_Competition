# CSE 151B Spring 2026 — Math Reasoning Competition

End-to-end inference for the competition. Single entry point: `run_inference()`
in [`run_inference.py`](./run_inference.py).

---

## 1. GPU & Generation Time

We ran on **DSMLP** with whichever 24 GB GPU the scheduler assigned. Pipeline
defaults are tuned for 24 GB VRAM.

| GPU | VRAM |
|---|---|
| NVIDIA RTX A30 | 24 GB HBM2 |
| NVIDIA RTX A5000 | 24 GB GDDR6 |
| NVIDIA RTX PRO 6000 Blackwell (b24gb MIG slice) | 24 GB |

**Approximate total generation time on the full private set (943 records):**

| Mode | A30 | A5000 / b24gb |
|---|---|---|
| Full pipeline (olympiad K=8 SC enabled) | ~10–15 h | ~12–18 h |
| Fast mode (`--no-olympiad-sc`) | ~6–10 h | ~8–12 h |

---

## 2. Setup

```bash
# Install pinned stack. vLLM < 0.20 is REQUIRED — newer wheels link against
# CUDA 13 libs (libnvJitLink.so.13) that the standard DSMLP container
# (ghcr.io/ucsd-ets/sp26-cuda128:main, CUDA 12.8) does not provide.
pip install --user "vllm<0.20" bitsandbytes "antlr4-python3-runtime==4.11.1" ipykernel
pip install --user transformers tqdm sympy numpy
```

DSMLP launch (one 24 GB GPU, 8 cores, 32 GB RAM):

```bash
launch-sp26-cuda128.sh -l gpu-class=medium -W CSE151B_SP26_A00 -g 1 -c 8 -m 32
```

The base model `Qwen/Qwen3-4B-Thinking-2507` is auto-downloaded by vLLM on
first run (~8 GB, cached in `~/.cache/huggingface/`). No manual model setup
is needed.

Place the private set at `data/private.jsonl` (download from the Kaggle
competition page; gitignored).

---

## 3. How to Run

### CLI

```bash
# Default — full pipeline
python run_inference.py --data data/private.jsonl --output submission.csv

# Fast mode (~3x speedup, ~3-5pp accuracy cost on olympiad items)
python run_inference.py --data data/private.jsonl --output submission.csv --no-olympiad-sc
```

Other knobs (run `python run_inference.py --help` for the full list):
`--gpu`, `--max-model-len`, `--max-num-seqs`, `--max-num-batched-tokens`,
`--enforce-eager`, `--cache-dir`.

### Programmatic

```python
from run_inference import run_inference

run_inference(
    data_path="data/private.jsonl",
    output_path="submission.csv",
    enable_olympiad_sc=True,   # False for fast mode
)
```

### Output

CSV with columns `id`, `response` (`csv.QUOTE_ALL` so commas in responses
don't break parsing). Ready to upload to Kaggle.

### Crash safety

Cache is written to `results/cache/{prompt_hash}_seed13_{data_stem}.jsonl`
after every batch. Reruns resume from the last completed batch — a kernel
kill costs at most one batch of work. To force regeneration, delete the
cache file.

---

## 4. Notebooks in this Repo

`run_inference.py` is the canonical entry point. The notebooks below are
the development / research artifacts behind it. **Graders only need
`run_inference.py`** — the notebooks are for reading the pipeline
interactively or rerunning specific experiments.

| Notebook | Pipeline | Data | Purpose |
|---|---|---|---|
| `starter_code_cse151b_comp.ipynb` | **v2.9.3 baseline** | `data/public.jsonl` | Eval notebook — runs the baseline on the 100-question tailored subset with scoring + diagnostics. Use this to reproduce our eval numbers. |
| `starter_code_cse151b_comp_submission.ipynb` | **v2.9.3 baseline** | `data/private.jsonl` | **Submission notebook — what generated the shipped CSV.** Identical pipeline to the eval notebook minus the scoring cells, plus a CSV writer at the end. Functionally equivalent to `run_inference.py`. |
| `starter_code_cse151b_comp_sft.ipynb` | **v2.9.2-sft baseline + LoRA adapter** | `data/public.jsonl` | SFT eval — same pipeline as `starter_code_cse151b_comp.ipynb` but with a QLoRA adapter loaded via vLLM's `LoRARequest`. Diverges from baseline cell 17 config (max_model_len=49152, enforce_eager=True) to avoid a known vLLM + LoRA + bnb crash on 24 GB GPUs. Used to evaluate whether the SFT adapter improved over baseline. |
| `starter_code_cse151b_comp_submission_sft.ipynb` | **v2.9.2-sft + LoRA** | `data/private.jsonl` | SFT submission variant. Generates an alternate CSV using the LoRA adapter. Not shipped (final submission used the baseline pipeline). |
| `starter_code_cse151b_comp_submission_sft_bigbudget.ipynb` | **v2.9.2-sft + LoRA, restored olympiad budget** | `data/private.jsonl` | Experimental SFT submission variant. Restores `max_model_len=90112` + `MAX_TOKENS_OLYMPIAD=81920` by halving `max_num_seqs` to 16 — tries to give olympiad items full token budget while keeping LoRA stable. Not shipped. |
| `train_lora.ipynb` | **QLoRA SFT training** | `OpenR1-Math-220k` (HuggingFace) | Trains the LoRA adapter that the SFT notebooks load. Output checkpoint pushed to `daniel930324/qwen3-4b-math-lora` on HF Hub. |

All inference notebooks share the same cell IDs for preprocessing
(`da74ab54`), prompt router (`8bd056bc`), and generation loop (`caa6d179`) —
edits to one should be propagated to the others to maintain the
"identical pipeline, different config" invariant.
