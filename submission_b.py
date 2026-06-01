# %% [markdown]
# # CSE 151B Competition — Starter Notebook
# 
# Welcome to the **CSE 151B Spring 2026 Math Reasoning Competition**!  
# This notebook walks you through the full pipeline end-to-end:
# 
# 1. Setting up the Python environment with `uv`
# 2. Loading the competition dataset
# 3. Running inference with **Qwen3-4B-Thinking** via vLLM (INT8 quantized)
# 4. Scoring responses against ground-truth answers
# 5. Saving results to JSONL for submission
# 
# The public dataset (`public.jsonl`) contains questions **with** answers so you can measure accuracy locally.  
# The private test set used for the leaderboard does **not** include answers — for that, skip evaluation and submit the raw responses.

# %% [markdown]
# ## 1. Environment Setup
# 
# We use [`uv`](https://github.com/astral-sh/uv) for fast, reproducible package management.
# 
# The steps below:
# 1. Install `uv` into `~/.local/bin`
# 2. Create a virtual environment at `.venv/`
# 3. Install all required packages (This might take a while)
# 
# > **After running this cell, restart the kernel** so that the newly installed packages (especially `vllm` and `transformers`) are picked up by the current Python session.

# %% [markdown]
# ### Comment Out the cell below after first installation.

# %%
# # Install uv
# !wget -qO- https://astral.sh/uv/install.sh | sh

# # Create a virtual environment
# !uv venv .venv --seed

# # Install dependencies — this is fast thanks to uv's parallel resolver
# !.venv/bin/python -m pip install sympy numpy transformers vllm tqdm bitsandbytes antlr4-python3-runtime==4.11.1 ipykernel jupyter

# # Install Jupyter Kernel
# !.venv/bin/python -m ipykernel install --user --name cse151b --display-name "Python (cse151b)"

# print("Done. Restart the kernel before proceeding.")
# print("Selection process: on top right, click on current kernel '(ususally named python)' -> 'select another kernel' -> 'Jupyter Kernel' -> 'Python (cse151b)'.")

# %% [markdown]
# ### Run the cell below every time to activate the installed environment. 

# %%
# activate venv after installation. This needs to be run everytime.
# (notebook-only shell magic removed — run this script via `uv run python` or `.venv/bin/python`)
# !source ./.venv/bin/activate

# %% [markdown]
# ## 2. Imports & Configuration
# 
# All key settings are collected in one place.  
# - `DATA_PATH` — public dataset with ground-truth answers (use this to measure accuracy)
# - `OUTPUT_PATH` — where per-question results will be written
# - `GPU_ID` — which GPU to use (update if your machine has a different device index)
# - `MAX_TOKENS` — maximum tokens the model may generate per response

# %%
import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--gpu",        default="0",                                 help="CUDA_VISIBLE_DEVICES")
parser.add_argument("--data",       default="data/private.jsonl",                help="Path to input JSONL")
parser.add_argument("--output",     default="results/submission_v293_partB.csv", help="Path to output CSV")
parser.add_argument("--max-tokens", type=int, default=32768,                     help="Max tokens per (non-olympiad) response")
parser.add_argument("--limit",      type=int, default=None,                      help="Only run the first N questions of the partition (smoke test)")
parser.add_argument("--id-start",   type=int, default=707,                       help="Partition start id (inclusive). partA=471, partB=707")
parser.add_argument("--id-end",     type=int, default=943,                       help="Partition end id (exclusive). partA=707, partB=943")
parser.add_argument("--part-tag",   default="partB",                             help="Partition tag for the cache filename, e.g. partA / partB")
args = parser.parse_args()

# ── Environment (MUST be set before importing vllm) ────────────────────────────
os.environ["PYTORCH_ALLOC_CONF"]          = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"]     = "expandable_segments:True"
os.environ["CUDA_VISIBLE_DEVICES"]        = args.gpu
# DSMLP containers lack the CUDA toolkit (nvcc), so vLLM's JIT-compiled kernels
# (flashinfer sampler, DeepGEMM) fail to build. Force the native PyTorch fallbacks.
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_USE_DEEP_GEMM"]          = "0"

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tqdm import tqdm

MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"
GPU_ID      = args.gpu
DATA_PATH   = args.data
OUTPUT_PATH = args.output
MAX_TOKENS  = args.max_tokens
SEED        = 13
EVAL_N_MCQ  = 50
EVAL_N_FREE = 50
EVAL_PRECISION = 1e-2
SUBSET_PATH = "results/eval_subset.json"
LIMIT       = args.limit


# %% [markdown]
# ## 3. Load the Dataset
# 
# The dataset is stored as newline-delimited JSON (`.jsonl`). Each line is one question with the following fields:
# 
# | Field | Description |
# |---|---|
# | `id` | Unique question identifier |
# | `question` | Problem statement |
# | `options` | List of answer choices — present for **MCQ**, absent for **free-form** |
# | `answer` | Ground-truth answer (letter for MCQ, value/list for free-form) |

# %%
data = [json.loads(line) for line in open(DATA_PATH)]

n_mcq  = sum(bool(d.get("options")) for d in data)
n_free = sum(not d.get("options")   for d in data)
print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

# Preview one MCQ and one free-form item
mcq_sample  = next(d for d in data if d.get("options"))
free_sample = next(d for d in data if not d.get("options"))

print("\n── MCQ sample ──")
print(json.dumps(mcq_sample, indent=2))
print("\n── Free-form sample ──")
print(json.dumps(free_sample, indent=2))

# %% [markdown]
# ## 4.1 Preprocessing
# 
# Before sending a question to the model, we apply deterministic cleanup:
# 
# 1. **LaTeX repair** — `repair_latex(text)` re-attaches missing backslashes to
#    known math commands. The dataset has systematic LaTeX corruption (e.g.
#    `int_{-infty}^{+infty} frac{a}{s^2+a^2}` should be
#    `\int_{-\infty}^{+\infty} \frac{a}{s^2+a^2}`). Even strong external
#    reasoners cannot recover from this on free-form questions; on MCQ they
#    can back-infer from options at the cost of accuracy.
# 2. **Question preprocessing** — `preprocess_question(item)` wraps
#    `repair_latex` and applies it to the question text and (if present) each
#    option string. Returns a new dict; does not mutate the input.
# 
# Both functions are pure (no model dependency) and idempotent.
# 

# %%
_UNICODE_MATH_MAP = {
    "∫": r"\int", "∬": r"\iint", "∭": r"\iiint", "∮": r"\oint",
    "∑": r"\sum", "∏": r"\prod",
    "√": r"\sqrt", "∞": r"\infty",
    "∂": r"\partial", "∇": r"\nabla",
    "α": r"\alpha", "β": r"\beta", "γ": r"\gamma", "δ": r"\delta",
    "ε": r"\epsilon", "ζ": r"\zeta", "η": r"\eta", "θ": r"\theta",
    "ι": r"\iota", "κ": r"\kappa", "λ": r"\lambda", "μ": r"\mu",
    "ν": r"\nu", "ξ": r"\xi", "π": r"\pi", "ρ": r"\rho",
    "σ": r"\sigma", "τ": r"\tau", "υ": r"\upsilon", "φ": r"\phi",
    "χ": r"\chi", "ψ": r"\psi", "ω": r"\omega",
    "Γ": r"\Gamma", "Δ": r"\Delta", "Θ": r"\Theta", "Λ": r"\Lambda",
    "Ξ": r"\Xi", "Π": r"\Pi", "Σ": r"\Sigma", "Υ": r"\Upsilon",
    "Φ": r"\Phi", "Ψ": r"\Psi", "Ω": r"\Omega",
    "≥": r"\ge", "≤": r"\le", "≠": r"\ne", "≈": r"\approx",
    "≡": r"\equiv", "±": r"\pm", "∓": r"\mp",
    "⋅": r"\cdot", "×": r"\times", "÷": r"\div",
    "∈": r"\in", "∉": r"\notin", "⊂": r"\subset", "⊆": r"\subseteq",
    "⊃": r"\supset", "⊇": r"\supseteq", "∅": r"\emptyset",
    "∪": r"\cup", "∩": r"\cap",
    "∀": r"\forall", "∃": r"\exists", "¬": r"\neg",
    "∧": r"\land", "∨": r"\lor",
    "→": r"\to", "←": r"\leftarrow", "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow", "⇐": r"\Leftarrow", "⇔": r"\Leftrightarrow",
    "↦": r"\mapsto",
    "ℝ": r"\mathbb{R}", "ℤ": r"\mathbb{Z}", "ℚ": r"\mathbb{Q}",
    "ℕ": r"\mathbb{N}", "ℂ": r"\mathbb{C}",
    "°": r"^{\circ}",
}


def normalize_unicode_math(text: str) -> str:
    """Convert Unicode math symbols (∫ ∑ √ π ≥ …) to their LaTeX commands."""
    if not text:
        return text
    for sym, latex in _UNICODE_MATH_MAP.items():
        if sym not in text:
            continue
        # Insert trailing space when followed by another letter so '\intx' doesn't form.
        # Lambda is required because re.sub treats backslash as a template escape.
        if latex.startswith("\\") and latex[-1].isalpha():
            replacement = latex + " "
            text = re.sub(re.escape(sym) + r"(?=[A-Za-z])", lambda _: replacement, text)
        text = text.replace(sym, latex)
    return text


LATEX_CMDS = [
    "int", "iint", "iiint", "oint", "sum", "prod", "lim",
    "infty", "partial",
    "frac", "dfrac", "tfrac", "sqrt", "binom",
    "sin", "cos", "tan", "cot", "sec", "csc",
    "arcsin", "arccos", "arctan",
    "sinh", "cosh", "tanh",
    "log", "ln", "exp",
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta",
    "theta", "iota", "kappa", "lambda", "mu", "nu",
    "xi", "pi", "rho", "sigma", "tau", "phi", "chi", "psi", "omega",
    "Gamma", "Delta", "Theta", "Lambda", "Pi", "Sigma", "Phi", "Psi", "Omega",
    "pm", "mp", "times", "cdot", "leq", "geq", "neq",
    "approx", "equiv", "sim", "to", "in", "notin",
    "subset", "supset", "cup", "cap",
    "mathbb", "mathrm", "mathbf", "mathcal",
    "left", "right", "text",
]

# Math-context gate: only repair when followed by `{ } _ ^ ( )` so English words
# ("the tan of x", "to compute") aren't mangled.
_LATEX_MATH_CONTEXT = r"[{}_^()]"
_LATEX_PATTERNS = [
    (re.compile(rf"(?<!\\)\b{cmd}(?={_LATEX_MATH_CONTEXT})"), rf"\\{cmd}")
    for cmd in LATEX_CMDS
]


def repair_latex(text: str) -> str:
    """Re-attach missing backslashes to known LaTeX commands in math context."""
    for pattern, repl in _LATEX_PATTERNS:
        text = pattern.sub(repl, text)
    return text


def preprocess_question(item: dict) -> dict:
    """Return a copy of `item` with question + options run through unicode + LaTeX repair."""
    out = dict(item)
    q = normalize_unicode_math(item["question"])
    out["question"] = repair_latex(q)
    if item.get("options"):
        out["options"] = [repair_latex(normalize_unicode_math(opt)) for opt in item["options"]]
    return out


# %%
# Verify on real dataset samples
print("LaTeX repair on Q1 (MCQ with mangled formula)")
q1 = data[1]
print(f"  before: {q1['question']}")
print(f"  after : {repair_latex(q1['question'])}")
print(f"  options[0:3] before: {q1['options'][:3]}")
print(f"  options[0:3] after : {[repair_latex(o) for o in q1['options'][:3]]}")

print("\nFalse-positive check (English words that collide with command names)")
for s in ["the tan of x", "in this case", "to compute", "consider tangent", "Bob is tan"]:
    out = repair_latex(s)
    flag = " <= CHANGED" if out != s else ""
    print(f"  {s!r:35} -> {out!r}{flag}")

print("\nIdempotence check")
sample = "int_{-infty}^{+infty} frac{a}{s^2+a^2}"
once  = repair_latex(sample)
twice = repair_latex(once)
print(f"  repair_latex once == twice: {once == twice}")
print(f"  result: {once}")

# %% [markdown]
# ## 4.2 Prompt Construction
# 
# We define the system prompts and the prompt-building functions:
# 
# - `SYSTEM_PROMPT_MATH` — solve-and-box instructions for free-form questions.
# - `SYSTEM_PROMPT_MCQ` — letter-only selection for multiple-choice questions.
# - `MCQ_STAGE2_INSTRUCTION` — short reconciliation instruction for the two-stage
#   MCQ flow (used only when stage-1's answer doesn't match any option).
# - `build_prompt(question, options)` — **baseline** single-call prompt
#   construction. Returns `(system, user)` where `user` includes the options
#   inline for MCQ. Kept unchanged for direct comparison against v2.
# - `select_prompt(item)` — **v2** conditional prompt routing. For MCQ, returns
#   the *stage-1* prompts (question only, no options shown) so the model must
#   derive its own answer before being anchored by the option set. For
#   free-form, routes by question complexity (multi-part / long / short).
# - `verify_against_options(response, options)` — uses `Judger.auto_judge` to
#   check whether the stage-1 answer matches any option. Returns the matching
#   letter or `None`.
# - `build_mcq_stage2_messages(item, stage1_response)` — builds the multi-turn
#   chat for the stage-2 reconciliation call, invoked only when
#   `verify_against_options` returned `None`.
# 
# `build_prompt` is the **baseline pipeline**; `select_prompt` plus the
# two-stage MCQ helpers are the **v2 pipeline**. Both coexist so we can A/B
# test cleanly.
# 

# %%
# Free-form / single-answer default. The bare "reason step by step" instruction
# is the model card's recommended baseline for Qwen3-4B-Thinking-2507.
SYSTEM_PROMPT_MATH = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# MCQ stage-1 reuses the math prompt; the MCQ-specific letter contract is
# appended at the user-prompt tail by _append_mcq_letter_instruction.
SYSTEM_PROMPT_MCQ = SYSTEM_PROMPT_MATH

# Stage-2 reconcile: only sent when lenient-verify failed to map stage-1's
# free-form answer onto any option (transcription noise in the question text).
MCQ_STAGE2_INSTRUCTION = (
    "Your previous answer does not match any of the options shown above. The "
    "question text may have minor transcription errors; pick the option that "
    "makes the problem mathematically sensible. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. "
    "\\boxed{C}. Do not include the option text or any other content inside "
    "the box."
)

# Long free-form: restate-then-enumerate scaffolding keeps the model from
# losing track of given conditions on >500-char problems.
SYSTEM_PROMPT_LONG = (
    "Before solving, restate the problem in your own words and list every given "
    "condition and the quantity to find. Then solve.\n\n"
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# Multipart contract: all sub-answers in ONE comma-separated \boxed{} so the
# judger's last-contiguous-group extraction picks them up as a single unit.
SYSTEM_PROMPT_MULTIPART = (
    "This problem contains multiple sub-answers marked by [ANS] placeholders "
    "in the question. Identify each sub-question, then solve them in the order "
    "they appear.\n\n"
    "After completing all reasoning, output every answer inside ONE final "
    "\\boxed{}, comma-separated, in the same order as the [ANS] slots — for "
    "example \\boxed{a, b, c} for three sub-answers. Do NOT split answers across "
    "multiple \\boxed{} expressions; the grader only reads one contiguous boxed "
    "group, so intermediate \\boxed{} blocks during your derivation will "
    "corrupt the extraction.\n\n"
    "If any sub-question shows inline lettered options (A. ... B. ... C. ...), "
    "use the corresponding letter as that slot's answer inside the same single "
    "box, e.g. \\boxed{6, C, 12}."
)


_EMBEDDED_MCQ_RE = re.compile(
    r"\[ANS\][\s\.\$\"',]{0,40}"
    r"A[\.\)][\s]+[^\n]{3,300}\s+"
    r"B[\.\)][\s]+[^\n]{3,300}",
    re.DOTALL,
)

_INLINE_OPTS_RE = re.compile(
    r"([A-J])[\.\)]\s+(.+?)(?=\s+[A-J][\.\)]\s+|\s*$)",
    re.DOTALL,
)


def detect_embedded_mcq(question: str) -> Optional[list]:
    """Return inline option texts (A/B/C/... order) for free-form items that
    actually carry MCQ choices in their body, or None.

    Caller MUST gate on n_ans <= 1: multipart questions use A./B./C. as subpart
    labels that structurally look identical to MCQ options.
    """
    if not _EMBEDDED_MCQ_RE.search(question):
        return None
    m_after = re.search(r"\[ANS\]([\s\S]+)$", question)
    if not m_after:
        return None
    after_ans = m_after.group(1)
    opt_matches = _INLINE_OPTS_RE.findall(after_ans)
    if len(opt_matches) < 2:
        return None
    letters = [m[0] for m in opt_matches]
    expected = [chr(65 + i) for i in range(len(letters))]
    if letters != expected:
        return None
    return [text.strip() for _, text in opt_matches]


# Wordlist is intentionally narrow: 'simplify' alone is ~25% false-positive,
# only the more specific 'in simplest form' / 'simplified form' is matched.
_FORMAT_CONSTRAINT_PATTERNS = [
    (re.compile(r"\b(?:round(?:ed)? to|correct to|to)\s+(\w+)\s+decimal\s+places?\b", re.I),
        "round to {1} decimal places"),
    (re.compile(r"\b(?:round(?:ed)? to|correct to|with|to)\s+(\w+)\s+(?:significant\s+(?:figures?|digits?)|sig\.?\s*figs?)\b", re.I),
        "give {1} significant figures"),
    (re.compile(r"\bnearest\s+(tenth|hundredth|thousandth|ten-thousandth|hundred-thousandth)\b", re.I),
        "round to the nearest {1}"),
    (re.compile(r"\bnearest\s+(?:integer|whole(?:\s+number)?)\b", re.I),
        "round to the nearest integer"),
    (re.compile(r"\bnearest\s+(cent|dollar)\b", re.I),
        "round to the nearest {1}"),
    (re.compile(r"\bexact\s+(?:value|answer|form)\b", re.I),
        "give the exact value (do not round)"),
    # Negative lookahead skips "do not round intermediate steps" (output IS rounded).
    (re.compile(r"\bdo\s+not\s+round(?!\s+(?:in\s+)?intermediate|\s+(?:in\s+)?(?:any\s+)?calculations?|\s+(?:any\s+)?intermediate)", re.I),
        "do not round"),
    (re.compile(r"\bno\s+rounding\b", re.I),
        "do not round"),
    (re.compile(r"\bdo\s+not\s+simplify\b", re.I),
        "do not simplify"),
    # Negative lookahead excludes 'degrees Celsius/Fahrenheit/Rankine' (temperature scale, not format).
    (re.compile(r"\bin\s+degrees\b(?!\s*(?:Celsius|Fahrenheit|Rankine|C\b|F\b))", re.I),
        "express the answer in degrees"),
    (re.compile(r"\bin\s+radians\b", re.I),
        "express the answer in radians"),
    (re.compile(r"\bas\s+a\s+fraction\b", re.I),
        "express the answer as a fraction"),
    (re.compile(r"\bas\s+a\s+decimal\b", re.I),
        "express the answer as a decimal"),
    (re.compile(r"\bin\s+(?:lowest|simplest)\s+(?:terms|form)\b", re.I),
        "express the answer in simplest form"),
    (re.compile(r"\bsimplified\s+form\b", re.I),
        "express the answer in simplified form"),
    (re.compile(r"\bin\s+scientific\s+notation\b", re.I),
        "express the answer in scientific notation"),
    (re.compile(r"\bin\s+interval\s+notation\b", re.I),
        "express the answer in interval notation"),
    (re.compile(r"\bas\s+a\s+percent(?:age)?\b", re.I),
        "express the answer as a percent"),
    (re.compile(r"\bin\s+(vertex|factored|standard)\s+form\b", re.I),
        "express the answer in {1} form"),
]


def detect_format_constraints(question: str) -> list:
    """Extract human-readable format hints (e.g. 'round to 2 decimal places') from question text."""
    seen = []
    for pat, template in _FORMAT_CONSTRAINT_PATTERNS:
        m = pat.search(question)
        if not m:
            continue
        hint = template.format(*([None] + list(m.groups()))) if "{1}" in template else template
        if hint not in seen:
            seen.append(hint)
    return seen


def _append_format_constraints(user_prompt: str, question_text: str) -> str:
    """Echo any detected format hints at the user-prompt tail under a 'Required output format:' label."""
    constraints = detect_format_constraints(question_text)
    if not constraints:
        return user_prompt
    return user_prompt + "\n\nRequired output format: " + "; ".join(constraints) + "."


def _append_multipart_slotcount(user_prompt: str, n_ans: int) -> str:
    """Tell the model exactly how many comma-separated values to put in the single multipart \\boxed{}."""
    return user_prompt + (
        f"\n\nThis problem has {n_ans} sub-answers. Provide exactly {n_ans} "
        f"comma-separated values inside ONE final \\boxed{{}} block. Do not "
        f"emit any other \\boxed{{}} during your reasoning."
    )


def _append_mcq_letter_instruction(user_prompt: str) -> str:
    """Append the MCQ letter-only contract to the user prompt (tail placement, not system)."""
    return user_prompt + (
        "\n\nOutput ONLY the letter of your chosen option inside \\boxed{}, "
        "e.g. \\boxed{C}. Do not include the option text or any other "
        "content inside the box."
    )


_PRECISION_INSTRUCTION_RE = re.compile(
    r"\b("
    r"round(?:ed)?\s+(?:off\s+|up\s+|down\s+)?to\b"
    r"|nearest\s+(?:tenth|hundredth|thousandth|ten-thousandth|hundred-thousandth"
    r"|integer|whole(?:\s+number)?|cent|dollar|degree|minute|second|year|day)"
    r"|to\s+\w+\s+decimal\s+places?"
    r"|with\s+\w+\s+decimal\s+places?"
    r"|correct\s+to\s+\w+\s+decimal"
    r"|accurate\s+to\s+\w+\s+decimal"
    r"|\w+\s+(?:significant\s+(?:figures?|digits?)|sig\.?\s*figs?)"
    r"|exact\s+(?:value|answer|form)"
    r"|do\s+not\s+round"
    r"|no\s+rounding"
    r")", re.IGNORECASE,
)


def has_precision_instruction(question: str) -> bool:
    """True iff the question text explicitly specifies a rounding/precision target."""
    return bool(_PRECISION_INSTRUCTION_RE.search(question or ""))


# Default: ask for full precision in the box to avoid silent self-rounding.
PRECISION_HINT_DEFAULT = (
    "\n\nWhen giving a numerical answer inside \\boxed{}, use the FULL-PRECISION "
    "value (at least 10 significant figures for floats, or exact fractions / "
    "symbolic forms when convenient). Do not pre-round to 2-3 decimal places."
)

# Rounding-instruction branch: keep reasoning at full precision but emit only the rounded value.
PRECISION_HINT_ROUND = (
    "\n\nThe question specifies a rounding/precision target. In your reasoning, "
    "compute the full-precision value first; in your final \\boxed{} answer, "
    "give ONLY the rounded/truncated value as the question requests."
)


def _append_precision_hint(user_prompt: str, question: str) -> str:
    """Pick the precision hint variant based on whether the question requests rounding."""
    if has_precision_instruction(question):
        return user_prompt + PRECISION_HINT_ROUND
    return user_prompt + PRECISION_HINT_DEFAULT


_OPTIONS_REFERENCED = re.compile(
    r"\b(?:"
    r"which (?:of the )?(?:following|options?|statements?|choices?)"
    r"|(?:all|none) of the above"
    r"|select all"
    r"|what (?:is|are) (?:correct|right|true)"
    r"|determine the (?:corresponding )?output"
    r")\b",
    re.IGNORECASE,
)
_OPT_SELF_REFERENCE = re.compile(r"\b(?:all|none) of the above\b", re.IGNORECASE)


def mcq_needs_options(question: str, options: list) -> bool:
    """True iff the question text or options self-reference, requiring options to be shown to the model."""
    if _OPTIONS_REFERENCED.search(question):
        return True
    return any(_OPT_SELF_REFERENCE.search(opt) for opt in options)


def build_prompt(question: str, options: Optional[list]) -> tuple:
    """Baseline single-call prompt construction (kept for A/B comparison)."""
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


def select_prompt(item: dict) -> tuple:
    """Route an item to (system, user, options_visible).

    options_visible=True means the model saw the option set in stage-1 (visible
    MCQ / embedded MCQ); the pipeline trusts the letter and skips stage-2.
    """
    question = item["question"]
    options  = item.get("options")

    if not options:
        n_ans = question.count("[ANS]")
        qlen  = len(question)

        # Multipart must precede embedded-MCQ detection: multipart questions
        # often label subparts A./B./C. and would false-positive otherwise.
        if n_ans > 1:
            user = _append_multipart_slotcount(
                _append_format_constraints(question, question),
                n_ans,
            )
            return SYSTEM_PROMPT_MULTIPART, user, False

        embedded_options = detect_embedded_mcq(question)
        if embedded_options is not None:
            user = _append_mcq_letter_instruction(
                _append_format_constraints(question, question),
            )
            return SYSTEM_PROMPT_MCQ, user, True

        # qlen >= 150 floor filters out short trig-simplify items with corrupted
        # [ANS] markers that would otherwise misroute to olympiad.
        if n_ans == 0 and qlen >= 150:
            # Olympiad branch: no system prompt + MathArena-style minimal user
            # prompt (asks for exact closed-form, no approximation).
            user = (
                "Solve the following problem. Put your final answer within \\boxed{}. "
                "If the answer is a closed-form expression, leave it in exact form "
                "(do not approximate)."
                "\n\n" + question
            )
            user = _append_format_constraints(user, question)
            return None, user, False

        if qlen > 500:
            user = _append_precision_hint(_append_format_constraints(question, question), question)
            return SYSTEM_PROMPT_LONG, user, False

        user = _append_precision_hint(_append_format_constraints(question, question), question)
        return SYSTEM_PROMPT_MATH, user, False

    if mcq_needs_options(question, options):
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        user = _append_mcq_letter_instruction(
            _append_format_constraints(f"{question}\n\nOptions:\n{opts_text}", question),
        )
        return SYSTEM_PROMPT_MCQ, user, True

    # Hidden-options MCQ: stage-1 derives an open answer; lenient-verify maps it to a letter.
    user = _append_precision_hint(_append_format_constraints(question, question), question)
    return SYSTEM_PROMPT_MATH, user, False


def verify_against_options(response: str, options: list) -> Optional[str]:
    """Strict option matcher (kept for diff/A-B; production uses lenient variant)."""
    sys.path.insert(0, ".")
    from judger import Judger
    judger_local = Judger(strict_extract=True)
    for i, opt in enumerate(options):
        try:
            if judger_local.auto_judge(pred=response, gold=[opt], options=[[]]):
                return chr(65 + i)
        except Exception:
            continue
    return None


_STAGE2_THINK_STRIP_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def build_mcq_stage2_messages(item: dict, stage1_response: str) -> list:
    """Build the multi-turn chat for the stage-2 MCQ reconcile call.

    Stage-1's <think> block is stripped from the assistant history per the
    Qwen3-Thinking chat-template contract (rolling-checkpoint behavior).
    """
    question = item["question"]
    options  = item["options"]
    labels   = [chr(65 + i) for i in range(len(options))]
    opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))

    stage1_stripped = _STAGE2_THINK_STRIP_RE.sub("", stage1_response).strip()

    return [
        {"role": "system",    "content": SYSTEM_PROMPT_MATH},
        {"role": "user",      "content": question},
        {"role": "assistant", "content": stage1_stripped},
        {"role": "user",      "content": f"Options:\n{opts_text}\n\n{MCQ_STAGE2_INSTRUCTION}"},
    ]


def verify_against_options_lenient(
    response: str,
    options: list,
    precision: float = 1e-2,
) -> Optional[str]:
    """Best-match option lookup: tries numeric distance, then sympy equality, then decimal compat."""
    sys.path.insert(0, ".")
    from judger import Judger
    pred = Judger(strict_extract=True).extract_ans(response)
    if not pred:
        return None

    # MCQ distractors cluster within 1% of each other; the caller's free-form
    # precision would over-match across multiple options.
    precision = min(precision, 1e-4)

    def _clean(s: str) -> str:
        s = s.strip().strip("$").strip()
        return s.rstrip(";,.!? ").strip()

    _CLEAN_NUM_RE = re.compile(r"^-?(?:\d+(?:\.\d+)?|\.\d+|\d+/\d+)$")
    def _is_clean_number(s: str) -> bool:
        return bool(_CLEAN_NUM_RE.match(s.strip()))

    def _to_float(s: str) -> float:
        s = s.strip().replace(",", "")
        if "/" in s:
            num, den = s.split("/", 1)
            return float(num) / float(den)
        return float(s)

    def _numeric_distance(a: str, b: str) -> Optional[float]:
        if not (_is_clean_number(a) and _is_clean_number(b)):
            return None
        try:
            av, bv = _to_float(a), _to_float(b)
            denom = max(abs(bv), 1e-12)
            return abs(av - bv) / denom
        except (ValueError, TypeError, ZeroDivisionError):
            return None

    def _try_sympy_equality(a: str, b: str) -> bool:
        try:
            from sympy import simplify, sympify
            from sympy.parsing.latex import parse_latex
        except ImportError:
            return False

        def _parse(s: str):
            # parse_latex("sin(x)**2") silently atomizes to garbage; detect flavor first.
            looks_like_latex = ("\\" in s) or ("{" in s and "}" in s)
            s2 = s.replace("\\cdot", "*").replace("\\times", "*").replace("^", "**")
            if looks_like_latex:
                try:
                    return parse_latex(s)
                except Exception:
                    pass
                try:
                    return sympify(s2)
                except Exception:
                    return None
            else:
                try:
                    return sympify(s2)
                except Exception:
                    pass
                try:
                    return parse_latex(s)
                except Exception:
                    return None

        ae, be = _parse(a), _parse(b)
        if ae is None or be is None:
            return False

        # SIGALRM guard: simplify can hang on adversarial expressions.
        import signal
        class _Timeout(Exception): pass
        def _handler(signum, frame): raise _Timeout()
        old_handler = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(3)
        try:
            return simplify(ae - be) == 0
        except Exception:
            return False
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def _try_decimal_compat(a: str, b: str) -> bool:
        # Both-numeric path that ignores trailing-zero precision (e.g. 3.14 == 3.140).
        if not (_is_clean_number(a) and _is_clean_number(b)):
            return False
        try:
            av = _to_float(a)
            bv = _to_float(b)
        except (ValueError, TypeError, ZeroDivisionError):
            return False

        def _dp_count(s: str) -> int:
            s_clean = s.strip().replace(",", "").lstrip("-+")
            if "." not in s_clean:
                return 0
            return len(s_clean.split(".")[1].rstrip("0"))

        a_dp, b_dp = _dp_count(a), _dp_count(b)
        common_dp = min(a_dp, b_dp)
        if common_dp == 0:
            return abs(av - bv) < 0.5
        return round(av, common_dp) == round(bv, common_dp)

    pred_norm = _clean(pred)

    best_letter   = None
    best_distance = float("inf")

    for i, opt in enumerate(options):
        letter   = chr(65 + i)
        opt_norm = _clean(opt)

        if pred_norm == opt_norm:
            return letter

        d = _numeric_distance(pred_norm, opt_norm)
        if d is not None:
            if d < precision and d < best_distance:
                best_distance = d
                best_letter   = letter
            continue

        if _try_sympy_equality(pred_norm, opt_norm):
            if 0 < best_distance:
                best_distance = 0
                best_letter   = letter
            continue

        if _try_decimal_compat(pred_norm, opt_norm):
            if 0 < best_distance:
                best_distance = 0
                best_letter   = letter

    return best_letter


_WORD2DIGIT = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_NUM_TOKEN = r"(\d+|" + "|".join(_WORD2DIGIT) + r")"


def _parse_num_token(tok: str) -> int:
    return int(tok) if tok.isdigit() else _WORD2DIGIT[tok.lower()]


def infer_precision_from_question(question: str, default: float = 1e-2) -> float:
    """Parse question text for explicit precision spec; return relative tolerance."""
    q = (question or "").lower()
    if re.search(r"\b(?:exact(?:\s+(?:value|answer))?|do not round|no rounding)\b", q):
        return 1e-8
    m = re.search(
        rf"(?:to|correct to|accurate to|round(?:ed)? to|at least|use(?:s|d|ing)?|with)\s+{_NUM_TOKEN}\s+decimal\s+places?",
        q,
    )
    if m:
        return 10 ** (-_parse_num_token(m.group(1)))
    m = re.search(
        rf"(?:to|correct to|accurate to|with|using)\s+{_NUM_TOKEN}\s+(?:significant\s+(?:figures?|digits?)|sig\.?\s*figs?)",
        q,
    )
    if m:
        return 10 ** (-(_parse_num_token(m.group(1)) - 1))
    m = re.search(r"nearest\s+(tenth|hundredth|thousandth|ten-thousandth|hundred-thousandth)", q)
    if m:
        return {"tenth":1e-1,"hundredth":1e-2,"thousandth":1e-3,"ten-thousandth":1e-4,"hundred-thousandth":1e-5}[m.group(1)]
    return default


# %%
# Verify on real dataset samples
print("Baseline build_prompt on MCQ sample")
sys_p, usr_p = build_prompt(mcq_sample["question"], mcq_sample.get("options"))
print(f"  system: {sys_p[:60]}...")
print(f"  user  : {usr_p[:100]}... (includes options)")

print("\nselect_prompt on MCQ sample (derivation MCQ — stage 1, no options)")
prepped = preprocess_question(mcq_sample)
sys_p, usr_p, opts_visible = select_prompt(prepped)
print(f"  system: {sys_p[:60]}...")
print(f"  user  : {usr_p[:80]}...")
print(f"  options_visible = {opts_visible}  (False => two-stage flow)")

print("\nRouting on free-form sample (Q0, short)")
prepped = preprocess_question(free_sample)
sys_p, usr_p, opts_visible = select_prompt(prepped)
print(f"  qlen={len(prepped['question'])} n_ans={prepped['question'].count('[ANS]')}")
print(f"  system: {sys_p[:60]}...")
print(f"  options_visible = {opts_visible}")

print("\nHeuristic check on an options-required MCQ (id 281, 'which of the following ... is unbiased')")
opt_required_sample = next((d for d in data if d.get("id") == 281), None)
if opt_required_sample is not None:
    prepped = preprocess_question(opt_required_sample)
    sys_p, usr_p, opts_visible = select_prompt(prepped)
    print(f"  system: {sys_p[:60]}...")
    print(f"  user  : {usr_p[:120]}...")
    print(f"  options_visible = {opts_visible}  (True => single-stage with options inline)")
else:
    print("  (id 281 not present in this dataset)")

print("\nStage-2 message structure (mock — no real stage-1 response yet)")
mock_stage1 = "After computing the integral, I get \\boxed{2/a}."
msgs = build_mcq_stage2_messages(preprocess_question(mcq_sample), mock_stage1)
for m in msgs:
    print(f"  [{m['role']:9s}] {m['content'][:80]}...")


# %% [markdown]
# ## 5. Load Model with vLLM
# 
# We load **Qwen3-4B-Thinking-2507** with **INT8 quantization** via BitsAndBytes.  
# Setting `load_format="bitsandbytes"` tells vLLM to apply on-the-fly INT8 weight quantization, roughly halving GPU memory usage compared to BF16.
# 
# Key parameters:
# - `gpu_memory_utilization` — fraction of GPU VRAM reserved for the model and KV cache
# - `max_model_len` — maximum sequence length (prompt + generation)
# - `max_num_seqs` — maximum number of sequences processed in parallel

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# max_model_len=90112 holds MAX_TOKENS_OLYMPIAD=81920 + input headroom; max_num_seqs
# halved to 32 because 24 GB A30 can't fit 64 sequences at that context length.
llm = LLM(
    model=MODEL_ID,
    quantization="bitsandbytes",       # INT8 weight quant; fits 4B model in ~8 GB
    load_format="bitsandbytes",
    enable_prefix_caching=True,        # olympiad SC re-prefills the same prefix 7+ times per item
    gpu_memory_utilization=0.90,
    max_model_len=90112,               # hard ceiling on prompt + generation tokens
    trust_remote_code=True,
    max_num_seqs=32,                   # max concurrent sequences in the batched scheduler
    max_num_batched_tokens=16384,      # per-step prefill budget (bounds peak memory)
    enforce_eager=False,               # let vLLM compile CUDA graphs for ~10% speedup
)

# Default sampling: top-p/top-k/min-p values are from the Qwen3-Thinking model card.
sampling_params = SamplingParams(
    max_tokens=MAX_TOKENS,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    seed=SEED,
)

# Olympiad-specific sampler with extended token budget for long competition proofs.
MAX_TOKENS_OLYMPIAD = 81920
sampling_params_olympiad = SamplingParams(
    max_tokens=MAX_TOKENS_OLYMPIAD,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    seed=SEED,
)

print("Model loaded.")


# %%


# %% [markdown]
# ## 6. Generate Responses
# 
# We format every question into a chat-template prompt, then call `llm.generate()` in one batched pass.  
# vLLM handles batching and scheduling internally — no manual batching needed.

# %%


# %%
# DISABLED for submission: subset = full private set (set in next cell)
import hashlib


# %%
# ── Submission partition partB: ids [707, 943) ─────────────────
# Each teammate runs a disjoint slice; their cache files are merged into one
# final CSV after all runs complete.
MY_ID_START, MY_ID_END = args.id_start, args.id_end
subset = [d for d in data if MY_ID_START <= d["id"] < MY_ID_END]
if LIMIT is not None:
    subset = subset[:LIMIT]
    print(f"[--limit {LIMIT}] testing on first {len(subset)} question(s) of the partition")
n_mcq  = sum(1 for d in subset if d.get("options"))
print(f"Partition {args.part_tag}: ids [{MY_ID_START}, {MY_ID_END}) = "
      f"{len(subset)} questions ({n_mcq} MCQ, {len(subset)-n_mcq} free-form)")

# %% [markdown]
# For save/resume from interuption

# %%
# Cache key = hash(all system prompts + PIPELINE_TAG); data_tag suffix keeps
# the public-eval cache separate from the private-submission cache.
PIPELINE_TAG = "v2.9.3"
prompt_hash = hashlib.md5(
    (SYSTEM_PROMPT_MATH + "||" + SYSTEM_PROMPT_MCQ + "||"
     + MCQ_STAGE2_INSTRUCTION + "||" + SYSTEM_PROMPT_LONG + "||"
     + SYSTEM_PROMPT_MULTIPART + "||" + PIPELINE_TAG).encode()
).hexdigest()[:8]
data_tag = Path(DATA_PATH).stem
PART_TAG = args.part_tag   # disjoint partition tag; matches the slice above
CACHE_PATH = f"results/cache/{prompt_hash}_seed{SEED}_{data_tag}_{PART_TAG}.jsonl"
Path(CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)

# Scorer only needs 'response'; 'stage1' is diagnostic.
cache = {}
if Path(CACHE_PATH).exists():
    with open(CACHE_PATH) as f:
        for line in f:
            e = json.loads(line)
            cache[e["id"]] = e["response"]
print(f"Cache hits: {len(cache)} for prompt hash {prompt_hash} ({PIPELINE_TAG})")

to_generate = [item for item in subset if item["id"] not in cache]
n_to_gen_mcq  = sum(1 for it in to_generate if it.get("options"))
print(f"Need to generate: {len(to_generate)} ({n_to_gen_mcq} MCQ stage-1, {len(to_generate)-n_to_gen_mcq} free-form)")


# %%
# Crash-safe mini-batched generation pipeline (submission variant — no eval scoring).
# Per batch:
#   1. stage-1 sample (olympiad/long-MCQ-hidden get the 81920-token sampler)
#   2. olympiad multi-temperature self-consistency (K=8 samples, vote + shortest tie-break)
#   3. MathArena last-chance reprompt for any response missing \boxed{}
#   4. MCQ classify: visible-opts trust-the-letter / hidden-opts lenient-verify -> stage-2 reconcile
#   5. append batch to JSONL cache (at most one batch lost on interrupt)
BATCH_SIZE = 32

# Verbatim from https://github.com/eth-sri/matharena/blob/main/src/matharena/runner.py
LAST_CHANCE_REPROMPT_BASE = (
    "Your last message does not provide a final answer in a way that follows the "
    "formatting instructions. Please based on the conversation history, report the "
    "final answer again within \\boxed{}. If you did not find the answer, please use "
    "\\boxed{None}. Do not reason about the problem again or use tools, simply try "
    "to extract the final answer from the previous reasoning. Boxed answers in "
    "thinking/reasoning stages will be ignored; only the final response message is "
    "considered."
)


def _build_reprompt(options_visible: bool, options_list: Optional[list]) -> str:
    """Construct the last-chance reprompt; visible-MCQ items get a letter-only appendage."""
    # Hidden-options MCQ must derive openly; appending options would bias stage-1 reprompt.
    if options_visible and options_list:
        labels   = [chr(65 + i) for i in range(len(options_list))]
        opts_str = ", ".join(f"\\boxed{{{l}}}" for l in labels)
        return (
            LAST_CHANCE_REPROMPT_BASE
            + f" Recall that this is a multiple choice problem, and the only valid "
              f"answers to put within \\boxed{{}} are: {opts_str}."
        )
    return LAST_CHANCE_REPROMPT_BASE


# Reprompt is a deterministic re-extraction call, not a re-derivation: T=0, short budget.
sampling_params_reprompt = SamplingParams(
    max_tokens=2048,
    temperature=0.0,
    top_p=1.0,
    seed=SEED,
    n=1,
)

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _build_chat(sys_p, user_p):
    """Build messages list, omitting system role when sys_p is None (olympiad)."""
    if sys_p is None:
        return [{"role": "user", "content": user_p}]
    return [
        {"role": "system", "content": sys_p},
        {"role": "user",   "content": user_p},
    ]


def _has_final_box(response: str) -> bool:
    """True iff a \\boxed{...} exists OUTSIDE <think>...</think>."""
    text = _THINK_BLOCK_RE.sub("", response)
    return bool(re.search(r"\\boxed\{", text))


from collections import Counter

OLYMPIAD_SC_TEMPS = []   # deadline crunch: skip multi-temp SC — olympiad gets only stage-1 sample
OLYMPIAD_SC_K = 1 + len(OLYMPIAD_SC_TEMPS)
# Raised to 5 to avoid wrong-confident-consensus traps where 3 samples agree on a bad answer.
OLYMPIAD_SC_EARLY_STOP_MIN = 5


def _extract_last_box(text: str) -> str:
    """Extract content of last \\boxed{...} after stripping <think> blocks; handles nested braces."""
    text = _THINK_BLOCK_RE.sub("", text)
    boxes = []
    i = 0
    while True:
        m = re.search(r"\\boxed\{", text[i:])
        if not m:
            break
        start = i + m.end()
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        if depth == 0:
            boxes.append(text[start:j-1])
            i = j
        else:
            break
    return boxes[-1].strip() if boxes else ""


def _normalize_vote_key_olympiad(boxed_str: str) -> str:
    """Canonical vote key: float -> sympy.simplify(parse_latex).srepr -> cleaned string."""
    s = (boxed_str or "").strip()
    if not s:
        return ""
    try:
        return f"{float(s):.10g}"
    except (ValueError, TypeError):
        pass
    # parse_latex can hang on adversarial input.
    try:
        import signal
        from sympy import simplify, srepr
        from sympy.parsing.latex import parse_latex

        class _Timeout(Exception): pass
        def _handler(signum, frame): raise _Timeout()
        _old = signal.signal(signal.SIGALRM, _handler)
        signal.alarm(1)
        try:
            expr = parse_latex(s)
            return srepr(simplify(expr))
        except Exception:
            pass
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, _old)
    except ImportError:
        pass
    return re.sub(r"\s+|\\left|\\right|\\!|\\,|\\;", "", s).lower()


def _majority_vote_shortest(samples: list, key_fn) -> tuple:
    """Plurality vote with shortest-trace tie-break; excludes empty (failed-extract) keys."""
    if not samples:
        return "", 0
    keys = [key_fn(s) for s in samples]
    non_empty = [(s, k) for s, k in zip(samples, keys) if k]
    if not non_empty:
        # All samples failed to box; return shortest as placeholder with vote_count=0.
        return min(samples, key=lambda s: len(_THINK_BLOCK_RE.sub("", s))), 0
    counter = Counter(k for _, k in non_empty)
    top_count = counter.most_common(1)[0][1]
    tied_keys = {k for k, c in counter.items() if c == top_count}
    candidates = [(s, k) for s, k in non_empty if k in tied_keys]
    winner, _ = min(candidates, key=lambda sk: len(_THINK_BLOCK_RE.sub("", sk[0])))
    return winner, top_count


total_batches = (len(to_generate) + BATCH_SIZE - 1) // BATCH_SIZE if to_generate else 0
total_mcq_visible_opts = 0
total_mcq_s1_hits = 0
total_stage2_calls = 0
total_reprompts = 0

for batch_start in range(0, len(to_generate), BATCH_SIZE):
    batch = to_generate[batch_start:batch_start + BATCH_SIZE]
    batch_num = batch_start // BATCH_SIZE + 1
    prepped = [preprocess_question(item) for item in batch]

    # Stage 1
    stage1_prompts = []
    options_visible_flags = []
    sys_user_pairs = []
    for p in prepped:
        sys_p, user_p, opts_visible = select_prompt(p)
        options_visible_flags.append(opts_visible)
        sys_user_pairs.append((sys_p, user_p))
        stage1_prompts.append(tokenizer.apply_chat_template(
            _build_chat(sys_p, user_p),
            tokenize=False,
            add_generation_prompt=True,
        ))
    # Per-prompt sampling: olympiad AND long hidden-options MCQ use the larger token budget.
    sp_list = []
    for i, (sys_p, _) in enumerate(sys_user_pairs):
        p = prepped[i]
        is_long_mcq_hidden = (
            p.get("options") is not None
            and not options_visible_flags[i]
            and len(p["question"]) > 500
        )
        if sys_p is None or is_long_mcq_hidden:
            sp_list.append(sampling_params_olympiad)
        else:
            sp_list.append(sampling_params)
    stage1_outs = llm.generate(stage1_prompts, sampling_params=sp_list)
    stage1_texts = [o.outputs[0].text.strip() for o in stage1_outs]

    # Olympiad multi-temperature SC: collect K=8 samples, vote with shortest tie-break.
    olympiad_idx = [i for i, (sys_p, _) in enumerate(sys_user_pairs) if sys_p is None]
    olympiad_sc_meta = {}
    if olympiad_idx:
        oly_samples = {i: [stage1_texts[i]] for i in olympiad_idx}
        oly_done = set()
        for t_pass_idx, temp in enumerate(OLYMPIAD_SC_TEMPS, start=1):
            alive = [i for i in olympiad_idx if i not in oly_done]
            if not alive:
                break
            sp_t = SamplingParams(
                max_tokens=MAX_TOKENS_OLYMPIAD,
                temperature=temp,
                top_p=0.95,
                top_k=20,
                min_p=0.0,
                seed=SEED + t_pass_idx,
                n=1,
            )
            alive_prompts = [stage1_prompts[i] for i in alive]
            t_outs = llm.generate(alive_prompts, sampling_params=sp_t)
            for j, i in enumerate(alive):
                oly_samples[i].append(t_outs[j].outputs[0].text.strip())
                if len(oly_samples[i]) >= OLYMPIAD_SC_EARLY_STOP_MIN:
                    keys = [_normalize_vote_key_olympiad(_extract_last_box(s))
                            for s in oly_samples[i]]
                    # Never early-stop on empty consensus (all-failed-to-box != real agreement).
                    if all(k for k in keys) and len(set(keys)) == 1:
                        oly_done.add(i)
        for i in olympiad_idx:
            winner, votes = _majority_vote_shortest(
                oly_samples[i],
                key_fn=lambda s: _normalize_vote_key_olympiad(_extract_last_box(s)),
            )
            stage1_texts[i] = winner
            olympiad_sc_meta[i] = (len(oly_samples[i]), votes, i in oly_done)
        n_oly = len(olympiad_idx)
        n_early = sum(1 for _, _, es in olympiad_sc_meta.values() if es)
        n_samples_taken = sum(k for k, _, _ in olympiad_sc_meta.values())
        print(f"  olympiad SC: {n_oly} items, {n_samples_taken} samples, "
              f"{n_early} early-stopped")

    # Last-chance reprompt for items missing \boxed{} after </think>.
    reprompt_idx = [i for i, s1 in enumerate(stage1_texts) if not _has_final_box(s1)]
    if reprompt_idx:
        reprompt_prompts = []
        for i in reprompt_idx:
            sys_p, user_p = sys_user_pairs[i]
            opts_list = prepped[i].get("options") if options_visible_flags[i] else None
            # Embedded-MCQ has no item['options']; recover them from inline structure.
            if options_visible_flags[i] and not opts_list:
                opts_list = detect_embedded_mcq(prepped[i]["question"])
            reprompt_user = _build_reprompt(options_visible_flags[i], opts_list)
            base_msgs = _build_chat(sys_p, user_p)
            msgs = base_msgs + [
                {"role": "assistant", "content": stage1_texts[i]},
                {"role": "user",      "content": reprompt_user},
            ]
            reprompt_prompts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            ))
        reprompt_outs = llm.generate(reprompt_prompts, sampling_params=sampling_params_reprompt)
        for k, i in enumerate(reprompt_idx):
            # Append (not replace) so judger's last-contiguous-group extraction picks up the reprompt box.
            stage1_texts[i] = stage1_texts[i] + "\n\n" + reprompt_outs[k].outputs[0].text.strip()

    final_response = [None] * len(batch)
    stage1_field   = [None] * len(batch)
    needs_stage2_idx = []
    n_mcq_visible_opts = 0
    n_mcq_s1_hits = 0
    n_freeform = 0

    for i, (orig_item, prep_item, s1) in enumerate(zip(batch, prepped, stage1_texts)):
        if not orig_item.get("options") and not options_visible_flags[i]:
            final_response[i] = s1
            n_freeform += 1
        elif options_visible_flags[i]:
            final_response[i] = s1
            n_mcq_visible_opts += 1
        else:
            stage1_field[i] = s1
            item_precision = infer_precision_from_question(
                prep_item["question"], default=EVAL_PRECISION
            )
            letter = verify_against_options_lenient(
                s1, prep_item["options"], precision=item_precision
            )
            if letter is not None:
                final_response[i] = f"\\boxed{{{letter}}}"
                n_mcq_s1_hits += 1
            else:
                needs_stage2_idx.append(i)

    if needs_stage2_idx:
        s2_prompts = []
        for i in needs_stage2_idx:
            msgs = build_mcq_stage2_messages(prepped[i], stage1_texts[i])
            s2_prompts.append(tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            ))
        s2_outs = llm.generate(s2_prompts, sampling_params=sampling_params)
        for k, i in enumerate(needs_stage2_idx):
            final_response[i] = s2_outs[k].outputs[0].text.strip()

    with open(CACHE_PATH, "a") as f:
        for i, item in enumerate(batch):
            rec = {"id": item["id"], "response": final_response[i]}
            if stage1_field[i] is not None:
                rec["stage1"] = stage1_field[i]
            if i in olympiad_sc_meta:
                k_used, vote_count, early_stopped = olympiad_sc_meta[i]
                rec["sc_K"] = k_used
                rec["sc_vote_count"] = vote_count
                rec["sc_early_stopped"] = early_stopped
            cache[item["id"]] = final_response[i]
            f.write(json.dumps(rec) + "\n")

    total_mcq_visible_opts += n_mcq_visible_opts
    total_mcq_s1_hits += n_mcq_s1_hits
    total_stage2_calls += len(needs_stage2_idx)
    total_reprompts += len(reprompt_idx)
    print(f"Batch {batch_num}/{total_batches}: "
          f"MCQ visible-opts={n_mcq_visible_opts}, hidden s1-hit={n_mcq_s1_hits}, "
          f"stage-2={len(needs_stage2_idx)}, free-form={n_freeform}, "
          f"reprompts={len(reprompt_idx)} | total cached={len(cache)}/{len(subset)}")

if to_generate:
    print(f"\nGeneration complete. MCQ visible-opts: {total_mcq_visible_opts}, "
          f"hidden s1-hits: {total_mcq_s1_hits}, stage-2 calls: {total_stage2_calls}, "
          f"last-chance reprompts: {total_reprompts}")

responses = [cache[item["id"]] for item in subset]
print(f"Ready: {len(responses)} responses for scoring")

for i in range(min(2, len(responses))):
    print(f"\n── Response (id={subset[i]['id']}, type={'MCQ' if subset[i].get('options') else 'free-form'}) ──")
    print(responses[i][:300], "..." if len(responses[i]) > 300 else "")


# %%
assert len(responses) == len(subset), \
    f"Length mismatch: {len(responses)} responses vs {len(subset)} subset items"

mcq_missing_box = []
for item, resp in zip(subset, responses):
    if item.get("options") and not re.search(r"\\boxed\{[A-Z]\}", resp):
        mcq_missing_box.append(item["id"])
if mcq_missing_box:
    print(f"WARNING: {len(mcq_missing_box)} MCQ items have no single-letter \\boxed: {mcq_missing_box}")
    print("  (scorer will fall back to last standalone capital letter — acceptable but worth noting)")
else:
    print(f"OK: all {sum(1 for d in subset if d.get('options'))} MCQ items have a \\boxed{{LETTER}}")

visible_opt_ids = [
    item["id"] for item in subset
    if item.get("options") and mcq_needs_options(item["question"], item["options"])
]
print(f"\nMCQ items routed visible-options (n={len(visible_opt_ids)}): {visible_opt_ids}")

print("\nSample cache entries (for human inspection):")
sample_count = 0
with open(CACHE_PATH) as f:
    for line in f:
        if sample_count >= 2:
            break
        e = json.loads(line)
        if "stage1" not in e:
            continue
        is_synthetic = e["response"].startswith("\\boxed{") and len(e["response"]) <= 12
        kind = "stage-1 hit (synthetic \\boxed{})" if is_synthetic else "stage-2 used (full response)"
        resp_preview   = e["response"][:120] + ("..." if len(e["response"]) > 120 else "")
        stage1_preview = e["stage1"][:120]   + ("..." if len(e["stage1"])   > 120 else "")
        print(f"\n  [id={e['id']}] {kind}")
        print(f"    response: {resp_preview}")
        print(f"    stage1  : {stage1_preview}")
        sample_count += 1


# %% [markdown]
# ## 7. Score Responses
# 
# Scoring differs by question type:
# 
# - **MCQ**: extract the predicted letter from `\boxed{}` and compare to the gold letter (exact match).
# - **Free-form**: use `Judger.auto_judge()` which handles symbolic and numeric equivalence.
# 
# Each result record contains `{id, is_mcq, gold, response, correct}`.

# %%
# DISABLED for submission (no gold answers in private set)


# %% [markdown]
# ## 8. Summary
# 
# Print accuracy broken down by question type.

# %%
# DISABLED for submission (no gold answers in private set)


# %% [markdown]
# ## 9. Save Results
# 
# Results are written as newline-delimited JSON.
# 
# **With evaluation** (public set — you have ground-truth):  
# Each line: `{id, is_mcq, gold, response, correct}`
# 
# **Without evaluation** (private test set — no ground-truth available):  
# Each line: `{id, is_mcq, response}` — omit `gold` and `correct`.
# 
# Toggle `SAVE_EVAL` below accordingly.

# %%
# DISABLED for submission (no gold answers in private set)


# %%
# Kaggle expects two columns (id, response); quote-all so LaTeX commas don't break the CSV.
import csv
out_path = Path(OUTPUT_PATH)
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", newline="") as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(["id", "response"])
    for item, resp in zip(subset, responses):
        writer.writerow([item["id"], resp])
print(f"Saved {len(subset)} rows to {out_path}")


# %% [markdown]
# ## Next Steps
# 
# This notebook gives you a working baseline. Here are directions to improve your score:
# 
# - **Prompt engineering** — try different system prompts or few-shot examples inside the user turn
# - **Sampling parameters** — adjust `temperature`, `top_p`, or use majority voting across multiple samples
# - **Fine-tuning** — the competition allows model fine-tuning; see the course resources for guidance
# 
# Good luck!


