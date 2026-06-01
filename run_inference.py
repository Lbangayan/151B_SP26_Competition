"""Single entry-point for the CSE 151B Math Reasoning Competition submission.

Wraps the v2.9.3 baseline pipeline (conditional routing across 7 question
categories, optional olympiad multi-temperature self-consistency, MathArena
last-chance reprompt, two-stage hidden-options MCQ flow) into one callable.

Usage
-----
    python run_inference.py --data data/private.jsonl --output submission.csv

    # Fast mode (olympiad SC disabled, ~3x speedup, ~3-5pp olympiad accuracy cost)
    python run_inference.py --data data/private.jsonl --output submission.csv --no-olympiad-sc

Or programmatically:
    from run_inference import run_inference
    run_inference(data_path="data/private.jsonl", output_path="submission.csv")

The script reads from `data_path`, generates responses with the configured
pipeline, writes incremental cache to `cache_dir`, and emits a final CSV with
columns [id, response] ready for Kaggle upload. Cache is crash-safe: rerunning
resumes from the last completed batch.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional


# ─── Default config ─────────────────────────────────────────────────────────
MODEL_ID       = "Qwen/Qwen3-4B-Thinking-2507"
MAX_TOKENS     = 32768
MAX_TOKENS_OLYMPIAD_DEFAULT = 81920
SEED           = 13
BATCH_SIZE     = 32
EVAL_PRECISION = 1e-2
PIPELINE_TAG   = "v2.9.3"

# K=8 olympiad SC schedule (1 stage-1 + 7 inner passes). Set to [] for fast mode.
OLYMPIAD_SC_TEMPS_DEFAULT = [0.4, 0.6, 0.8, 0.8, 1.0, 1.0, 1.2]
OLYMPIAD_SC_EARLY_STOP_MIN = 5


# ═══════════════════════════════════════════════════════════════════════════
# Preprocessing: Unicode-math normalization + LaTeX bare-command repair
# ═══════════════════════════════════════════════════════════════════════════

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


_LATEX_CMDS = [
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
    for cmd in _LATEX_CMDS
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


# ═══════════════════════════════════════════════════════════════════════════
# System prompts + user-prompt-tail helpers + router
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# Generation helpers (used inside the per-batch loop)
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════

def run_inference(
    data_path: str = "data/private.jsonl",
    output_path: str = "submission.csv",
    cache_dir: str = "results/cache",
    gpu_id: str = "0",
    max_model_len: int = 90112,
    max_num_seqs: int = 32,
    max_num_batched_tokens: int = 16384,
    enforce_eager: bool = False,
    enable_olympiad_sc: bool = True,
    olympiad_sc_temps: Optional[list] = None,
    max_tokens_olympiad: int = MAX_TOKENS_OLYMPIAD_DEFAULT,
) -> str:
    """Run the full v2.9.3 inference pipeline end-to-end. Returns path to the CSV.

    Parameters
    ----------
    data_path
        JSONL file with one question per line: {id, question, [options], ...}
    output_path
        Where to write the Kaggle submission CSV (id, response).
    cache_dir
        Per-batch JSONL cache directory; reruns resume from here.
    gpu_id
        CUDA_VISIBLE_DEVICES value (single GPU only).
    max_model_len, max_num_seqs, max_num_batched_tokens, enforce_eager
        vLLM init params. Defaults tuned for a 24 GB Ampere card (A30 / A5000)
        running Qwen3-4B-Thinking at bnb 4-bit. Lower max_num_seqs / max_model_len
        if you observe OOM on smaller GPUs.
    enable_olympiad_sc
        If True, run K=8 multi-temperature self-consistency on olympiad items
        (~3x wall-clock cost on olympiad-heavy batches). If False, only stage-1.
    olympiad_sc_temps
        Override the SC temperature schedule. None means use the v2.9.2 default.
    max_tokens_olympiad
        Output budget for olympiad and long-MCQ-hidden items. 81920 by default;
        reduce if max_model_len is reduced.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Late imports so module import works even without vllm installed.
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    olympiad_sc_temps = (
        olympiad_sc_temps if olympiad_sc_temps is not None
        else (OLYMPIAD_SC_TEMPS_DEFAULT if enable_olympiad_sc else [])
    )

    # ── Load data ───────────────────────────────────────────────────────────
    data = [json.loads(line) for line in open(data_path)]
    subset = data
    n_mcq = sum(1 for d in subset if d.get("options"))
    print(f"Loaded {len(subset)} questions from {data_path} ({n_mcq} MCQ, {len(subset)-n_mcq} free-form)")

    # ── Init vLLM ───────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=MODEL_ID,
        quantization="bitsandbytes",       # 4-bit weight quant; ~2 GB resident for 4B model
        load_format="bitsandbytes",
        enable_prefix_caching=True,        # olympiad SC re-prefills the same prompt 7+ times per item
        gpu_memory_utilization=0.90,
        max_model_len=max_model_len,       # hard ceiling on prompt + generation tokens
        trust_remote_code=True,
        max_num_seqs=max_num_seqs,         # max concurrent sequences in the scheduler
        max_num_batched_tokens=max_num_batched_tokens,
        enforce_eager=enforce_eager,
    )

    # Sampling params (top-p / top-k / min-p values come from Qwen3-Thinking model card).
    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        seed=SEED,
    )

    # Olympiad-specific sampler with extended budget for long competition proofs.
    sampling_params_olympiad = SamplingParams(
        max_tokens=max_tokens_olympiad,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        seed=SEED,
    )

    # Reprompt is deterministic re-extraction, not re-derivation: T=0, short budget.
    sampling_params_reprompt = SamplingParams(
        max_tokens=2048,
        temperature=0.0,
        top_p=1.0,
        seed=SEED,
        n=1,
    )
    print("Model loaded.")

    # ── Cache key (hash of all system prompts + PIPELINE_TAG) ───────────────
    prompt_hash = hashlib.md5(
        (SYSTEM_PROMPT_MATH + "||" + SYSTEM_PROMPT_MCQ + "||"
         + MCQ_STAGE2_INSTRUCTION + "||" + SYSTEM_PROMPT_LONG + "||"
         + SYSTEM_PROMPT_MULTIPART + "||" + PIPELINE_TAG).encode()
    ).hexdigest()[:8]
    data_tag = Path(data_path).stem
    cache_path = Path(cache_dir) / f"{prompt_hash}_seed{SEED}_{data_tag}.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                e = json.loads(line)
                cache[e["id"]] = e["response"]
    print(f"Cache hits: {len(cache)} for prompt hash {prompt_hash} ({PIPELINE_TAG})")

    to_generate = [item for item in subset if item["id"] not in cache]
    n_to_gen_mcq = sum(1 for it in to_generate if it.get("options"))
    print(f"Need to generate: {len(to_generate)} ({n_to_gen_mcq} MCQ, {len(to_generate)-n_to_gen_mcq} free-form)")

    # ── Per-batch generation loop ───────────────────────────────────────────
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

        # Per-prompt sampling: olympiad AND long hidden-options MCQ get the bigger budget.
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

        # Olympiad multi-temperature SC (K=8 by default; skipped if olympiad_sc_temps == []).
        olympiad_idx = [i for i, (sys_p, _) in enumerate(sys_user_pairs) if sys_p is None]
        olympiad_sc_meta = {}
        if olympiad_idx:
            oly_samples = {i: [stage1_texts[i]] for i in olympiad_idx}
            oly_done = set()
            for t_pass_idx, temp in enumerate(olympiad_sc_temps, start=1):
                alive = [i for i in olympiad_idx if i not in oly_done]
                if not alive:
                    break
                sp_t = SamplingParams(
                    max_tokens=max_tokens_olympiad,
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

        # Crash-safe batch flush.
        with open(cache_path, "a") as f:
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

    # ── Write CSV ───────────────────────────────────────────────────────────
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    missing = [d["id"] for d in subset if d["id"] not in cache]
    if missing:
        print(f"WARNING: {len(missing)} ids missing from cache (writing empty responses for those)")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        writer.writerow(["id", "response"])
        for item in subset:
            writer.writerow([item["id"], cache.get(item["id"], "")])
    print(f"Wrote {len(subset)} rows to {out_path}")

    return str(out_path)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CSE 151B Math Reasoning Competition — v2.9.3 inference pipeline",
    )
    parser.add_argument("--data", default="data/private.jsonl",
                        help="JSONL input (default: data/private.jsonl)")
    parser.add_argument("--output", default="submission.csv",
                        help="CSV output path (default: submission.csv)")
    parser.add_argument("--cache-dir", default="results/cache",
                        help="Per-batch cache directory (default: results/cache)")
    parser.add_argument("--gpu", default="0",
                        help="CUDA_VISIBLE_DEVICES value (default: 0)")
    parser.add_argument("--max-model-len", type=int, default=90112,
                        help="vLLM max_model_len; lower if OOM (default: 90112)")
    parser.add_argument("--max-num-seqs", type=int, default=32,
                        help="vLLM max_num_seqs; lower if OOM (default: 32)")
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384,
                        help="vLLM max_num_batched_tokens (default: 16384)")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Disable CUDA graphs (~10%% slower but more memory-stable)")
    parser.add_argument("--no-olympiad-sc", action="store_true",
                        help="Disable olympiad multi-temperature SC (~3x speedup, ~3-5pp accuracy cost on olympiad items)")
    args = parser.parse_args()

    run_inference(
        data_path=args.data,
        output_path=args.output,
        cache_dir=args.cache_dir,
        gpu_id=args.gpu,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=args.enforce_eager,
        enable_olympiad_sc=not args.no_olympiad_sc,
    )
