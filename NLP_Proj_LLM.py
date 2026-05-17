import re
import json
import time
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from typing import Optional
from datetime import datetime
from dataclasses import dataclass, field

# ── CLI args ──────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Lexical simplification benchmark")
    parser.add_argument("-b", "--benchls", type=str, default="benchls.txt")
    parser.add_argument("-d", "--delay", type=float, default=0)
    parser.add_argument("-m", "--models", type=str, nargs="+", default=["llama3.1:8b"])
    parser.add_argument(
        "-v", "--variant", type=str, choices=["v1", "v2", "v3"], default="v1"
    )
    return parser.parse_args()


args = parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
TEMPERATURE = 0.0
MAX_TOKENS = 1024
RETRY_LIMIT = 3
RETRY_DELAY = 0.5

# ── Logging ───────────────────────────────────────────────────────────────────


def setup_logging(model_name: str) -> tuple[logging.Logger, str]:
    safe_model = model_name.replace("/", "_").replace(":", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"eval_{safe_model}_{ts}.log"

    logger = logging.getLogger(f"lexsimpl.{safe_model}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s"))

    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Run started — model: {model_name} | log: {log_file}")
    return logger, log_file


# ── Client ────────────────────────────────────────────────────────────────────

client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")

# ── Prompts ───────────────────────────────────────────────────────────────────

SYSTEM_ZERO_SHOT_V2 = """You are a lexical simplification expert.

Your job: replace a complex word with a SIMPLER, MORE COMMON single word that fits naturally in the same sentence.

Strict rules:
- Each candidate must be exactly ONE word (no hyphens, no phrases).
- Each candidate must preserve the original sentence meaning when swapped in.
- Each candidate must match the grammatical form of the target (tense, number, part of speech).
- Rank candidates from most to least suitable substitute.
- Do NOT include the target word itself, proper nouns, or technical terms.

Output ONLY this JSON, nothing else:
{"candidates": ["word1", "word2", "word3", "word4", "word5"]}"""

FEW_SHOT_EXAMPLES = [
    {
        "sentence": "The physician examined the patient carefully.",
        "target_word": "physician",
        "candidates": ["doctor", "medic", "healer", "practitioner", "clinician"],
    },
    {
        "sentence": "She was unable to comprehend the complex instructions.",
        "target_word": "comprehend",
        "candidates": ["understand", "grasp", "follow", "process", "absorb"],
    },
    {
        "sentence": "The legislation was passed by a narrow majority.",
        "target_word": "legislation",
        "candidates": ["law", "bill", "rule", "act", "statute"],
    },
]


def _format_few_shot_examples() -> str:
    blocks = []
    for ex in FEW_SHOT_EXAMPLES:
        block = (
            f'Sentence: "{ex["sentence"]}"\n'
            f'Target word: "{ex["target_word"]}"\n'
            f'Output: {json.dumps({"candidates": ex["candidates"]})}'
        )
        blocks.append(block)
    return "\n\n".join(blocks)


SYSTEM_FEW_SHOT_V2 = f"""You are a lexical simplification expert.

Your job: replace a complex word with a SIMPLER, MORE COMMON single word that fits naturally in the same sentence.

Strict rules:
- Each candidate must be exactly ONE word (no hyphens, no phrases).
- Each candidate must preserve the original sentence meaning when swapped in.
- Each candidate must match the grammatical form of the target (tense, number, part of speech).
- Rank candidates from most to least suitable substitute.
- Do NOT include the target word itself, proper nouns, or technical terms.

Examples:
{_format_few_shot_examples()}

Output ONLY this JSON, nothing else:
{{"candidates": ["word1", "word2", "word3", "word4", "word5"]}}"""

SYSTEM_COT_V2 = """You are a lexical simplification expert.

Your job: replace a complex word with a SIMPLER, MORE COMMON single word.

Think step by step:
1. What does the target word mean in THIS sentence? (not in general)
2. What grammatical form is it? (noun/verb tense/adjective/etc.)
3. List 5 single-word substitutes that: (a) are simpler/more common, (b) keep the sentence meaning intact, (c) match the grammatical form exactly.
4. Put the best candidate FIRST.

Output ONLY valid JSON, nothing after it:
{"reasoning": "<your step-by-step thinking>", "candidates": ["word1", "word2", "word3", "word4", "word5"]}"""


def prompt_user(sentence: str, target_word: str) -> str:
    return (
        f'Sentence: "{sentence}"\n'
        f'Target word: "{target_word}"\n'
        f"Provide 5 simpler single-word substitutes that preserve the sentence meaning."
    )


SYSTEM_ZERO_SHOT = """You are an expert in lexical simplification.

Rules:
- Return ONLY the JSON object, nothing else.
- Each candidate must be a SINGLE word (no phrases).
- Each candidate must be simpler and more common than the target word.
- Each candidate must fit grammatically (same tense, number, part of speech).
- Do NOT include the target word itself.
- Do NOT include proper nouns.

Output schema (exactly):
{"candidates": ["word1", "word2", "word3", "word4", "word5"]}"""


def prompt_zero_shot(sentence: str, target_word: str) -> str:
    return (
        f'Sentence: "{sentence}"\n'
        f'Target word: "{target_word}"\n'
        f"Provide 5 simpler substitutes for the target word."
    )


FEW_SHOT_EXAMPLES = [
    {
        "sentence": "The physician examined the patient carefully.",
        "target_word": "physician",
        "candidates": ["doctor", "medic", "healer", "practitioner", "clinician"],
    },
    {
        "sentence": "She was unable to comprehend the complex instructions.",
        "target_word": "comprehend",
        "candidates": ["understand", "grasp", "follow", "process", "absorb"],
    },
    {
        "sentence": "The legislation was passed by a narrow majority.",
        "target_word": "legislation",
        "candidates": ["law", "bill", "rule", "act", "statute"],
    },
]


def _format_few_shot_examples() -> str:
    blocks = []
    for ex in FEW_SHOT_EXAMPLES:
        block = (
            f'Sentence: "{ex["sentence"]}"\n'
            f'Target word: "{ex["target_word"]}"\n'
            f'Output: {json.dumps({"candidates": ex["candidates"]})}'
        )
        blocks.append(block)
    return "\n\n".join(blocks)


SYSTEM_FEW_SHOT = f"""{SYSTEM_ZERO_SHOT}

Examples:
{_format_few_shot_examples()}"""


def prompt_few_shot(sentence: str, target_word: str) -> str:
    return prompt_zero_shot(sentence, target_word)


SYSTEM_COT = """You are an expert in lexical simplification.
Your task is to suggest simpler synonyms for a given target word in context.

Think step by step:
1. Identify the meaning and grammatical form of the target word in this sentence.
2. List potential simpler synonyms that preserve meaning and grammar.
3. Rank them from simplest to most complex.
4. Return ONLY valid JSON in this exact schema — no extra text after it:

{"reasoning": "<your reasoning>",
 "candidates": ["word1", "word2", "word3", "word4", "word5"]}"""


def prompt_cot(sentence: str, target_word: str) -> str:
    return prompt_zero_shot(sentence, target_word)


# ── Single-word variants (top-1 only) ────────────────────────────────────────

SYSTEM_ZERO_SHOT_TOP1 = """You are a lexical simplification expert.

Your job: replace a complex word with the single BEST, SIMPLEST, most common word that fits naturally in the same sentence.

Rules:
- Return exactly ONE word — the most suitable substitute.
- It must be simpler and more common than the target word.
- It must preserve the original sentence meaning when swapped in.
- It must match the grammatical form of the target (tense, number, part of speech).
- Do NOT return the target word itself, proper replacement, or technical terms.

Output ONLY this JSON, nothing else:
{"candidate": "word"}"""


SYSTEM_FEW_SHOT_TOP1 = f"""You are a lexical simplification expert.

Your job: replace a complex word with the single BEST, SIMPLEST, most common word that fits naturally in the same sentence.

Rules:
- Return exactly ONE word — the most suitable substitute.
- It must be simpler and more common than the target word.
- It must preserve the original sentence meaning when swapped in.
- It must match the grammatical form of the target (tense, number, part of speech).
- Do NOT return the target word itself, proper replacement, or technical terms.

Examples:
Sentence: "The physician examined the patient carefully."
Target word: "physician"
Output: {{"candidate": "doctor"}}

Sentence: "She was unable to comprehend the complex instructions."
Target word: "comprehend"
Output: {{"candidate": "understand"}}

Sentence: "The legislation was passed by a narrow majority."
Target word: "legislation"
Output: {{"candidate": "law"}}

Output ONLY this JSON, nothing else:
{{"candidate": "word"}}"""


SYSTEM_COT_TOP1 = """You are a lexical simplification expert.

Your job: replace a complex word with the single BEST, SIMPLEST word.

Think step by step:
1. What does the target word mean in THIS sentence?
2. What grammatical form is it? (noun/verb tense/adjective/etc.)
3. What is the most common, everyday word that could replace it while keeping the meaning identical?

Output ONLY valid JSON, nothing after it:
{"candidate": "word"}"""


def prompt_top1(sentence: str, target_word: str) -> str:
    return (
        f'Sentence: "{sentence}"\n'
        f'Target word: "{target_word}"\n'
        f"Provide the single best simpler word that preserves the sentence meaning."
    )


variants_v1 = {
    "zero_shot": (SYSTEM_ZERO_SHOT, prompt_zero_shot),
    "few_shot": (SYSTEM_FEW_SHOT, prompt_few_shot),
    "cot": (SYSTEM_COT, prompt_cot),
}

variants_v2 = {
    "zero_shot_v2": (SYSTEM_ZERO_SHOT_V2, prompt_user),
    "few_shot_v2": (SYSTEM_FEW_SHOT_V2, prompt_user),
    "cot_v2": (SYSTEM_COT_V2, prompt_user),
}

variants_v3 = {
    "zero_shot_v3": (SYSTEM_ZERO_SHOT_TOP1, prompt_top1),
    "few_shot_v3": (SYSTEM_FEW_SHOT_TOP1, prompt_top1),
    "cot_v3": (SYSTEM_COT_TOP1, prompt_top1),
}

VARIANTS = {}
if args.variant == "v1":
    VARIANTS = variants_v1
elif args.variant == "v2":
    VARIANTS = variants_v2
elif args.variant == "v3":
    VARIANTS = variants_v3

# ── LLM call ──────────────────────────────────────────────────────────────────


def call_llm(
    system_prompt: str, user_prompt: str, model_name: str, logger: logging.Logger
) -> str:
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = client.chat.completions.create(
                model=model_name,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.warning(f"LLM call failed (attempt {attempt}/{RETRY_LIMIT}): {e}")
            if attempt == RETRY_LIMIT:
                raise
            time.sleep(RETRY_DELAY)


def parse_candidates(raw: str) -> list[str]:
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group())
        if "candidate" in data:
            word = data["candidate"]
            return [word.strip().lower()] if isinstance(word, str) else []
        candidates = data.get("candidates", [])
        return [c.strip().lower() for c in candidates if isinstance(c, str)]
    except json.JSONDecodeError:
        return []


def is_rate_limit_error(e: Exception) -> bool:
    msg = str(e).lower()
    return any(
        k in msg
        for k in ["rate limit", "too many requests", "429", "quota", "throttle"]
    )


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class PredictionResult:
    sentence: str
    target_word: str
    variant: str
    candidates: list[str] = field(default_factory=list)
    top1: str = ""
    raw_response: str = ""
    error: Optional[str] = None


# ── Predict ───────────────────────────────────────────────────────────────────


def predict(
    sentence: str,
    target_word: str,
    variant: str,
    model_name: str,
    logger: logging.Logger,
) -> PredictionResult:
    system_prompt, user_prompt_fn = VARIANTS[variant]
    user_prompt = user_prompt_fn(sentence, target_word)
    result = PredictionResult(
        sentence=sentence, target_word=target_word, variant=variant
    )

    logger.debug(f"[{variant}] Predicting for: '{target_word}'")
    logger.debug(f"[{variant}] User prompt:\n{user_prompt}")

    try:
        raw = call_llm(system_prompt, user_prompt, model_name, logger)
        result.raw_response = raw
        result.candidates = parse_candidates(raw)
        result.top1 = result.candidates[0] if result.candidates else ""
        logger.debug(f"[{variant}] Raw response: {raw}")
        logger.debug(f"[{variant}] Parsed candidates: {result.candidates}")
    except Exception as e:
        result.error = str(e)
        logger.error(f"[{variant}] Error for '{target_word}': {e}")

    return result


# ── BenchLS loader ────────────────────────────────────────────────────────────


def load_benchls(path: str) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            sentence = parts[0]
            target_word = parts[1]
            ref_subs = [
                re.sub(r"^\d+:", "", c).strip().lower() for c in parts[3:] if c.strip()
            ]
            rows.append(
                {
                    "sentence": sentence,
                    "target_word": target_word,
                    "references": ref_subs,
                }
            )
    return pd.DataFrame(rows)


# ── Evaluate ──────────────────────────────────────────────────────────────────


def evaluate(
    df: pd.DataFrame,
    variant: str,
    model_name: str,
    logger: logging.Logger,
    delay: float = 0.001,
    max_delay: float = 10.0,
    backoff_factor: float = 2.0,
    recovery: float = 0.9,
) -> dict:
    results, hits = [], 0
    current_delay = delay

    logger.info(
        f"Starting evaluation | variant={variant} | rows={len(df)} | model={model_name}"
    )

    pbar = tqdm(
        df.iterrows(),
        total=len(df),
        desc=f"{variant}",
        unit="row",
        dynamic_ncols=True,
        colour="cyan",
    )

    for n, (_, row) in enumerate(pbar):
        pred = predict(row["sentence"], row["target_word"], variant, model_name, logger)

        if pred.error and is_rate_limit_error(pred.error):
            current_delay = min(current_delay * backoff_factor, max_delay)
            logger.warning(
                f"Rate limit suspected → delay increased to {current_delay:.2f}s"
            )
        else:
            current_delay = max(delay, current_delay * recovery)

        hit = int(pred.top1 in row["references"])
        hits += hit
        p_at_1 = hits / (n + 1)

        logger.info(
            f"[{variant}] [{n+1:>3}/{len(df)}] "
            f"{row['target_word']:<15} → {pred.top1:<15} | "
            f"refs: {row['references']} | hit: {hit} | P@1: {p_at_1:.3f}"
        )
        pbar.set_postfix(
            {
                "target": row["target_word"],
                "top1": pred.top1 or "—",
                "hit": "✓" if hit else "✗",
                "P@1": f"{p_at_1:.3f}",
            }
        )

        results.append({**pred.__dict__, "references": row["references"], "hit": hit})
        time.sleep(current_delay)

    pbar.close()
    precision_at_1 = hits / len(df) if len(df) > 0 else 0.0
    logger.info(
        f"Done | variant={variant} | P@1={precision_at_1:.4f} ({hits}/{len(df)})"
    )

    return {
        "variant": variant,
        "precision@1": precision_at_1,
        "hits": hits,
        "total": len(df),
        "details": results,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

df = load_benchls(args.benchls)

all_summaries = []

for model_name in args.models:
    logger, log_file = setup_logging(model_name)

    print(f"\n{'═'*52}")
    print(f"  Model : {model_name}")
    print(f"  Log   : {log_file}")
    print(f"  Data  : {args.benchls} ({len(df)} rows)")
    print(f"{'═'*52}\n")

    logger.info(f"Loaded {len(df)} rows from {args.benchls}")

    summary_rows = []
    for variant in VARIANTS:
        print(f"\n── Variant: {variant} ──")
        metrics = evaluate(df, variant, model_name, logger, delay=args.delay)
        p1 = metrics["precision@1"]
        print(f"   Precision@1 = {p1:.4f}  ({metrics['hits']}/{metrics['total']})\n")
        summary_rows.append(
            {"Model": model_name, "Variant": variant, "Precision@1": round(p1, 4)}
        )

    summary = pd.DataFrame(summary_rows).sort_values("Precision@1", ascending=False)
    print(f"\n══ Summary — {model_name} ══")
    print(summary[["Variant", "Precision@1"]].to_string(index=False))
    logger.info(f"Final summary:\n{summary.to_string(index=False)}")

    all_summaries.append(summary)

# Cross-model comparison
if len(all_summaries) > 1:
    combined = pd.concat(all_summaries).sort_values(
        ["Variant", "Precision@1"], ascending=[True, False]
    )
    print("\n\n══ Cross-model comparison ══════════════════════════")
    print(combined.to_string(index=False))
