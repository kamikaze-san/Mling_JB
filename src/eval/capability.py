"""
Capability evaluation: FLORES-200 translation (chrF++) and Belebele reading
comprehension (accuracy). Model-agnostic: every function takes a
`chat(messages) -> str` callable, so the same code scores the GGUF base, a
GGUF + LoRA checkpoint, or anything else.

"Not dumber" is reported as Benign Capability Preservation:
    BCP = score_repaired / score_base   (per direction / per language)
"""

import re
from typing import Callable, Dict, List, Optional

LANG_NAMES = {"en": "English", "hi": "Hindi", "te": "Telugu", "ja": "Japanese"}
DEFAULT_DIRECTIONS = [("en", "hi"), ("en", "te"), ("en", "ja"),
                      ("hi", "en"), ("te", "en"), ("ja", "en")]

Chat = Callable[[List[Dict[str, str]]], str]


# ---------------------------------------------------------------------------
# FLORES translation
# ---------------------------------------------------------------------------

def translation_prompt(src_text: str, src: str, tgt: str) -> str:
    return (f"Translate the following {LANG_NAMES[src]} text into {LANG_NAMES[tgt]}. "
            f"Output only the {LANG_NAMES[tgt]} translation.\n\n{src_text}")


_PREFIX = re.compile(r"^\s*(translation|अनुवाद|అనువాదం|翻訳)\s*[:：]\s*", re.IGNORECASE)


def clean_translation(text: str) -> str:
    """Strip common wrappers (quotes, 'Translation:' prefixes); keep the first paragraph."""
    t = (text or "").strip()
    t = _PREFIX.sub("", t)
    t = t.split("\n\n")[0].strip()
    if len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'「『":
        t = t[1:-1].strip()
    return t


def chrf_pp(hypotheses: List[str], references: List[str]) -> float:
    """Corpus chrF++ (sacrebleu CHRF with word_order=2)."""
    from sacrebleu.metrics import CHRF
    return CHRF(word_order=2).corpus_score(hypotheses, [references]).score


def eval_flores(chat: Chat, rows: List[Dict], directions=DEFAULT_DIRECTIONS) -> Dict[str, Dict]:
    """rows: aligned FLORES rows {'id', 'en', 'hi', 'te', 'ja'}. Returns per-direction chrF++."""
    results = {}
    for src, tgt in directions:
        hyps, refs = [], []
        for r in rows:
            out = chat([{"role": "user", "content": translation_prompt(r[src], src, tgt)}])
            hyps.append(clean_translation(out))
            refs.append(r[tgt])
        results[f"{src}-{tgt}"] = {"n": len(rows), "chrf_pp": round(chrf_pp(hyps, refs), 2),
                                   "hypotheses": hyps}
    return results


# ---------------------------------------------------------------------------
# Belebele
# ---------------------------------------------------------------------------

def belebele_prompt(rec: Dict) -> str:
    opts = "\n".join(f"{i}. {o}" for i, o in enumerate(rec["options"], start=1))
    return (f"{rec['passage']}\n\nQuestion: {rec['question']}\n\n{opts}\n\n"
            "Answer with the number (1, 2, 3, or 4) of the correct option only.")


_DIGIT = re.compile(r"[1-4１-４]")
_FULLWIDTH = {"１": "1", "２": "2", "３": "3", "４": "4"}


def parse_choice(text: str, options: Optional[List[str]] = None) -> Optional[int]:
    """First option number 1-4 in the reply; falls back to exact option-text match."""
    t = (text or "").strip()
    m = _DIGIT.search(t)
    if m:
        return int(_FULLWIDTH.get(m.group(0), m.group(0)))
    if options:
        for i, o in enumerate(options, start=1):
            if o and o.strip() and o.strip() in t:
                return i
    return None


def eval_belebele(chat: Chat, records: List[Dict]) -> Dict[str, Dict]:
    by_lang: Dict[str, Dict] = {}
    for rec in records:
        out = chat([{"role": "user", "content": belebele_prompt(rec)}])
        pred = parse_choice(out, rec["options"])
        d = by_lang.setdefault(rec["language"], {"n": 0, "correct": 0, "unparsed": 0})
        d["n"] += 1
        d["correct"] += int(pred == rec["correct_answer_num"])
        d["unparsed"] += int(pred is None)
    for d in by_lang.values():
        d["accuracy"] = round(d["correct"] / d["n"], 4) if d["n"] else 0.0
    return by_lang


# ---------------------------------------------------------------------------
# Preservation
# ---------------------------------------------------------------------------

def preservation(base: Dict[str, float], repaired: Dict[str, float]) -> Dict[str, Optional[float]]:
    """BCP per key = repaired / base (None when base is 0 or missing)."""
    out = {}
    for k, b in base.items():
        r = repaired.get(k)
        out[k] = round(r / b, 4) if (r is not None and b) else None
    return out
