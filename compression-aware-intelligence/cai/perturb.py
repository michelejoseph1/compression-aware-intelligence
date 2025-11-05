from __future__ import annotations
from typing import List, Tuple
import random
import re

_SOFTENERS = ["briefly", "succinctly", "carefully", "concisely", "with precision"]
_INTENSIFIERS = ["in depth", "thoroughly", "step by step", "at a high level", "formally"]
_SYNONYMS = {
    "explain": ["describe", "clarify", "outline"],
    "difference": ["distinction", "contrast", "gap"],
    "between": ["among", "across"],
    "show": ["demonstrate", "illustrate", "display"],
    "why": ["reason", "cause"],
    "how": ["method", "process"],
}

def _swap_synonyms(text: str, rng) -> str:
    words = text.split()
    idxs = list(range(len(words)))
    rng.shuffle(idxs)
    for i in idxs:
        w = re.sub(r'\W+', '', words[i].lower())
        if w in _SYNONYMS and rng.rand() < 0.4:
            words[i] = _SYNONYMS[w][int(rng.rand() * len(_SYNONYMS[w]))]
            break
    return " ".join(words)

def _add_style(text: str, rng) -> str:
    if rng.rand() < 0.5:
        return f"{text} {rng.choice(_SOFTENERS)}."
    else:
        return f"{text} {rng.choice(_INTENSIFIERS)}."

def _reorder(text: str, rng) -> str:
    if "," in text and rng.rand() < 0.7:
        parts = [p.strip() for p in text.split(",")]
        rng.shuffle(parts)
        return ", ".join(parts)
    return text

def make_neighborhood(prompt: str, k: int = 8, rng=None) -> List[str]:
    rng = rng or random.Random(42)
    variants = []
    for _ in range(k):
        v = prompt
        # apply up to two small edits
        if hasattr(rng, "rand"):
            # numpy RandomState
            if rng.rand() < 0.6: v = _swap_synonyms(v, rng)
            if rng.rand() < 0.6: v = _add_style(v, rng)
            if rng.rand() < 0.4: v = _reorder(v, rng)
        else:
            if rng.random() < 0.6: v = _swap_synonyms(v, rng)
            if rng.random() < 0.6: v = _add_style(v, rng)
            if rng.random() < 0.4: v = _reorder(v, rng)
        variants.append(v)
    # ensure uniqueness
    uniq = []
    seen = set()
    for v in variants:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq

def perturb_grid(prompt: str, grid_size: int = 6, rng=None) -> Tuple[list, Tuple[list, list]]:
    rng = rng or random.Random(42)
    # axis 1: politeness/softness vs assertiveness
    axis1 = ["briefly", "politely", "neutrally", "directly", "strongly", "formally"]
    # axis 2: scope granularity
    axis2 = ["at a high level", "with examples", "step by step", "technically", "for experts", "for beginners"]
    prompts = []
    for a in axis1[:grid_size]:
        for b in axis2[:grid_size]:
            prompts.append(f"{prompt} {a}, {b}.")
    return prompts, (axis1[:grid_size], axis2[:grid_size])