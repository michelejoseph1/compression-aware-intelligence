# cai / coherence field index (cfi)

a small library for probing llm coherence stability through prompt perturbations.

## install
pip install -U scikit-learn matplotlib
pip install .

## usage
from cai.cfi import CFI

def generate(prompt: str) -> str:
    return f"answer: {prompt}"

cfi = CFI(generate=generate)
score, res = cfi.measure("explain mass vs weight.", k=8, grid_size=6)
