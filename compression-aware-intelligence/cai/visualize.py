from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

def plot_coherence_heatmap(grid: np.ndarray, title: str = "Coherence Field Curvature"):
    fig, ax = plt.subplots(figsize=(4,3.5), dpi=160)
    im = ax.imshow(grid, origin="lower", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Axis 2")
    ax.set_ylabel("Axis 1")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig