from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional, Dict, Any
import numpy as np

from .latent import embed_texts, pca_explained_variance_ratio
from .perturb import make_neighborhood, perturb_grid
from .visualize import plot_coherence_heatmap

@dataclass
class CFIResult:
    score: float
    components: Dict[str, float]
    variants: List[str]
    outputs: List[str]
    heatmap: Any  # matplotlib Figure
    info: Dict[str, Any]

class CFI:
    """
    Coherence Field Index (CFI)
    ---------------------------------
    A geometric stability measure for LLM inference.
    Approximates internal narrative stability by measuring curvature
    in the output embedding manifold across small prompt perturbations.

    Works with any backend that exposes a callable: generate(prompt) -> text.
    If you don't have a backend yet, pass a lambda that returns a stub string
    to exercise the API.
    """
    def __init__(
        self,
        generate: Callable[[str], str],
        embed: Optional[Callable[[List[str]], np.ndarray]] = None,
        random_state: int = 42,
    ):
        self.generate = generate
        self.embed = embed or embed_texts
        self.rs = np.random.RandomState(random_state)

    def measure(
        self,
        prompt: str,
        k: int = 8,
        grid_size: int = 6,
        return_heatmap: bool = True,
    ) -> Tuple[float, CFIResult]:
        """
        Compute CFI for the given prompt.

        k: number of neighborhood lexical variants
        grid_size: size of 2D perturbation grid for visualization
        """
        # 1) Neighborhood variants (1D random small edits)
        variants = make_neighborhood(prompt, k=k, rng=self.rs)
        variants = [prompt] + variants

        # 2) Generate outputs
        outputs = [self.generate(v) for v in variants]

        # 3) Embeddings
        X = self.embed(outputs)  # shape [n, d]
        if X.ndim == 1:
            X = X[:, None]

        # 4) Curvature proxy: 1 - variance explained by first principal component
        evr = pca_explained_variance_ratio(X, n_components=1)
        residual = float(1.0 - evr[0])
        residual = float(max(0.0, min(1.0, residual)))  # clamp to [0,1]

        # 5) Local spread (normalized pairwise distance)
        # use cosine distance on normalized embeddings
        def _cosine_dist(a, b):
            denom = (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
            return 1.0 - float(np.dot(a, b) / denom)
        dists = []
        for i in range(len(X)):
            for j in range(i+1, len(X)):
                dists.append(_cosine_dist(X[i], X[j]))
        spread = float(np.mean(dists)) if dists else 0.0
        spread = float(min(1.0, spread / 0.6))  # cap and normalize

        # 6) Disagreement entropy: cluster via sign of projection on PC1 as a simple proxy
        # (two-bucket semantic clustering for a minimal footprint)
        from sklearn.decomposition import PCA
        if len(X) >= 2:
            pca = PCA(n_components=1, random_state=0)
            proj = pca.fit_transform(X).ravel()
            labels = (proj > np.median(proj)).astype(int)
            counts = np.bincount(labels, minlength=2).astype(float)
            p = counts / counts.sum()
            entropy = float(-(p * np.log(p + 1e-9)).sum() / np.log(2))  # normalize by log2
        else:
            entropy = 0.0

        # Normalize entropy to [0,1] since max with 2 buckets is 1
        entropy = float(max(0.0, min(1.0, entropy)))

        # 7) Compose CFI: a convex combination dominated by curvature residual
        # This keeps CFI focused and distinct from CTS.
        cfi = float(0.7 * residual + 0.2 * spread + 0.1 * entropy)
        cfi = float(max(0.0, min(1.0, cfi)))

        # 8) Optional heatmap over a 2D perturbation grid
        fig = None
        if return_heatmap:
            grid_prompts, (ax1, ax2) = perturb_grid(prompt, grid_size=grid_size, rng=self.rs)
            grid_outputs = [self.generate(p) for p in grid_prompts]
            GX = self.embed(grid_outputs)
            # project to 2D for plotting
            from sklearn.decomposition import PCA
            if GX.ndim == 1:
                GX = GX[:, None]
            p = PCA(n_components=2, random_state=0).fit_transform(GX)
            # compute local curvature proxy per cell: 1 - EVR1 in a small window
            # here we approximate by local variance around the centroid per cell index
            # build a grid of scores by grouping consecutive chunks
            cells = []
            step = max(1, len(grid_prompts) // (grid_size * grid_size))
            for g in range(0, len(grid_prompts), step):
                window = GX[g:g+step]
                if len(window) < 2:
                    cells.append(0.0)
                    continue
                evr_local = pca_explained_variance_ratio(window, n_components=1)[0]
                cells.append(float(1.0 - evr_local))
            # pad or trim to grid_size^2
            total = grid_size * grid_size
            if len(cells) < total:
                cells += [cells[-1] if cells else 0.0] * (total - len(cells))
            elif len(cells) > total:
                cells = cells[:total]
            grid = np.array(cells, dtype=float).reshape(grid_size, grid_size)
            fig = plot_coherence_heatmap(grid, title="Coherence Field Curvature")

        components = {
            "residual_curvature": float(residual),
            "spread": float(spread),
            "entropy": float(entropy),
        }
        info = {"k": k, "grid_size": grid_size}
        result = CFIResult(score=cfi, components=components, variants=variants, outputs=outputs, heatmap=fig, info=info)
        return cfi, result