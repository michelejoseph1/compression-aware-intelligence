from __future__ import annotations
from typing import List
import numpy as np

# Default embedding uses TF-IDF to keep the package offline-friendly.
# You can plug in OpenAI or HuggingFace embeddings by passing a custom embed() to CFI.
def embed_texts(texts: List[str]) -> np.ndarray:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as e:
        raise ImportError("scikit-learn is required for default embeddings. Install with `pip install scikit-learn`.") from e
    vec = TfidfVectorizer(max_features=2048, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    return X.toarray().astype(np.float32)

def pca_explained_variance_ratio(X: np.ndarray, n_components: int = 1):
    from sklearn.decomposition import PCA
    n = min(n_components, min(X.shape[0], X.shape[1]))
    pca = PCA(n_components=n, random_state=0)
    pca.fit(X)
    evr = pca.explained_variance_ratio_
    if n_components == 1 and np.isscalar(evr):
        return np.array([float(evr)])
    return evr