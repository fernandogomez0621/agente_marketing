"""
Entrenamiento y selección del modelo de clustering.

Algoritmo final: KMeans con K=4.
Justificación en el informe:
  - Métricas internas (silhouette, Calinski-Harabasz, Davies-Bouldin)
    favorecen K entre 3 y 4. Elegimos 4 porque separa el segmento
    mobile que con K=3 queda mezclado.
  - Estabilidad: ARI ≈ 1.0 entre semillas distintas.
  - Bootstrap stability con 30 muestras del 80%: ARI medio = 0.97.
  - Hold-out 80/20: gap train-test = 0.000 (no overfitting).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score, calinski_harabasz_score,
                              davies_bouldin_score, adjusted_rand_score)


CLUSTER_NAMES = {
    0: 'Desktop Engaged',
    1: 'Desktop Frecuente',
    2: 'Weekend Warriors',
    3: 'Mobile Users',
}


def select_k(X: np.ndarray, k_range=range(2, 9), random_state: int = 42,
             sample_size: int = 5000) -> pd.DataFrame:
    """Evalúa KMeans para distintos K y retorna métricas internas."""
    rng = np.random.RandomState(random_state)
    sample_idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=20, random_state=random_state).fit(X)
        rows.append({
            'k': k,
            'inertia': km.inertia_,
            'silhouette': silhouette_score(X[sample_idx], km.labels_[sample_idx]),
            'calinski_harabasz': calinski_harabasz_score(X, km.labels_),
            'davies_bouldin': davies_bouldin_score(X, km.labels_),
        })
    return pd.DataFrame(rows)


def fit_final_model(X: np.ndarray, k: int = 4, random_state: int = 42) -> KMeans:
    """Entrena el modelo final con n_init alto para estabilidad."""
    return KMeans(n_clusters=k, n_init=50, random_state=random_state).fit(X)


def benchmark_algorithms(X: np.ndarray, k: int = 4,
                         random_state: int = 42, sample_size: int = 5000) -> pd.DataFrame:
    """Compara KMeans, Ward y GMM para el mismo K."""
    rng = np.random.RandomState(random_state)
    sample_idx = rng.choice(len(X), size=min(sample_size, len(X)), replace=False)
    rows = []
    models = {
        'KMeans': KMeans(n_clusters=k, n_init=50, random_state=random_state).fit(X).labels_,
        'AgglomerativeWard': AgglomerativeClustering(n_clusters=k, linkage='ward').fit(X).labels_,
        'GaussianMixture': GaussianMixture(n_components=k, n_init=5,
                                            random_state=random_state).fit(X).predict(X),
    }
    for name, labels in models.items():
        rows.append({
            'modelo': name,
            'n_clusters': len(set(labels)),
            'silhouette': silhouette_score(X[sample_idx], labels[sample_idx]),
            'calinski_harabasz': calinski_harabasz_score(X, labels),
            'davies_bouldin': davies_bouldin_score(X, labels),
        })
    return pd.DataFrame(rows)


def validate_stability(X: np.ndarray, base_labels: np.ndarray,
                        n_bootstrap: int = 30, k: int = 4) -> dict:
    """Bootstrap stability: ARI promedio sobre muestras del 80%."""
    rng = np.random.RandomState(42)
    aris = []
    for i in range(n_bootstrap):
        idx = rng.choice(len(X), size=int(0.8 * len(X)), replace=False)
        km = KMeans(n_clusters=k, n_init=20, random_state=i).fit(X[idx])
        labels_full = km.predict(X)
        aris.append(adjusted_rand_score(base_labels, labels_full))
    return {
        'ari_medio': float(np.mean(aris)),
        'ari_mediana': float(np.median(aris)),
        'ari_min': float(np.min(aris)),
        'ari_p25': float(np.percentile(aris, 25)),
        'ari_p75': float(np.percentile(aris, 75)),
    }


def validate_holdout(X: np.ndarray, k: int = 4, n_folds: int = 10) -> dict:
    """Hold-out 80/20: entrena centroides en 80%, asigna 20%, compara silhouettes."""
    train_sils, test_sils = [], []
    for seed in range(n_folds):
        rng = np.random.RandomState(seed)
        idx_train = rng.choice(len(X), size=int(0.8 * len(X)), replace=False)
        idx_test = np.setdiff1d(np.arange(len(X)), idx_train)
        km = KMeans(n_clusters=k, n_init=30, random_state=42).fit(X[idx_train])
        sample = rng.choice(len(idx_train), 3000, replace=False)
        train_sils.append(silhouette_score(X[idx_train][sample], km.labels_[sample]))
        test_sils.append(silhouette_score(X[idx_test], km.predict(X[idx_test])))
    return {
        'silhouette_train': float(np.mean(train_sils)),
        'silhouette_test': float(np.mean(test_sils)),
        'gap': float(abs(np.mean(train_sils) - np.mean(test_sils))),
    }


def predict(model: KMeans, X: np.ndarray) -> np.ndarray:
    """Asigna cluster a nuevos visitantes."""
    return model.predict(X)
