"""
Preprocesamiento y feature engineering para clustering.

Decisiones técnicas justificadas en el informe (sección de EDA):
  - Variables de conteo (hits, pageviews, n_sessions) son right-skewed
    extremas (skew > 5, kurtosis > 60). Aplicamos clip al P99 y log1p
    para que las distancias euclidianas del clustering no estén dominadas
    por outliers.
  - hits y pageviews tienen Spearman = 0.97 y VIF > 27. Conservamos las
    dos pero derivamos hits_per_pageview como medida de interacción
    no-página.
  - channelGrouping y trafficSource.medium tienen Cramer's V = 0.98.
    Eliminamos trafficSource.medium por redundancia.
  - Categorías raras de browser y SO se agrupan en 'Other' para evitar
    columnas one-hot casi vacías.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


NUM_FEATURES = [
    'log_n_sessions', 'log_totals.hits',
    'log_pageviews_per_session', 'log_hits_per_session',
    'weekend_prop', 'bounce_prop', 'device.isMobile', 'hits_per_pageview',
]
CAT_FEATURES = [
    'channelGrouping', 'device.deviceCategory', 'part_of_day',
    'browser_top', 'os_top',
]
BROWSERS_TOP = ['Chrome', 'Safari', 'Firefox']
OS_TOP = ['Macintosh', 'Windows', 'iOS', 'Android']


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Construye las features derivadas que el clustering consume."""
    df = df.copy()
    df['hits_per_session'] = df['totals.hits'] / df['n_sessions']
    df['pageviews_per_session'] = df['totals.pageviews'] / df['n_sessions']
    df['hits_per_pageview'] = df['totals.hits'] / df['totals.pageviews']
    df['part_of_day'] = pd.cut(
        df['hour'], bins=[-0.1, 6, 12, 18, 23.1],
        labels=['madrugada', 'manana', 'tarde', 'noche']
    )
    for col in ['n_sessions', 'totals.hits', 'totals.pageviews',
                'hits_per_session', 'pageviews_per_session']:
        clipped = np.minimum(df[col], df[col].quantile(0.99))
        df[f'log_{col}'] = np.log1p(clipped)
    df['browser_top'] = df['device.browser'].where(
        df['device.browser'].isin(BROWSERS_TOP), 'Other'
    )
    df['os_top'] = df['device.operatingSystem'].where(
        df['device.operatingSystem'].isin(OS_TOP), 'Other'
    )
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[np.ndarray, StandardScaler, list[str]]:
    """Construye la matriz X final escalada y la lista de nombres de columnas."""
    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[NUM_FEATURES])
    X_cat = pd.get_dummies(df[CAT_FEATURES].astype(str), drop_first=False)
    cat_cols = X_cat.columns.tolist()
    X = np.hstack([X_num, X_cat.values.astype(float)])
    feature_names = NUM_FEATURES + cat_cols
    return X, scaler, feature_names
