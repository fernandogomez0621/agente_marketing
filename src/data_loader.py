"""
Carga y validación inicial del dataset de segmentación.

El dataset proviene de Google Analytics y viene pre-agregado a nivel
visitante: una fila por fullVisitorId. La columna sessionId corresponde
al número de sesiones del visitante, no a un identificador.
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path


REQUIRED_COLUMNS = [
    'fullVisitorId', 'channelGrouping', 'weekend_prop', 'hour',
    'sessionId', 'device.browser', 'device.deviceCategory',
    'device.isMobile', 'device.operatingSystem',
    'totals.hits', 'totals.pageviews', 'bounce_prop',
    'trafficSource.medium',
]


def load_raw(path: str | Path) -> pd.DataFrame:
    """Carga el CSV original y renombra sessionId a n_sessions."""
    df = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes en el dataset: {missing}")
    return df.rename(columns={'sessionId': 'n_sessions'})


def validate_schema(df: pd.DataFrame) -> dict:
    """Valida coherencia lógica del dataset y retorna un reporte."""
    report = {
        'n_filas': len(df),
        'n_visitantes_unicos': df['fullVisitorId'].nunique(),
        'una_fila_por_visitante': df['fullVisitorId'].is_unique,
        'duplicados_totales': int(df.duplicated().sum()),
        'nulos_por_columna': df.isnull().sum().to_dict(),
        'pageviews_menor_igual_hits': bool((df['totals.pageviews'] <= df['totals.hits']).all()),
        'bounce_en_rango': bool(df['bounce_prop'].between(0, 1).all()),
        'weekend_en_rango': bool(df['weekend_prop'].between(0, 1).all()),
        'hour_en_rango': bool(df['hour'].between(0, 23).all()),
        'sesiones_positivas': bool((df['n_sessions'] >= 1).all()),
    }
    return report
