"""
Tools del sistema multi-agente — decorados con @tool de LangChain.

Capas:
  1. Exploración   — perfiles, comparaciones, overview
  2. Estadística   — tests de significancia, drivers
  3. Búsqueda      — filtrado y conteo
  4. Estrategia    — recomendaciones de marketing
"""
from __future__ import annotations
import json, pickle, os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from pathlib import Path
from langchain_core.tools import tool

# ── Rutas ───────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[2]
_PROC = _ROOT / "data" / "processed"
_cache: dict = {}

def _clustered() -> pd.DataFrame:
    if "c" not in _cache:
        _cache["c"] = pd.read_csv(_PROC / "customers_segmentados.csv")
    return _cache["c"]

def _processed() -> pd.DataFrame:
    if "p" not in _cache:
        with open(_PROC / "df_processed.pkl", "rb") as f:
            df = pickle.load(f)
        seg = _clustered()[["fullVisitorId", "cluster", "cluster_name"]]
        _cache["p"] = df.merge(seg, on="fullVisitorId", how="left")
    return _cache["p"]


# ═══ CAPA 1: Exploración ═══════════════════════════════════════════

@tool
def get_cluster_overview() -> str:
    """Tabla resumen de los 4 clusters: nombre, tamaño, porcentaje y métricas clave.
    Punto de entrada para preguntas generales sobre la segmentación."""
    df = _clustered()
    rows = []
    for cid in sorted(df["cluster"].unique()):
        s = df[df["cluster"] == cid]
        rows.append({
            "cluster_id": int(cid), "nombre": s["cluster_name"].iloc[0],
            "n": len(s), "pct": round(len(s)/len(df)*100, 1),
            "hits_avg": round(s["totals.hits"].mean(), 1),
            "pv_avg": round(s["totals.pageviews"].mean(), 1),
            "bounce_avg": round(s["bounce_prop"].mean(), 3),
            "weekend_avg": round(s["weekend_prop"].mean(), 3),
            "sessions_avg": round(s["n_sessions"].mean(), 1),
        })
    return json.dumps(rows, ensure_ascii=False)

@tool
def get_cluster_profile(cluster_id: int) -> str:
    """Perfil completo de un cluster: medias numéricas con z-score vs global,
    y composición categórica top. Úsala para entender la identidad de un cluster."""
    df = _processed()
    s = df[df["cluster"] == cluster_id]
    if len(s) == 0:
        return json.dumps({"error": f"Cluster {cluster_id} no existe"})
    nums = ["n_sessions","totals.hits","totals.pageviews","bounce_prop",
            "weekend_prop","hits_per_session","pageviews_per_session","hour"]
    cats = ["channelGrouping","device.deviceCategory","device.operatingSystem","device.browser"]
    p = {"id": int(cluster_id), "nombre": s["cluster_name"].iloc[0],
         "n": len(s), "pct": round(len(s)/len(df)*100, 1)}
    n_data = {}
    for c in nums:
        if c in s.columns:
            gm, gs, cm = df[c].mean(), df[c].std(), s[c].mean()
            z = (cm - gm) / gs if gs > 0 else 0
            n_data[c] = {"media": round(float(cm),3), "global": round(float(gm),3),
                         "z": round(float(z),2)}
    p["numericas"] = n_data
    c_data = {}
    for c in cats:
        if c in s.columns:
            t = s[c].value_counts(normalize=True).head(5)
            c_data[c] = {k: round(float(v)*100,1) for k,v in t.items()}
    p["categoricas"] = c_data
    return json.dumps(p, ensure_ascii=False)

@tool
def compare_clusters(cluster_a: int, cluster_b: int) -> str:
    """Comparación lado a lado de dos clusters. Retorna las dimensiones
    ordenadas por effect size (Cohen's d). Úsala para preguntas tipo
    '¿en qué se diferencian C0 y C2?'."""
    df = _processed()
    sa, sb = df[df["cluster"]==cluster_a], df[df["cluster"]==cluster_b]
    if len(sa)==0 or len(sb)==0:
        return json.dumps({"error": "Cluster no encontrado"})
    nums = ["n_sessions","totals.hits","totals.pageviews","bounce_prop",
            "weekend_prop","hits_per_session","pageviews_per_session","hour"]
    comp = []
    for c in nums:
        if c in df.columns:
            d = abs(sa[c].mean()-sb[c].mean()) / (df[c].std() or 1)
            comp.append({"var": c, "A": round(float(sa[c].mean()),3),
                        "B": round(float(sb[c].mean()),3), "d": round(float(d),2)})
    comp.sort(key=lambda x: x["d"], reverse=True)
    return json.dumps({"a": {"id":cluster_a,"name":sa["cluster_name"].iloc[0]},
                       "b": {"id":cluster_b,"name":sb["cluster_name"].iloc[0]},
                       "diffs": comp}, ensure_ascii=False)


# ═══ CAPA 2: Estadística ══════════════════════════════════════════

@tool
def test_significance(cluster_a: int, cluster_b: int, variable: str) -> str:
    """Mann-Whitney U entre dos clusters en una variable numérica.
    Retorna p-value, effect size y si es significativo.
    
    Variables numéricas disponibles: n_sessions, totals.hits, totals.pageviews,
    bounce_prop, weekend_prop, hits_per_session, pageviews_per_session,
    hits_per_pageview, hour.
    
    IMPORTANTE: usa 'bounce_prop' para bounce, NO 'bounces' ni 'bounce_rate'."""
    df = _processed()
    if variable not in df.columns:
        similares = [c for c in df.columns if variable.lower() in c.lower() or c.lower() in variable.lower()]
        return json.dumps({"error": f"Variable '{variable}' no existe. "
                          f"Variables disponibles: {[c for c in df.columns if df[c].dtype in ['float64','int64']]}. "
                          f"Quizás quisiste decir: {similares}" if similares else ""})
    ga = df.loc[df["cluster"]==cluster_a, variable].dropna().values
    gb = df.loc[df["cluster"]==cluster_b, variable].dropna().values
    if len(ga)==0 or len(gb)==0:
        return json.dumps({"error": "Sin datos"})
    stat, p = mannwhitneyu(ga, gb, alternative="two-sided")
    r = 1 - (2*stat)/(len(ga)*len(gb))
    mag = "grande" if abs(r)>0.5 else "mediano" if abs(r)>0.3 else "pequeno" if abs(r)>0.1 else "trivial"
    return json.dumps({"var": variable, "a": cluster_a, "b": cluster_b,
                       "p": float(p), "r": round(float(r),4), "mag": mag,
                       "sig": bool(p < 0.05)}, ensure_ascii=False)

@tool
def rank_cluster_drivers(cluster_id: int, top_n: int = 5) -> str:
    """Variables que más distinguen un cluster del resto, por z-score absoluto.
    Identifica la identidad estadística del cluster."""
    df = _processed()
    s = df[df["cluster"]==cluster_id]; rest = df[df["cluster"]!=cluster_id]
    if len(s)==0:
        return json.dumps({"error": f"Cluster {cluster_id} no existe"})
    nums = ["n_sessions","totals.hits","totals.pageviews","bounce_prop",
            "weekend_prop","hits_per_session","pageviews_per_session","hits_per_pageview","hour"]
    drivers = []
    for c in nums:
        if c in df.columns:
            z = (s[c].mean()-rest[c].mean())/(rest[c].std() or 1)
            drivers.append({"var":c, "z":round(float(z),2), "dir":"mayor" if z>0 else "menor"})
    drivers.sort(key=lambda x: abs(x["z"]), reverse=True)
    return json.dumps({"id":cluster_id, "name":s["cluster_name"].iloc[0],
                       "drivers":drivers[:top_n]}, ensure_ascii=False)


# ═══ CAPA 3: Búsqueda ═════════════════════════════════════════════

@tool
def count_visitors(cluster_id: int, filters: str = "") -> str:
    """Cuenta visitantes de un cluster con filtros opcionales (pandas query syntax).
    Siempre cuenta primero antes de traer filas, para no desperdiciar tokens."""
    df = _clustered(); s = df[df["cluster"]==cluster_id]
    if filters:
        for f in filters.split(","):
            try: s = s.query(f.strip())
            except Exception as e: return json.dumps({"error": str(e)})
    return json.dumps({"cluster":cluster_id, "filters":filters or "ninguno",
                       "total": int(len(df[df["cluster"]==cluster_id])),
                       "filtrado": int(len(s))})

@tool
def get_top_categories(cluster_id: int, dimension: str) -> str:
    """Distribución categórica de un cluster en una dimensión
    (channelGrouping, device.browser, device.deviceCategory, device.operatingSystem)."""
    df = _processed(); s = df[df["cluster"]==cluster_id]
    if dimension not in s.columns:
        return json.dumps({"error": f"'{dimension}' no existe"})
    t = s[dimension].value_counts(normalize=True).head(8)
    return json.dumps({"cluster":cluster_id, "dim":dimension,
                       "dist":{k:round(float(v)*100,1) for k,v in t.items()}}, ensure_ascii=False)


# ═══ CAPA 4: Estrategia ═══════════════════════════════════════════

_MKT = {
    0: {"perfil":"Desktop alto engagement","canal":"Email + retargeting display",
        "timing":"Laboral 9-18h entre semana","mensaje":"Guías, comparativas, lealtad",
        "kpis":["páginas/sesión","tiempo en sitio","conversión"],"budget":"Alto"},
    1: {"perfil":"Desktop frecuente superficial","canal":"Push + email con CTAs directos",
        "timing":"Laboral primeras horas","mensaje":"Ofertas directas, urgencia, descuentos",
        "kpis":["profundidad","tasa rebote","conversión"],"budget":"Medio"},
    2: {"perfil":"Weekend warriors","canal":"Social media + display fin de semana",
        "timing":"Viernes tarde a domingo","mensaje":"Flash sales, contenido lifestyle",
        "kpis":["CTR fin de semana","engagement social","ticket promedio"],"budget":"Medio"},
    3: {"perfil":"Mobile users sesiones cortas","canal":"SMS + push mobile + stories",
        "timing":"Commute 7-9h/17-19h y noche 21-23h","mensaje":"Micro-contenido, un tap",
        "kpis":["conversión mobile","bounce mobile","app installs"],"budget":"Medio-bajo"},
}

@tool
def suggest_marketing_strategy(cluster_id: int) -> str:
    """Recomendaciones de marketing pre-definidas para un cluster.
    La lógica de negocio es codificada, NO generada por el LLM — esto evita hallucinations."""
    if cluster_id not in _MKT:
        return json.dumps({"error": f"Cluster {cluster_id} sin estrategia"})
    return json.dumps({"cluster":cluster_id, **_MKT[cluster_id]}, ensure_ascii=False)


# ═══ Agrupaciones por agente ══════════════════════════════════════

ANALYST_TOOLS = [get_cluster_overview, get_cluster_profile, compare_clusters, rank_cluster_drivers]
STATISTICIAN_TOOLS = [test_significance, count_visitors, get_top_categories]
STRATEGIST_TOOLS = [suggest_marketing_strategy, get_cluster_profile, get_cluster_overview]
ALL_TOOLS = [get_cluster_overview, get_cluster_profile, compare_clusters,
             test_significance, rank_cluster_drivers, count_visitors,
             get_top_categories, suggest_marketing_strategy]
