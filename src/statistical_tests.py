"""
Tests estadísticos formales sobre los clusters obtenidos.

Con n ≈ 10000, todo p-value sale prácticamente cero. Por eso reportamos
también effect size, que es lo que diferencia diferencias triviales
de diferencias importantes:
  - eta-cuadrado para Kruskal-Wallis: 0.01 pequeño, 0.06 mediano, 0.14 grande
  - Cramer's V para chi-cuadrado: 0.1 débil, 0.3 moderado, 0.5 fuerte

Decidimos Kruskal-Wallis sobre ANOVA porque ninguna variable es normal
dentro de los clusters (test de D'Agostino-Pearson rechaza normalidad
en todos los casos).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import kruskal, chi2_contingency, f_oneway


def test_normality_by_cluster(df: pd.DataFrame, variables: list[str],
                               cluster_col: str = 'cluster',
                               sample_size: int = 500) -> dict[str, bool]:
    """D'Agostino-Pearson por cluster. True si TODOS los clusters son normales."""
    results = {}
    for var in variables:
        all_normal = True
        for c in df[cluster_col].unique():
            sub = df.loc[df[cluster_col] == c, var]
            sample = sub.sample(min(sample_size, len(sub)), random_state=42)
            _, p = stats.normaltest(sample)
            if p < 0.05:
                all_normal = False
                break
        results[var] = all_normal
    return results


def eta_squared_kw(groups: list[np.ndarray]) -> float:
    """Eta-cuadrado para Kruskal-Wallis: (H - k + 1) / (n - k)."""
    h, _ = kruskal(*groups)
    n = sum(len(g) for g in groups)
    return (h - len(groups) + 1) / (n - len(groups))


def eta_squared_anova(groups: list[np.ndarray]) -> float:
    """Eta-cuadrado para ANOVA: SS_between / SS_total."""
    all_data = np.concatenate(groups)
    grand_mean = np.mean(all_data)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = sum((x - grand_mean) ** 2 for g in groups for x in g)
    return ss_between / ss_total


def cramers_v(a: pd.Series, b: pd.Series) -> float:
    """Cramer's V para tabla de contingencia."""
    ct = pd.crosstab(a, b)
    chi2, _, _, _ = chi2_contingency(ct)
    n = ct.values.sum()
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))


def run_omnibus_numerical(df: pd.DataFrame, variables: list[str],
                           cluster_col: str = 'cluster') -> pd.DataFrame:
    """Kruskal-Wallis + eta-cuadrado para cada variable numérica."""
    rows = []
    for var in variables:
        groups = [df.loc[df[cluster_col] == c, var].values
                  for c in sorted(df[cluster_col].unique())]
        h, p = kruskal(*groups)
        eta = eta_squared_kw(groups)
        magnitud = ('grande' if eta > 0.14
                    else 'mediano' if eta > 0.06
                    else 'pequeno' if eta > 0.01
                    else 'trivial')
        rows.append({
            'variable': var, 'estadistico_H': h, 'p_value': p,
            'eta_cuadrado': eta, 'magnitud': magnitud,
        })
    return pd.DataFrame(rows).sort_values('eta_cuadrado', ascending=False)


def run_omnibus_categorical(df: pd.DataFrame, variables: list[str],
                             cluster_col: str = 'cluster') -> pd.DataFrame:
    """Chi-cuadrado + Cramer's V para cada variable categórica."""
    rows = []
    for var in variables:
        ct = pd.crosstab(df[cluster_col], df[var])
        chi2, p, dof, _ = chi2_contingency(ct)
        v = cramers_v(df[cluster_col], df[var])
        fuerza = ('fuerte' if v > 0.5
                  else 'moderado' if v > 0.3
                  else 'debil' if v > 0.1
                  else 'trivial')
        rows.append({
            'variable': var, 'chi2': chi2, 'p_value': p,
            'cramers_v': v, 'fuerza': fuerza,
        })
    return pd.DataFrame(rows).sort_values('cramers_v', ascending=False)


def manova_test(df: pd.DataFrame, variables: list[str],
                 cluster_col: str = 'cluster') -> str:
    """MANOVA con Wilks' Lambda. Retorna el resumen como string."""
    from statsmodels.multivariate.manova import MANOVA
    df_m = df[variables + [cluster_col]].copy()
    df_m.columns = [c.replace('.', '_') for c in df_m.columns]
    vars_clean = [v.replace('.', '_') for v in variables]
    formula = ' + '.join(vars_clean) + f' ~ {cluster_col}'
    maov = MANOVA.from_formula(formula, data=df_m)
    return str(maov.mv_test())
