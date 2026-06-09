"""
modelos_anomalias.py
====================
Modelado NO SUPERVISADO de la detección de anomalías. Tres enfoques
complementarios, todos reproducibles (semilla fija):

1. **Isolation Forest** sobre la matriz estandarizada de features de anomalía:
   aísla órdenes que se separan del comportamiento conjunto.
2. **Autoencoder** (MLP de scikit-learn que reconstruye su propia entrada): las
   órdenes con mayor error de reconstrucción son las más atípicas. Es opcional;
   si falla, el flujo continúa con los otros métodos.
3. **Análisis de grafo** (networkx) de la red entidad–proveedor: grado,
   concentración bilateral (lock-in), comunidades y proveedores cautivos.

Cada método devuelve un puntaje por orden, normalizado a [0, 1] por rango
percentil para que sean combinables en `riesgo.py`.
"""

from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ml.shared import config as cb
from . import features_anomalias as fa

logger = logging.getLogger(__name__)


def fijar_semillas() -> None:
    """Fija las semillas globales para reproducibilidad."""
    import random

    random.seed(cb.SEMILLA)
    np.random.seed(cb.SEMILLA)


def _a_percentil(x: np.ndarray | pd.Series) -> np.ndarray:
    """Normaliza un puntaje a [0, 1] por rango percentil (robusto a la escala)."""
    return pd.Series(np.asarray(x, dtype=float)).rank(pct=True).to_numpy()


# ---------------------------------------------------------------------------
# 1) Isolation Forest
# ---------------------------------------------------------------------------
def entrenar_isolation_forest(feats: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """
    Entrena un Isolation Forest sobre las features estandarizadas y devuelve el
    puntaje de anomalía por orden (mayor = más anómala), normalizado a [0, 1].

    Devuelve
    --------
    (score_if, info) : tuple[np.ndarray, dict]
        `score_if` en [0,1] por orden; `info` con metadatos del modelo.
    """
    X = feats[list(fa.COLUMNAS_MODELO)].to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(X)

    modelo = IsolationForest(
        n_estimators=cb.IF_N_ESTIMADORES,
        max_samples=cb.IF_MAX_SAMPLES,
        contamination=cb.IF_CONTAMINACION,
        random_state=cb.SEMILLA,
        n_jobs=-1,
    )
    modelo.fit(Xs)
    # score_samples: cuanto MENOR, más anómala. Invertimos para que mayor = peor.
    crudo = -modelo.score_samples(Xs)
    score_if = _a_percentil(crudo)
    # El umbral interno (predict==-1) sirve de referencia de "cuántas marca".
    n_outliers = int((modelo.predict(Xs) == -1).sum())
    info = {
        "n_outliers_if": n_outliers,
        "pct_outliers_if": 100 * n_outliers / len(feats),
        "n_features": len(fa.COLUMNAS_MODELO),
    }
    logger.info(
        "Isolation Forest: %d órdenes marcadas como outlier (%.2f%%).",
        n_outliers, info["pct_outliers_if"],
    )
    return score_if, info


# ---------------------------------------------------------------------------
# 2) Autoencoder (opcional, MLP de scikit-learn)
# ---------------------------------------------------------------------------
def entrenar_autoencoder(feats: pd.DataFrame) -> tuple[np.ndarray | None, dict]:
    """
    Entrena un autoencoder simple (MLPRegressor que reconstruye su entrada) y
    devuelve el error de reconstrucción por orden, normalizado a [0, 1].

    Es opcional: ante cualquier problema (o si está desactivado en config) se
    devuelve `None` y el flujo sigue con los demás métodos.
    """
    if not cb.USAR_AUTOENCODER:
        return None, {"autoencoder": "desactivado"}
    try:
        from sklearn.neural_network import MLPRegressor

        X = feats[list(fa.COLUMNAS_MODELO)].to_numpy(dtype=float)
        Xs = StandardScaler().fit_transform(X)
        ae = MLPRegressor(
            hidden_layer_sizes=cb.AE_CAPAS_OCULTAS,
            activation="relu",
            solver="adam",
            max_iter=cb.AE_MAX_ITER,
            random_state=cb.SEMILLA,
        )
        ae.fit(Xs, Xs)                       # objetivo = entrada (reconstrucción)
        recon = ae.predict(Xs)
        error = np.mean((Xs - recon) ** 2, axis=1)
        score_ae = _a_percentil(error)
        logger.info("Autoencoder entrenado (error recon. medio=%.4f).", float(error.mean()))
        return score_ae, {"autoencoder": "ok", "error_medio": float(error.mean())}
    except Exception as exc:  # noqa: BLE001 - es opcional, no debe romper el flujo
        logger.warning("Autoencoder omitido por error: %s", exc)
        return None, {"autoencoder": f"omitido ({exc})"}


# ---------------------------------------------------------------------------
# 3) Análisis de grafo de la red entidad–proveedor
# ---------------------------------------------------------------------------
def analizar_red(df: pd.DataFrame, feats: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Construye el grafo bipartito entidad–proveedor (aristas ponderadas por el
    gasto del par) y deriva, con networkx, señales de relación atípica:

    * **Grado**: nº de entidades por proveedor y de proveedores por entidad.
    * **Concentración bilateral (lock-in)**: media geométrica entre la dependencia
      de la entidad respecto al proveedor y la del proveedor respecto a la entidad.
      Es alta cuando ambos dependen casi exclusivamente el uno del otro.
    * **Comunidades**: detección por propagación de etiquetas (reproducible).
    * **Proveedor cautivo**: proveedor que atiende a una sola entidad.

    Devuelve
    --------
    (red, info) : tuple[pd.DataFrame, dict]
        `red` está alineada al índice de `feats` con las columnas de red por
        orden (incluido `SCORE_RED` en [0,1]); `info` resume el grafo y lista las
        relaciones más atípicas para el informe.
    """
    e, p, t = cb.COL_RUC_ENTIDAD, cb.COL_RUC_PROVEEDOR, cb.COL_TOTAL

    # --- Agregados por par y por nodo ----------------------------------------
    par = df.groupby([e, p])[t].agg(gasto="sum", n="size").reset_index()
    gasto_ent = df.groupby(e)[t].sum()
    gasto_prov = df.groupby(p)[t].sum()
    par["s_ent_prov"] = par["gasto"].values / gasto_ent.loc[par[e]].values   # dependencia entidad->prov
    par["s_prov_ent"] = par["gasto"].values / gasto_prov.loc[par[p]].values  # dependencia prov->entidad
    par["lock_in"] = np.sqrt(par["s_ent_prov"] * par["s_prov_ent"])          # candado bilateral

    # --- Grafo bipartito con networkx ----------------------------------------
    G = nx.Graph()
    nodos_e = ["E:" + x for x in par[e]]
    nodos_p = ["P:" + x for x in par[p]]
    G.add_weighted_edges_from(zip(nodos_e, nodos_p, par["gasto"].to_numpy()))

    grado_prov = {n[2:]: d for n, d in G.degree() if n.startswith("P:")}  # nº entidades
    grado_ent = {n[2:]: d for n, d in G.degree() if n.startswith("E:")}   # nº proveedores
    par["grado_prov"] = par[p].map(grado_prov)
    par["grado_ent"] = par[e].map(grado_ent)
    par["prov_cautivo"] = (par["grado_prov"] == 1).astype(int)

    # --- Comunidades (Louvain, reproducible con semilla) ---------------------
    # Louvain (modularidad) es rápido y da comunidades informativas; mucho más
    # veloz que la propagación asíncrona de etiquetas sobre ~190k aristas.
    try:
        comunidades = list(
            nx.community.louvain_communities(G, weight="weight", seed=cb.SEMILLA)
        )
        nodo_com = {nodo: i for i, com in enumerate(comunidades) for nodo in com}
        tam_com = {i: len(com) for i, com in enumerate(comunidades)}
        par["comunidad"] = ["E:" + x for x in par[e]]
        par["comunidad"] = par["comunidad"].map(nodo_com)
        par["tam_comunidad"] = par["comunidad"].map(tam_com)
        n_comunidades = len(comunidades)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Detección de comunidades omitida: %s", exc)
        par["comunidad"] = -1
        par["tam_comunidad"] = np.nan
        n_comunidades = 0

    # --- Puntaje de red por par ----------------------------------------------
    # El lock-in es el eje; se refuerza si el proveedor es cautivo (atiende a una
    # sola entidad), porque ahí la relación es exclusiva en ambos sentidos.
    bruto = par["lock_in"] * (1.0 + 0.5 * par["prov_cautivo"])
    par["SCORE_RED"] = _a_percentil(bruto)

    # --- Volcado a nivel de orden (merge por par) ----------------------------
    cols_par = [e, p, "s_prov_ent", "lock_in", "grado_prov", "grado_ent",
                "prov_cautivo", "tam_comunidad", "SCORE_RED"]
    red = feats[[cb.COL_RUC_ENTIDAD, cb.COL_RUC_PROVEEDOR]].merge(
        par[cols_par], on=[e, p], how="left"
    )
    red.index = feats.index
    red = red.rename(columns={
        "s_prov_ent": "DEP_PROV_ENTIDAD", "lock_in": "LOCK_IN",
        "grado_prov": "N_ENTIDADES_PROV", "grado_ent": "N_PROVEEDORES_ENT",
        "prov_cautivo": "FLAG_PROV_CAUTIVO", "tam_comunidad": "TAM_COMUNIDAD",
    })

    # --- Relaciones más atípicas (para el informe) ---------------------------
    top = par.sort_values("SCORE_RED", ascending=False).head(15)
    info = {
        "n_nodos": G.number_of_nodes(),
        "n_aristas": G.number_of_edges(),
        "n_entidades": len(grado_ent),
        "n_proveedores": len(grado_prov),
        "n_comunidades": n_comunidades,
        "n_prov_cautivos": int((par["prov_cautivo"] == 1).sum()),
        "grado_prov_max": int(max(grado_prov.values())) if grado_prov else 0,
        "grado_ent_max": int(max(grado_ent.values())) if grado_ent else 0,
        "top_relaciones": top,
    }
    logger.info(
        "Grafo: %d nodos, %d aristas, %d comunidades, %d proveedores cautivos.",
        info["n_nodos"], info["n_aristas"], info["n_comunidades"], info["n_prov_cautivos"],
    )
    return red, info


# ---------------------------------------------------------------------------
# Orquestación de los modelos
# ---------------------------------------------------------------------------
def ejecutar_modelos(feats: pd.DataFrame, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Ejecuta los tres enfoques y devuelve un DataFrame de puntajes por orden.

    Columnas de salida: SCORE_IF, SCORE_AE (si hay), y todas las de red
    (SCORE_RED, LOCK_IN, etc.). `info` agrega los metadatos de cada método.
    """
    fijar_semillas()
    info: dict = {}

    score_if, info_if = entrenar_isolation_forest(feats)
    info.update(info_if)

    score_ae, info_ae = entrenar_autoencoder(feats)
    info.update(info_ae)

    red, info_red = analizar_red(df, feats)
    info["red"] = info_red

    salida = pd.DataFrame(index=feats.index)
    salida["SCORE_IF"] = score_if
    if score_ae is not None:
        salida["SCORE_AE"] = score_ae
    salida = salida.join(red.drop(columns=[cb.COL_RUC_ENTIDAD, cb.COL_RUC_PROVEEDOR]))
    return salida, info
