"""
features_anomalias.py
=====================
Ingeniería de variables de anomalía — **el corazón del Módulo B**.

Construye, a nivel de ORDEN, variables que capturan los comportamientos de riesgo
descritos en el alcance (montos atípicos, IGV inconsistente, concentración de
proveedor, fraccionamiento, patrones temporales). **Todos los parámetros y puntos
de corte se DERIVAN de la distribución real de los datos**, no se asumen:

* El punto de corte del fraccionamiento (umbral "Gran Compra") se descubre a
  partir de la discontinuidad de P(Gran Compra | TOTAL) — ver
  `derivar_umbral_fraccionamiento`.
* La tolerancia del ratio de IGV se deriva de la propia distribución del ratio.
* Las desviaciones de monto se miden con z-scores ROBUSTOS (mediana y MAD) por
  categoría, calculados sobre los datos.

La función `cargar_dataset_limpio` REUTILIZA el dataset limpio del Módulo A; no
reimplementa la lectura ni la limpieza de los CSV.
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd

from . import config_b as cb

logger = logging.getLogger(__name__)

# Factor que vuelve comparable la MAD con la desviación estándar bajo normalidad.
_K_MAD: float = 1.4826


# ---------------------------------------------------------------------------
# 1) Carga del dataset limpio (REUTILIZA el Módulo A)
# ---------------------------------------------------------------------------
def cargar_dataset_limpio() -> pd.DataFrame:
    """
    Carga el dataset consolidado y limpio que generó el Módulo A.

    Si el parquet existe (lo habitual), se lee directamente. Si no existe, se
    ejecutan los módulos del Módulo A (`ingesta` + `limpieza`) para generarlo;
    en ningún caso el Módulo B reimplementa la lectura o la limpieza.

    Devuelve
    --------
    pd.DataFrame
        Órdenes válidas con las columnas normalizadas por el Módulo A.
    """
    ruta = cb.RUTA_DATASET_LIMPIO
    if ruta.exists():
        logger.info("Reutilizando dataset limpio del Módulo A: %s", ruta.name)
        return pd.read_parquet(ruta)

    logger.warning(
        "No existe %s; se ejecutan ingesta + limpieza del Módulo A para generarlo.",
        ruta.name,
    )
    from . import ingesta, limpieza

    df_crudo, _ = ingesta.consolidar(guardar=True)
    df_limpio, _ = limpieza.limpiar(df_crudo, guardar=True)
    return df_limpio


# ---------------------------------------------------------------------------
# 2) Normalización de la categoría (ACUERDO_MARCO)
# ---------------------------------------------------------------------------
# Los códigos de proceso ("EXT-CE-2021-7 ", "IM-CE-2020-5 ", ...) anteceden a la
# misma familia de productos y la fragmentan en muchas categorías. Quitarlos
# agrupa la misma familia (en los datos: 72 valores -> 33 categorías), lo que da
# una referencia por categoría más robusta para las desviaciones de monto.
_PATRON_CODIGO_PROCESO = re.compile(r"^[A-Z]{2,4}-CE-\d{4}-\d+\s+")


def normalizar_categoria(serie: pd.Series) -> pd.Series:
    """Quita el código de proceso inicial de `ACUERDO_MARCO` y unifica espacios."""
    def _norm(s: object) -> object:
        if not isinstance(s, str):
            return s
        s2 = _PATRON_CODIGO_PROCESO.sub("", s)
        return re.sub(r"\s+", " ", s2).strip()

    return serie.map(_norm)


# ---------------------------------------------------------------------------
# 3) Derivación de umbrales A PARTIR DE LOS DATOS
# ---------------------------------------------------------------------------
def derivar_umbral_fraccionamiento(
    df: pd.DataFrame, ancho_bin: float = 5000.0, tope: float = 300000.0
) -> dict:
    """
    Descubre el punto de corte del fraccionamiento (umbral de "Gran Compra")
    a partir de la **discontinuidad** de la probabilidad de que una orden sea
    "Gran Compra" en función de su `TOTAL`.

    Método (sin asumir ninguna cifra):
        1. Se divide `TOTAL` en tramos de `ancho_bin` soles hasta `tope`.
        2. Se calcula P(Gran Compra | tramo) en cada tramo.
        3. Se busca el mayor SALTO ascendente de esa probabilidad entre tramos
           consecutivos, exigiendo que tras el salto P supere 0.5 (la orden pasa
           a ser predominantemente Gran Compra). El borde izquierdo de ese tramo
           es el umbral.

    Devuelve
    --------
    dict con: 'umbral', 'p_antes', 'p_despues', 'salto' y la tabla por tramos.
    """
    es_gc = (df[cb.COL_TIPO_PROC] == "Gran Compra").astype(int)
    bordes = np.arange(0, tope + ancho_bin, ancho_bin)
    tramos = pd.cut(df[cb.COL_TOTAL], bins=bordes, right=True)
    tabla = (
        pd.DataFrame({"es_gc": es_gc, "tramo": tramos})
        .groupby("tramo", observed=True)["es_gc"]
        .agg(["size", "mean"])
        .rename(columns={"size": "n", "mean": "p_gc"})
    )
    # Salto de probabilidad entre tramos consecutivos.
    tabla["salto"] = tabla["p_gc"].diff()
    # Solo consideramos saltos que llevan la probabilidad por encima de 0.5.
    candidatos = tabla[(tabla["p_gc"] > 0.5) & (tabla["salto"] > 0)]
    if candidatos.empty:
        # Respaldo conservador: percentil 99 del TOTAL de "Compra ordinaria".
        umbral = float(
            df.loc[df[cb.COL_TIPO_PROC] == "Compra ordinaria", cb.COL_TOTAL].quantile(0.99)
        )
        logger.warning("No se halló discontinuidad clara; umbral de respaldo=%.0f", umbral)
        return {"umbral": umbral, "p_antes": np.nan, "p_despues": np.nan,
                "salto": np.nan, "tabla": tabla}

    tramo_corte = candidatos["salto"].idxmax()
    umbral = float(tramo_corte.left)
    idx = tabla.index.get_loc(tramo_corte)
    p_antes = float(tabla["p_gc"].iloc[idx - 1]) if idx > 0 else np.nan
    p_despues = float(tabla["p_gc"].iloc[idx])
    logger.info(
        "Umbral de fraccionamiento derivado de los datos: S/ %s "
        "(P(GranCompra) %.2f -> %.2f).", f"{umbral:,.0f}", p_antes, p_despues,
    )
    return {"umbral": umbral, "p_antes": p_antes, "p_despues": p_despues,
            "salto": float(candidatos["salto"].max()), "tabla": tabla}


def derivar_tolerancia_igv(df: pd.DataFrame) -> dict:
    """
    Deriva la tolerancia para marcar una orden con IGV inconsistente, a partir de
    la distribución real del ratio `IGV / SUB_TOTAL`.

    En los datos el ratio es bimodal: ~96 % de las órdenes está EXACTAMENTE en
    0.18 (con ruido de punto flotante <1e-4) y el resto se aparta de forma neta
    (sobre todo órdenes con IGV=0). Por eso cualquier tolerancia dentro del enorme
    hueco separa los dos modos. Se fija la tolerancia en 100× la desviación
    mediana absoluta del ratio respecto a 0.18 (acotada a un mínimo de 0.005 =
    medio punto porcentual), de modo que el ruido numérico nunca dispara la alarma.
    """
    ratio = df[cb.COL_IGV] / df[cb.COL_SUBTOTAL].replace(0, np.nan)
    desv = (ratio - cb.IGV_LEGAL).abs()
    mad = float(np.nanmedian(desv))
    tol = max(100.0 * mad, 0.005)
    n_inconsistentes = int((desv > tol).sum())
    logger.info(
        "Tolerancia IGV derivada: %.4f -> %d órdenes inconsistentes (%.2f%%).",
        tol, n_inconsistentes, 100 * n_inconsistentes / len(df),
    )
    return {"tolerancia": tol, "mad": mad, "n_inconsistentes": n_inconsistentes}


def _z_robusto(valores: pd.Series, grupos: pd.Series) -> pd.Series:
    """
    Z-score ROBUSTO de `valores` dentro de cada `grupo`: (x - mediana) / (k·MAD).

    Usa mediana y MAD (robustas a la cola larga). Donde la MAD del grupo es 0
    (todos los valores iguales) se cae al MAD global para no dividir por cero.
    """
    med = valores.groupby(grupos).transform("median")
    mad = valores.groupby(grupos).transform(lambda s: np.median(np.abs(s - np.median(s))))
    mad_global = float(np.median(np.abs(valores - np.median(valores)))) or 1.0
    escala = _K_MAD * mad.where(mad > 0, mad_global)
    return (valores - med) / escala


# ---------------------------------------------------------------------------
# 4) Conteos en ventana temporal por par entidad–proveedor (fraccionamiento)
# ---------------------------------------------------------------------------
def _conteos_ventana(
    df: pd.DataFrame, par: pd.Series, ventana_dias: int
) -> tuple[pd.Series, pd.Series]:
    """
    Para cada orden, cuenta cuántas órdenes del MISMO par entidad–proveedor
    cayeron en la ventana temporal `[día - ventana_dias, día]` (incluida la propia
    orden), y suma sus `TOTAL`.

    Implementación vectorizada en NumPy: se ordena por (par, día) y se usa una
    búsqueda binaria global con las claves desplazadas por grupo, de modo que la
    ventana nunca cruza de un par a otro. Es O(n log n) y maneja con holgura las
    ~190k relaciones entidad–proveedor.

    Devuelve dos series alineadas al índice original: (conteo, suma).
    """
    n = len(df)
    g = pd.factorize(par.values)[0].astype(np.int64)                 # id de par
    t_dias = (
        df[cb.COL_FECHA_FORM].values.astype("datetime64[D]")
        - np.datetime64("2000-01-01")
    ).astype("int64")                                                # día calendario
    total = df[cb.COL_TOTAL].to_numpy(dtype=float)

    orden = np.lexsort((t_dias, g))                                  # ordena por par, luego día
    gs, ts, tots = g[orden], t_dias[orden], total[orden]

    base = ts - ts.min()
    span = int(base.max()) if n else 0
    # Separación entre pares mayor que la ventana -> los grupos no se solapan.
    paso = span + ventana_dias + 2
    clave = gs * paso + base
    consulta = gs * paso + (base - ventana_dias)
    izq = np.searchsorted(clave, consulta, side="left")             # inicio de la ventana

    pos = np.arange(n)
    conteo = pos - izq + 1
    prefijo = np.concatenate([[0.0], np.cumsum(tots)])
    suma = prefijo[pos + 1] - prefijo[izq]

    # Deshacer el orden para realinear con el índice original.
    conteo_orig = np.empty(n, dtype=float)
    suma_orig = np.empty(n, dtype=float)
    conteo_orig[orden] = conteo
    suma_orig[orden] = suma
    return (
        pd.Series(conteo_orig, index=df.index),
        pd.Series(suma_orig, index=df.index),
    )


# ---------------------------------------------------------------------------
# 5) Construcción del conjunto de features
# ---------------------------------------------------------------------------
def construir_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Construye todas las variables de anomalía a nivel de orden.

    Parámetros
    ----------
    df : pd.DataFrame
        Dataset limpio del Módulo A (órdenes válidas).

    Devuelve
    --------
    (feats, meta) : tuple[pd.DataFrame, dict]
        `feats` tiene una fila por orden con columnas identificadoras + las
        variables de anomalía. `meta` guarda los umbrales DERIVADOS de los datos
        (umbral de fraccionamiento, tolerancia de IGV, etc.) para el informe.
    """
    df = df.copy()
    meta: dict = {}

    # --- Identificadores y categoría normalizada ------------------------------
    df["CAT_NORM"] = normalizar_categoria(df[cb.COL_ACUERDO])

    # --- Umbrales derivados de los datos --------------------------------------
    info_umbral = derivar_umbral_fraccionamiento(df)
    umbral = info_umbral["umbral"]
    meta["umbral_fraccionamiento"] = info_umbral
    info_igv = derivar_tolerancia_igv(df)
    tol_igv = info_igv["tolerancia"]
    meta["tolerancia_igv"] = info_igv

    f = pd.DataFrame(index=df.index)
    f[cb.COL_ORDEN] = df[cb.COL_ORDEN].values
    f[cb.COL_RUC_ENTIDAD] = df[cb.COL_RUC_ENTIDAD].values
    f[cb.COL_RUC_PROVEEDOR] = df[cb.COL_RUC_PROVEEDOR].values

    # =========================================================================
    # A) MONTOS ATÍPICOS
    # =========================================================================
    f["TOTAL"] = df[cb.COL_TOTAL].values
    f["LOG_TOTAL"] = np.log1p(df[cb.COL_TOTAL].values)
    # Desviación robusta del log-monto respecto a su categoría y al global.
    f["Z_TOTAL_CAT"] = _z_robusto(np.log1p(df[cb.COL_TOTAL]), df["CAT_NORM"]).values
    f["Z_TOTAL_GLOBAL"] = _z_robusto(
        np.log1p(df[cb.COL_TOTAL]), pd.Series("__all__", index=df.index)
    ).values

    # =========================================================================
    # B) INCONSISTENCIA DE IGV
    # =========================================================================
    ratio_igv = (df[cb.COL_IGV] / df[cb.COL_SUBTOTAL].replace(0, np.nan)).fillna(0.0)
    f["RATIO_IGV"] = ratio_igv.values
    f["ABS_DESV_IGV"] = (ratio_igv - cb.IGV_LEGAL).abs().values
    f["FLAG_IGV"] = (f["ABS_DESV_IGV"] > tol_igv).astype(int)

    # =========================================================================
    # C) CONCENTRACIÓN DE PROVEEDOR (relación entidad–proveedor)
    # =========================================================================
    gasto_par = df.groupby([cb.COL_RUC_ENTIDAD, cb.COL_RUC_PROVEEDOR])[cb.COL_TOTAL].transform("sum")
    n_par = df.groupby([cb.COL_RUC_ENTIDAD, cb.COL_RUC_PROVEEDOR])[cb.COL_TOTAL].transform("size")
    gasto_ent = df.groupby(cb.COL_RUC_ENTIDAD)[cb.COL_TOTAL].transform("sum")
    n_ent = df.groupby(cb.COL_RUC_ENTIDAD)[cb.COL_TOTAL].transform("size")

    f["SHARE_GASTO_PROV"] = (gasto_par / gasto_ent).values          # share del gasto de la entidad a este proveedor
    f["SHARE_ORD_PROV"] = (n_par / n_ent).values                    # share de órdenes
    f["N_ORD_PAR"] = n_par.values                                   # nº de órdenes del par
    f["N_PROV_ENTIDAD"] = df.groupby(cb.COL_RUC_ENTIDAD)[cb.COL_RUC_PROVEEDOR].transform("nunique").values

    # HHI de proveedores por entidad (suma de cuadrados de los shares de gasto).
    share_prov_unico = (
        df.groupby([cb.COL_RUC_ENTIDAD, cb.COL_RUC_PROVEEDOR])[cb.COL_TOTAL].sum()
        / df.groupby(cb.COL_RUC_ENTIDAD)[cb.COL_TOTAL].sum()
    )
    hhi = (share_prov_unico**2).groupby(level=0).sum().rename("HHI")
    f["HHI_ENTIDAD"] = df[cb.COL_RUC_ENTIDAD].map(hhi).values

    # =========================================================================
    # D) POSIBLE FRACCIONAMIENTO (cercanía al umbral + ráfaga en ventana corta)
    # =========================================================================
    meta["umbral_usado"] = umbral
    total = df[cb.COL_TOTAL]
    # Proximidad por debajo del umbral: 1 justo en el umbral, 0 lejos o por encima.
    en_banda = (total >= cb.BANDA_PROXIMIDAD_UMBRAL * umbral) & (total < umbral)
    f["PROXIMIDAD_UMBRAL"] = np.where(en_banda, total / umbral, 0.0)
    f["FLAG_BAJO_UMBRAL"] = en_banda.astype(int).values
    # Solo aplica a "Compra ordinaria" (el procedimiento que evita la Gran Compra).
    f["FLAG_BAJO_UMBRAL"] *= (df[cb.COL_TIPO_PROC] == "Compra ordinaria").astype(int).values

    par = df[cb.COL_RUC_ENTIDAD].astype(str) + "|" + df[cb.COL_RUC_PROVEEDOR].astype(str)
    for w in cb.VENTANAS_FRACC_DIAS:
        conteo, suma = _conteos_ventana(df, par, w)
        f[f"N_ORD_VENTANA_{w}D"] = conteo.values
        f[f"SUMA_VENTANA_{w}D"] = suma.values
    # Señal de fraccionamiento: ráfaga de órdenes del par en 7 días cuya SUMA
    # cruza el umbral (lo que habría exigido una Gran Compra si fuese una sola).
    w0 = cb.VENTANAS_FRACC_DIAS[0]
    f["FLAG_FRACCIONAMIENTO"] = (
        (f[f"N_ORD_VENTANA_{w0}D"] >= 2) & (f[f"SUMA_VENTANA_{w0}D"] >= umbral)
    ).astype(int)

    # =========================================================================
    # E) PATRONES TEMPORALES
    # =========================================================================
    dia = df[cb.COL_FECHA_FORM].dt.normalize()
    f["N_ORD_ENTIDAD_DIA"] = df.groupby([cb.COL_RUC_ENTIDAD, dia])[cb.COL_TOTAL].transform("size").values
    # Proximidad al cierre de mes.
    fin_mes = df[cb.COL_FECHA_FORM] + pd.offsets.MonthEnd(0)
    dias_para_fin = (fin_mes.dt.normalize() - dia).dt.days
    f["DIAS_PARA_FIN_MES"] = dias_para_fin.values
    f["FLAG_CIERRE_MES"] = (dias_para_fin <= cb.DIAS_CIERRE_MES).astype(int).values
    # Tiempo entre formalización y último estado (días).
    f["DIAS_ESTADO"] = (
        (df[cb.COL_FECHA_ULT] - df[cb.COL_FECHA_FORM]).dt.total_seconds() / 86400
    ).values

    # --- Limpieza de posibles NaN en features numéricas (rellena con 0) -------
    cols_num = f.select_dtypes(include=[np.number]).columns
    f[cols_num] = f[cols_num].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    meta["columnas_features"] = [c for c in f.columns if c not in
                                 (cb.COL_ORDEN, cb.COL_RUC_ENTIDAD, cb.COL_RUC_PROVEEDOR)]
    logger.info("Features construidas: %d columnas, %d órdenes.", f.shape[1], len(f))
    return f, meta


# Variables que entran al MODELO no supervisado (Isolation Forest / autoencoder).
# Se excluyen identificadores y banderas redundantes; se quedan las señales
# continuas y de conteo que describen el comportamiento de la orden.
COLUMNAS_MODELO: tuple[str, ...] = (
    "LOG_TOTAL", "Z_TOTAL_CAT", "Z_TOTAL_GLOBAL",
    "ABS_DESV_IGV",
    "SHARE_GASTO_PROV", "SHARE_ORD_PROV", "N_ORD_PAR", "HHI_ENTIDAD",
    "PROXIMIDAD_UMBRAL", "N_ORD_VENTANA_7D", "SUMA_VENTANA_7D",
    "N_ORD_VENTANA_30D",
    "N_ORD_ENTIDAD_DIA", "DIAS_PARA_FIN_MES", "DIAS_ESTADO",
)
