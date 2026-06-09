"""
serie_temporal.py
=================
Construcción de la serie temporal mensual del gasto a partir del dataset limpio.

* Serie principal: suma de `TOTAL` agrupada por mes usando `FECHA_FORMALIZACION`.
* Serie secundaria: gasto mensual por `ACUERDO_MARCO` (categoría).
* Detección AUTOMÁTICA de meses incompletos en la cola: se compara el conteo de
  órdenes de cada mes final con el mismo mes del año anterior (control de la
  fuerte estacionalidad: enero/febrero son bajos todos los años). Solo se recorta
  un mes de la cola si su conteo cae por debajo del umbral configurado.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from ml.shared import config

logger = logging.getLogger(__name__)


def construir_serie_total(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega el gasto mensual total.

    Devuelve un DataFrame indexado por mes (fin de mes, frecuencia 'ME') con:
      - `gasto`   : suma de TOTAL en el mes (soles)
      - `ordenes` : número de órdenes válidas en el mes
      - `ticket_promedio` : gasto / ordenes
    """
    s = df.dropna(subset=[config.COL_FECHA_FORMALIZACION]).copy()
    s["mes"] = s[config.COL_FECHA_FORMALIZACION].dt.to_period("M")
    agg = (
        s.groupby("mes")
        .agg(gasto=(config.COL_TOTAL, "sum"), ordenes=(config.COL_TOTAL, "size"))
    )
    # Reindexar a un rango continuo de meses (rellena meses ausentes con 0).
    rango = pd.period_range(agg.index.min(), agg.index.max(), freq="M")
    agg = agg.reindex(rango, fill_value=0)
    agg.index = agg.index.to_timestamp(how="end").normalize()
    agg.index.name = "fecha"
    agg["ticket_promedio"] = (agg["gasto"] / agg["ordenes"]).where(agg["ordenes"] > 0, 0)
    return agg


def construir_serie_diaria(df: pd.DataFrame) -> pd.DataFrame:
    """
    Serie de gasto a granularidad **diaria** (rango continuo, días sin órdenes = 0).

    Convierte las ~480 mil transacciones en ~1 600 puntos (vs. ~50 mensuales),
    recuperando el tamaño muestral que ahogaba a los modelos de ML/DL y permitiendo
    aprender el calendario (días hábiles, feriados, patrón intra-mes) directamente.
    """
    return _agregar_por_periodo(df, "D")


def construir_serie_semanal(df: pd.DataFrame) -> pd.DataFrame:
    """Serie de gasto a granularidad **semanal** (~230 puntos)."""
    return _agregar_por_periodo(df, "W")


def _agregar_por_periodo(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    Agrega el gasto por periodo `freq` ('D' diario, 'W' semanal) sobre
    `FECHA_FORMALIZACION`, con índice continuo (periodos sin órdenes = 0). Misma
    lógica que `construir_serie_total` pero parametrizada por frecuencia.
    """
    s = df.dropna(subset=[config.COL_FECHA_FORMALIZACION]).copy()
    fecha = s[config.COL_FECHA_FORMALIZACION].dt.normalize()
    agg = (
        s.assign(_f=fecha)
        .groupby("_f")
        .agg(gasto=(config.COL_TOTAL, "sum"), ordenes=(config.COL_TOTAL, "size"))
    )
    idx = pd.date_range(agg.index.min(), agg.index.max(), freq="D")
    agg = agg.reindex(idx, fill_value=0)
    if freq != "D":
        agg = agg.resample(freq).sum()
    agg.index.name = "fecha"
    agg["ticket_promedio"] = (agg["gasto"] / agg["ordenes"]).where(agg["ordenes"] > 0, 0)
    return agg


def detectar_meses_incompletos(serie: pd.DataFrame) -> dict:
    """
    Detecta, de forma automática y a partir de los propios datos, si los últimos
    meses de la serie están incompletos.

    Estrategia robusta frente a la estacionalidad: para cada mes de la cola se
    compara su número de órdenes con el del MISMO mes del año anterior. Si el
    cociente es menor que `config.UMBRAL_MES_INCOMPLETO`, el mes se marca como
    incompleto. Se examinan solo los últimos meses (la cola), de forma
    consecutiva desde el final hacia atrás.

    Devuelve un diccionario con la lista de meses incompletos y el detalle de la
    comparación, para documentar la decisión en el informe.
    """
    conteo = serie["ordenes"]
    detalle = []
    meses_incompletos = []

    # Recorremos la cola hacia atrás mientras haya evidencia de incompletitud.
    for i in range(len(serie) - 1, -1, -1):
        fecha_actual = serie.index[i]
        valor_actual = conteo.iloc[i]
        # Mismo mes del año anterior (12 posiciones atrás si la serie es mensual).
        if i - config.PERIODO_ESTACIONAL < 0:
            # No hay referencia del año previo: no podemos juzgar -> paramos.
            break
        valor_referencia = conteo.iloc[i - config.PERIODO_ESTACIONAL]
        if valor_referencia <= 0:
            break
        ratio = valor_actual / valor_referencia
        detalle.append(
            {
                "mes": fecha_actual.strftime("%Y-%m"),
                "ordenes": int(valor_actual),
                "ordenes_anio_previo": int(valor_referencia),
                "ratio": round(float(ratio), 3),
                "incompleto": ratio < config.UMBRAL_MES_INCOMPLETO,
            }
        )
        if ratio < config.UMBRAL_MES_INCOMPLETO:
            meses_incompletos.append(fecha_actual)
        else:
            # En cuanto un mes de la cola luce completo, dejamos de recortar.
            break

    return {
        "meses_incompletos": sorted(meses_incompletos),
        "detalle_cola": list(reversed(detalle)),
        "umbral": config.UMBRAL_MES_INCOMPLETO,
    }


def recortar_incompletos(serie: pd.DataFrame, deteccion: dict) -> pd.DataFrame:
    """Devuelve la serie sin los meses marcados como incompletos en la cola."""
    if not deteccion["meses_incompletos"]:
        return serie
    a_quitar = set(deteccion["meses_incompletos"])
    recortada = serie[~serie.index.isin(a_quitar)].copy()
    logger.info(
        "Se recortan %d meses incompletos de la cola: %s",
        len(a_quitar),
        [d.strftime("%Y-%m") for d in sorted(a_quitar)],
    )
    return recortada


def construir_serie_por_categoria(df: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """
    Construye la serie mensual de gasto por categoría (`ACUERDO_MARCO`).

    Para evitar la dispersión (95 categorías con variaciones de etiqueta), se
    devuelven solo las `top_n` categorías de mayor gasto acumulado; el resto se
    agrupa en 'OTRAS'. El índice es el mes y las columnas son las categorías.
    """
    s = df.dropna(subset=[config.COL_FECHA_FORMALIZACION]).copy()
    s["mes"] = s[config.COL_FECHA_FORMALIZACION].dt.to_period("M")
    gasto_cat = s.groupby(config.COL_ACUERDO_MARCO)[config.COL_TOTAL].sum()
    top = gasto_cat.sort_values(ascending=False).head(top_n).index
    s["categoria"] = s[config.COL_ACUERDO_MARCO].where(
        s[config.COL_ACUERDO_MARCO].isin(top), other="OTRAS"
    )
    tabla = (
        s.groupby(["mes", "categoria"])[config.COL_TOTAL]
        .sum()
        .unstack(fill_value=0)
        .sort_index()
    )
    tabla.index = tabla.index.to_timestamp(how="end").normalize()
    tabla.index.name = "fecha"
    return tabla


def _top_categorias(df: pd.DataFrame, top_n: int) -> pd.Index:
    """Índice de las `top_n` categorías (ACUERDO_MARCO) de mayor gasto acumulado."""
    gasto_cat = df.groupby(config.COL_ACUERDO_MARCO)[config.COL_TOTAL].sum()
    return gasto_cat.sort_values(ascending=False).head(top_n).index


def construir_panel_categorias(
    df: pd.DataFrame, top_n: int = config.TOP_N_CATEGORIAS
) -> pd.DataFrame:
    """
    Panel **largo** del gasto mensual por categoría, base del modelo global.

    Para cada `(categoria, mes)` devuelve `gasto`, `ordenes` y `ticket_promedio`,
    con un índice mensual continuo POR categoría (meses sin gasto = 0), de modo que
    todas las categorías comparten el mismo rango temporal. Se conservan las `top_n`
    categorías de mayor gasto (cubren ~95% del total) y el resto se agrupa en
    'OTRAS'; así la **suma de las series del panel reconstituye el gasto total**
    (reconciliación bottom-up exacta).

    Columnas: ['categoria', 'fecha', 'gasto', 'ordenes', 'ticket_promedio'].
    """
    s = df.dropna(subset=[config.COL_FECHA_FORMALIZACION]).copy()
    s["mes"] = s[config.COL_FECHA_FORMALIZACION].dt.to_period("M")
    top = _top_categorias(s, top_n)
    s["categoria"] = s[config.COL_ACUERDO_MARCO].where(
        s[config.COL_ACUERDO_MARCO].isin(top), other="OTRAS"
    )

    agg = (
        s.groupby(["categoria", "mes"])
        .agg(gasto=(config.COL_TOTAL, "sum"), ordenes=(config.COL_TOTAL, "size"))
    )

    # Rango mensual común y reindex por categoría (meses ausentes -> 0).
    rango = pd.period_range(s["mes"].min(), s["mes"].max(), freq="M")
    categorias = agg.index.get_level_values("categoria").unique()
    idx_completo = pd.MultiIndex.from_product([categorias, rango], names=["categoria", "mes"])
    agg = agg.reindex(idx_completo, fill_value=0).reset_index()

    agg["fecha"] = agg["mes"].dt.to_timestamp(how="end").dt.normalize()
    agg["ticket_promedio"] = (agg["gasto"] / agg["ordenes"]).where(agg["ordenes"] > 0, 0)
    return agg[["categoria", "fecha", "gasto", "ordenes", "ticket_promedio"]].sort_values(
        ["categoria", "fecha"]
    ).reset_index(drop=True)


def construir_drivers_mensuales(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drivers internos mensuales (señales coincidentes derivadas del propio dataset).

    Indexado por mes (fin de mes, igual que `construir_serie_total`) con:
      - `n_entidades_activas`  : nº de ENTIDAD distintas que compraron en el mes.
      - `n_proveedores`        : nº de RUC_PROVEEDOR distintos.
      - `n_categorias_activas` : nº de ACUERDO_MARCO distintos.
      - `share_<TIPO>`         : proporción de órdenes por TIPO_PROCEDIMIENTO.

    Son señales **coincidentes**: para pronosticar deben usarse REZAGADAS (lo hace
    `modelos.py`), nunca contemporáneas, para no inducir fuga de información.
    """
    s = df.dropna(subset=[config.COL_FECHA_FORMALIZACION]).copy()
    s["mes"] = s[config.COL_FECHA_FORMALIZACION].dt.to_period("M")

    aggs = {"n_categorias_activas": (config.COL_ACUERDO_MARCO, "nunique")}
    if config.COL_ENTIDAD in s.columns:
        aggs["n_entidades_activas"] = (config.COL_ENTIDAD, "nunique")
    if "RUC_PROVEEDOR" in s.columns:
        aggs["n_proveedores"] = ("RUC_PROVEEDOR", "nunique")
    elif config.COL_PROVEEDOR in s.columns:
        aggs["n_proveedores"] = (config.COL_PROVEEDOR, "nunique")

    drivers = s.groupby("mes").agg(**aggs)

    # Composición por tipo de procedimiento (proporción de órdenes del mes).
    if config.COL_TIPO_PROC in s.columns:
        comp = (
            s.groupby(["mes", config.COL_TIPO_PROC]).size().unstack(fill_value=0)
        )
        comp = comp.div(comp.sum(axis=1).where(comp.sum(axis=1) > 0, 1), axis=0)
        comp.columns = [
            "share_" + re.sub(r"\W+", "_", str(c)).strip("_").lower() for c in comp.columns
        ]
        drivers = drivers.join(comp)

    rango = pd.period_range(drivers.index.min(), drivers.index.max(), freq="M")
    drivers = drivers.reindex(rango, fill_value=0)
    drivers.index = drivers.index.to_timestamp(how="end").normalize()
    drivers.index.name = "fecha"
    return drivers


def construir_series(
    df: pd.DataFrame, guardar: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Orquesta la construcción de las series y la detección de meses incompletos.

    Devuelve
    --------
    (serie_total, serie_categoria, deteccion) donde `serie_total` ya viene
    recortada de los meses incompletos detectados.
    """
    serie_total = construir_serie_total(df)
    deteccion = detectar_meses_incompletos(serie_total)
    serie_total_recortada = recortar_incompletos(serie_total, deteccion)
    serie_categoria = construir_serie_por_categoria(df)

    if guardar:
        config.asegurar_directorios()
        serie_total_recortada.to_csv(config.RUTA_SERIE_MENSUAL)
        serie_categoria.to_csv(config.RUTA_SERIE_CATEGORIA)

    return serie_total_recortada, serie_categoria, deteccion
