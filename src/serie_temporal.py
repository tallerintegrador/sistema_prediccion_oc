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

import pandas as pd

from . import config

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
