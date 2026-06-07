"""
limpieza.py
===========
Etapa de LIMPIEZA y VALIDACIÓN del dataset consolidado.

Operaciones (todas justificadas por lo observado en la exploración real):

* `strip()` a todas las columnas de texto: el archivo UTF-8 (202403) traía
  TABULADORES al inicio de varios campos ('\\tACEPTADA', '\\t2024-03-08...').
* Conversión de montos a número: `SUB_TOTAL`, `IGV`, `TOTAL` (decimal '.').
* Conversión de fechas probando los DOS formatos hallados
  ('%Y-%m-%d %H:%M:%S' y '%d/%m/%Y %H:%M').
* Filtrado de estados: se excluyen 'RESUELTA' y 'ORDEN DE COMPRA NULA' por no
  representar gasto efectivo.
* Reporte de nulos, duplicados y registros inválidos (sin frenar el proceso).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .. import config

logger = logging.getLogger(__name__)


def _limpiar_texto(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica strip() a todas las columnas de texto y convierte '' en NaN."""
    df = df.copy()
    columnas_texto = df.select_dtypes(include="object").columns
    for col in columnas_texto:
        df[col] = df[col].str.strip().replace("", np.nan)
    return df


def parsear_fechas(serie: pd.Series) -> pd.Series:
    """
    Convierte una serie de texto a `datetime` probando, en orden, todos los
    formatos de `config.FORMATOS_FECHA`. Cada formato rellena los nulos que dejó
    el anterior, de modo que se aprovechan ambos sin sobrescribir aciertos.

    Devuelve una serie `datetime64`; los valores irreparables quedan como NaT.
    """
    serie = serie.astype("string").str.strip()
    resultado = pd.to_datetime(serie, errors="coerce", format=config.FORMATOS_FECHA[0])
    for fmt in config.FORMATOS_FECHA[1:]:
        faltan = resultado.isna()
        if not faltan.any():
            break
        resultado.loc[faltan] = pd.to_datetime(
            serie[faltan], errors="coerce", format=fmt
        )
    return resultado


def parsear_montos(serie: pd.Series) -> pd.Series:
    """Convierte una columna de monto a float (coerciendo errores a NaN)."""
    return pd.to_numeric(serie, errors="coerce")


def limpiar(df: pd.DataFrame, guardar: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Limpia y valida el DataFrame consolidado.

    Parámetros
    ----------
    df : pd.DataFrame
        Consolidado crudo (salida de `ingesta.consolidar`).
    guardar : bool
        Si True, guarda el dataset limpio en parquet.

    Devuelve
    --------
    (df_limpio, reporte) : tuple[pd.DataFrame, dict]
        DataFrame limpio (solo órdenes válidas) y un diccionario con el detalle
        de la validación (nulos, duplicados, fechas/montos inválidos, estados).
    """
    reporte: dict = {}
    n_inicial = len(df)
    reporte["filas_iniciales"] = n_inicial

    # 1) Limpieza de texto (tabuladores, espacios).
    df = _limpiar_texto(df)

    # 2) Nulos por columna ANTES de transformar (foto de calidad).
    reporte["nulos_por_columna"] = (
        (df.isna().mean() * 100).round(2).sort_values(ascending=False).to_dict()
    )

    # 3) Conversión de montos.
    montos_invalidos: dict[str, int] = {}
    for col in config.COLUMNAS_MONTO:
        if col in df.columns:
            convertida = parsear_montos(df[col])
            montos_invalidos[col] = int(convertida.isna().sum())
            df[col] = convertida
    reporte["montos_no_convertibles"] = montos_invalidos

    # 4) Conversión de fechas.
    fechas_invalidas: dict[str, int] = {}
    for col in config.COLUMNAS_FECHA:
        if col in df.columns:
            convertida = parsear_fechas(df[col])
            fechas_invalidas[col] = int(convertida.isna().sum())
            df[col] = convertida
    reporte["fechas_no_convertibles"] = fechas_invalidas

    # 5) Duplicados (exactos y por identificador de orden).
    cols_negocio = [c for c in df.columns if not c.startswith("__")]
    reporte["duplicados_exactos"] = int(df.duplicated(subset=cols_negocio).sum())
    if config.COL_ORDEN in df.columns:
        reporte["duplicados_por_orden"] = int(
            df.duplicated(subset=[config.COL_ORDEN]).sum()
        )
        # Si hubiera duplicados de orden, nos quedamos con el primero.
        df = df.drop_duplicates(subset=[config.COL_ORDEN], keep="first")

    # 6) Distribución de estados (sobre datos ya 'stripeados').
    if config.COL_ESTADO in df.columns:
        reporte["estados"] = df[config.COL_ESTADO].value_counts(dropna=False).to_dict()

    # 7) Registros inválidos para el pronóstico: sin fecha de formalización o sin
    #    TOTAL. Se registran y se descartan (no pueden ubicarse en la serie).
    sin_fecha = df[config.COL_FECHA_FORMALIZACION].isna()
    sin_total = df[config.COL_TOTAL].isna()
    reporte["sin_fecha_formalizacion"] = int(sin_fecha.sum())
    reporte["sin_total"] = int(sin_total.sum())
    df = df[~(sin_fecha | sin_total)].copy()

    # 8) Filtrado por estado: excluir órdenes resueltas/anuladas.
    if config.COL_ESTADO in df.columns:
        antes = len(df)
        excluidas_mask = df[config.COL_ESTADO].isin(config.ESTADOS_EXCLUIDOS)
        reporte["ordenes_excluidas_por_estado"] = int(excluidas_mask.sum())
        df = df[~excluidas_mask].copy()
        reporte["filas_tras_filtro_estado"] = len(df)
        logger.info(
            "Excluidas %d órdenes por estado %s",
            antes - len(df),
            config.ESTADOS_EXCLUIDOS,
        )

    # 9) Outliers de monto (informativo, NO se eliminan: son montos reales de
    #    grandes compras). Se reporta con el criterio del rango intercuartílico.
    total = df[config.COL_TOTAL]
    q1, q3 = total.quantile(0.25), total.quantile(0.75)
    iqr = q3 - q1
    limite_sup = q3 + 3 * iqr  # criterio amplio (x3) por la cola muy larga
    reporte["outliers_total_iqr"] = {
        "limite_superior": float(limite_sup),
        "n_por_encima": int((total > limite_sup).sum()),
        "monto_maximo": float(total.max()),
    }

    reporte["filas_finales"] = len(df)
    logger.info(
        "Limpieza: %d filas iniciales -> %d filas válidas finales.",
        n_inicial,
        len(df),
    )

    if guardar:
        config.asegurar_directorios()
        df.to_parquet(config.RUTA_LIMPIO_PARQUET, index=False)
        logger.info("Dataset limpio guardado en %s", config.RUTA_LIMPIO_PARQUET)

    return df, reporte


if __name__ == "__main__":
    from . import f01_ingesta as ingesta

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    _crudo, _ = ingesta.consolidar(guardar=False)
    _limpio, _rep = limpiar(_crudo)
    print(f"Filas finales válidas: {_rep['filas_finales']:,}")
