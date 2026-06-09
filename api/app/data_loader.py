"""
data_loader.py
==============
Capa de acceso a datos del backend.

Lee las series, el pronóstico y las alertas desde la **base de datos** (las pobló
el ETL `ml/load_to_db.py`) y las métricas del ML desde los **JSON** que genera
`ml/metrics_export.py`. Mantiene los resultados en caché en memoria para no
reconsultar en cada petición.

Regla del proyecto: aquí SOLO se LEE. No se reentrena ni se reprocesa nada.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import pandas as pd

from . import config
from .db import engine


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------
def _leer_tabla(nombre: str) -> pd.DataFrame:
    """Lee una tabla completa de la base de datos, con error claro si falta."""
    try:
        return pd.read_sql_table(nombre, engine)
    except ValueError as e:
        raise FileNotFoundError(
            f"No se encontró la tabla '{nombre}' en la base de datos. "
            "Ejecuta el ETL: `python -m ml.load_to_db`."
        ) from e


def _leer_json(ruta: Path) -> dict:
    """Lee un JSON de métricas, con error claro si no existe."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de métricas: {ruta}. "
            "Ejecútalo con `python -m ml.metrics_export`."
        )
    return json.loads(ruta.read_text(encoding="utf-8"))


def df_a_registros(df: pd.DataFrame) -> list[dict]:
    """
    Convierte un DataFrame a una lista de dicts lista para JSON.

    Usa el serializador de pandas para que los NaN se vuelvan `null` y los tipos
    de numpy (int64, float64) se conviertan a tipos nativos de Python; así
    FastAPI los serializa sin problemas. `force_ascii=False` preserva las tildes.
    """
    return json.loads(df.to_json(orient="records", force_ascii=False))


# ---------------------------------------------------------------------------
# Cargas cacheadas desde la base de datos (una sola vez por proceso)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def cargar_serie_total() -> pd.DataFrame:
    """Serie histórica mensual total: fecha, gasto, ordenes, ticket_promedio."""
    return _leer_tabla("serie_mensual_total")


@functools.lru_cache(maxsize=1)
def cargar_pronostico() -> pd.DataFrame:
    """Pronóstico del Módulo A: fecha, pred, lower, upper."""
    return _leer_tabla("pronostico")


@functools.lru_cache(maxsize=1)
def cargar_serie_categoria() -> pd.DataFrame:
    """Serie mensual por categoría (ACUERDO_MARCO), formato ancho."""
    return _leer_tabla("serie_mensual_categoria")


@functools.lru_cache(maxsize=1)
def cargar_alertas() -> pd.DataFrame:
    """
    Alertas del Módulo B, ya rankeadas por riesgo (RANK=1 más riesgosa).

    El ETL ya las guardó con la columna RANK; se reordena de forma defensiva por
    RANK para garantizar el orden independientemente del motor de base de datos.
    """
    df = _leer_tabla("alertas")
    if "RANK" in df.columns:
        df = df.sort_values("RANK", kind="stable").reset_index(drop=True)
    return df


@functools.lru_cache(maxsize=1)
def leer_mejor_modelo() -> str | None:
    """Nombre del mejor modelo del Módulo A (de las métricas JSON o del .txt)."""
    if config.RUTA_METRICAS_A.exists():
        datos = json.loads(config.RUTA_METRICAS_A.read_text(encoding="utf-8"))
        if datos.get("best_model"):
            return datos["best_model"]
    ruta = config.RUTA_MEJOR_MODELO
    if not ruta.exists():
        return None
    texto = ruta.read_text(encoding="utf-8").strip()
    # El archivo tiene el formato "Mejor modelo del Módulo A: Ensemble".
    return texto.split(":", 1)[-1].strip() if ":" in texto else texto


# ---------------------------------------------------------------------------
# Métricas del ML (JSON generados por ml/metrics_export.py)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def cargar_metricas_a() -> dict:
    """Métricas del Módulo A: comparación de modelos, mejor modelo, pronóstico."""
    return _leer_json(config.RUTA_METRICAS_A)


@functools.lru_cache(maxsize=1)
def cargar_metricas_b() -> dict:
    """Métricas del Módulo B: precision@k, recall, concordancia, distribuciones."""
    return _leer_json(config.RUTA_METRICAS_B)
