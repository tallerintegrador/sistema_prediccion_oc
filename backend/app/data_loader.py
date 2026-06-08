"""
data_loader.py
==============
Capa de acceso a datos del backend.

Carga los CSV de resultados (Módulos A y B) y los mantiene en caché en memoria
para no releerlos en cada petición. Todos los archivos se leen en UTF-8 (se
verificó que `alertas.csv` y las series están en esa codificación).

Regla del proyecto: aquí SOLO se LEE. No se reentrena ni se reprocesa nada.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

import pandas as pd

from . import config


# ---------------------------------------------------------------------------
# Utilidades internas
# ---------------------------------------------------------------------------
def _leer_csv(ruta: Path) -> pd.DataFrame:
    """Lee un CSV de resultados en UTF-8 o lanza un error claro si no existe."""
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de resultados esperado: {ruta}. "
            "Verifica que los Módulos A y B ya generaron sus salidas en outputs/."
        )
    return pd.read_csv(ruta, encoding="utf-8")


def df_a_registros(df: pd.DataFrame) -> list[dict]:
    """
    Convierte un DataFrame a una lista de dicts lista para JSON.

    Usa el serializador de pandas para que los NaN se vuelvan `null` y los tipos
    de numpy (int64, float64) se conviertan a tipos nativos de Python; así
    FastAPI los serializa sin problemas. `force_ascii=False` preserva las tildes.
    """
    return json.loads(df.to_json(orient="records", force_ascii=False))


# ---------------------------------------------------------------------------
# Cargas cacheadas (se calculan una sola vez por proceso)
# ---------------------------------------------------------------------------
@functools.lru_cache(maxsize=1)
def cargar_serie_total() -> pd.DataFrame:
    """Serie histórica mensual total: fecha, gasto, ordenes, ticket_promedio."""
    df = _leer_csv(config.RUTA_SERIE_TOTAL)
    return df


@functools.lru_cache(maxsize=1)
def cargar_pronostico() -> pd.DataFrame:
    """Pronóstico del Módulo A: fecha, pred, lower, upper."""
    df = _leer_csv(config.RUTA_PRONOSTICO)
    return df


@functools.lru_cache(maxsize=1)
def cargar_serie_categoria() -> pd.DataFrame:
    """Serie mensual por categoría (ACUERDO_MARCO), formato ancho."""
    df = _leer_csv(config.RUTA_SERIE_CATEGORIA)
    return df


@functools.lru_cache(maxsize=1)
def cargar_alertas() -> pd.DataFrame:
    """
    Alertas del Módulo B, ya rankeadas por riesgo.

    Se reordena de forma defensiva por PUNTAJE_RIESGO descendente y se añade una
    columna RANK (1 = más riesgosa) para que el ranking sea estable y explícito.
    """
    df = _leer_csv(config.RUTA_ALERTAS)
    df = df.sort_values("PUNTAJE_RIESGO", ascending=False, kind="stable").reset_index(drop=True)
    df.insert(0, "RANK", df.index + 1)
    return df


@functools.lru_cache(maxsize=1)
def leer_mejor_modelo() -> str | None:
    """Lee el nombre del mejor modelo del Módulo A, si está disponible."""
    ruta = config.RUTA_MEJOR_MODELO
    if not ruta.exists():
        return None
    texto = ruta.read_text(encoding="utf-8").strip()
    # El archivo tiene el formato "Mejor modelo del Módulo A: Ensemble".
    return texto.split(":", 1)[-1].strip() if ":" in texto else texto
