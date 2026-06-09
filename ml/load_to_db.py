"""
load_to_db.py
=============
ETL: carga los artefactos limpios del ML a la base de datos.

Es el paso "limpiar -> usar en base de datos": lee los resultados ya generados por
los Módulos A y B (parquet/CSV) y los vuelca a tablas SQL mediante el engine
compartido (`api/app/db.py`). Es idempotente: cada tabla se reemplaza por completo.

Tablas creadas:
    ordenes                 <- dataset_limpio.parquet (~483k filas)
    alertas                 <- alertas.csv (rankeadas por riesgo)
    serie_mensual_total     <- serie_mensual_total.csv
    serie_mensual_categoria <- serie_mensual_por_categoria.csv
    pronostico              <- pronostico_final.csv
    comparacion_modelos     <- comparacion_modelos.csv

Uso (desde la raíz del proyecto):
    python -m ml.load_to_db                 # carga todo
    python -m ml.load_to_db --skip-ordenes  # omite la tabla grande (más rápido)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
from sqlalchemy import text

from api.app.db import DATABASE_URL, engine
from ml.shared import config

logger = logging.getLogger(__name__)

CHUNK = 5000  # filas por lote en la carga de la tabla grande


def _cargar_df(df: pd.DataFrame, tabla: str, chunksize: int | None = None) -> None:
    # Sin method="multi": evita el tope de variables por sentencia de SQLite
    # (máx. 999). El executemany por defecto es seguro en SQLite y Postgres.
    df.to_sql(tabla, engine, if_exists="replace", index=False, chunksize=chunksize)
    logger.info("Tabla '%s' cargada: %d filas, %d columnas", tabla, len(df), df.shape[1])


def _crear_indices() -> None:
    """Índices para consultas/paginación del API (compatibles SQLite y Postgres)."""
    indices = [
        "CREATE INDEX IF NOT EXISTS ix_alertas_puntaje ON alertas (\"PUNTAJE_RIESGO\")",
        "CREATE INDEX IF NOT EXISTS ix_alertas_nivel ON alertas (\"NIVEL\")",
        "CREATE INDEX IF NOT EXISTS ix_alertas_tipo ON alertas (\"TIPO_ANOMALIA\")",
        "CREATE INDEX IF NOT EXISTS ix_ordenes_fecha ON ordenes (\"FECHA_FORMALIZACION\")",
        "CREATE INDEX IF NOT EXISTS ix_ordenes_acuerdo ON ordenes (\"ACUERDO_MARCO\")",
    ]
    with engine.begin() as con:
        for sql in indices:
            try:
                con.execute(text(sql))
            except Exception as e:  # un índice sobre tabla ausente no debe frenar el ETL
                logger.warning("No se pudo crear índice (%s): %s", sql.split(" ON ")[0], e)
    logger.info("Índices creados/verificados.")


def cargar_alertas() -> None:
    """alertas.csv -> tabla 'alertas', rankeada por PUNTAJE_RIESGO (RANK=1 más riesgosa)."""
    ruta = config.RUTA_ALERTAS_CSV
    if not ruta.exists():
        logger.warning("No existe %s; se omite 'alertas'.", ruta)
        return
    df = pd.read_csv(ruta, encoding="utf-8")
    df = df.sort_values("PUNTAJE_RIESGO", ascending=False, kind="stable").reset_index(drop=True)
    df.insert(0, "RANK", df.index + 1)
    _cargar_df(df, "alertas")


def cargar_series_y_pronostico() -> None:
    mapeo = {
        config.RUTA_SERIE_MENSUAL: "serie_mensual_total",
        config.RUTA_SERIE_CATEGORIA: "serie_mensual_categoria",
        config.RUTA_PRONOSTICO: "pronostico",
        config.RUTA_METRICAS: "comparacion_modelos",
    }
    for ruta, tabla in mapeo.items():
        if ruta.exists():
            _cargar_df(pd.read_csv(ruta, encoding="utf-8"), tabla)
        else:
            logger.warning("No existe %s; se omite '%s'.", ruta, tabla)


def cargar_ordenes() -> None:
    """dataset_limpio.parquet -> tabla 'ordenes' (en lotes)."""
    ruta = config.RUTA_LIMPIO_PARQUET
    if not ruta.exists():
        logger.warning("No existe %s; se omite 'ordenes'.", ruta)
        return
    df = pd.read_parquet(ruta)
    # Fechas a texto ISO para evitar incompatibilidades de tipo entre motores.
    for col in df.select_dtypes(include=["datetime64[ns]", "datetimetz"]).columns:
        df[col] = df[col].astype(str)
    _cargar_df(df, "ordenes", chunksize=CHUNK)


def ejecutar_etl(con_ordenes: bool = True) -> None:
    config.asegurar_directorios()
    logger.info("Base de datos destino: %s", DATABASE_URL)
    cargar_series_y_pronostico()
    cargar_alertas()
    if con_ordenes:
        cargar_ordenes()
    _crear_indices()
    logger.info("ETL completado.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    parser = argparse.ArgumentParser(description="ETL: artefactos del ML -> base de datos")
    parser.add_argument("--skip-ordenes", action="store_true", help="Omite la tabla grande 'ordenes'.")
    args = parser.parse_args()
    ejecutar_etl(con_ordenes=not args.skip_ordenes)
