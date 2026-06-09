"""
main.py
=======
Punto de entrada del API (FastAPI) del SistemaPrediccionOC.

Expone, vía REST, los resultados ya calculados por el Módulo A (pronóstico) y el
Módulo B (detección de anomalías), además de las MÉTRICAS del ML para el tablero
de sustentación. El backend solo LEE: la base de datos la puebla `ml/load_to_db.py`
y los JSON de métricas los genera `ml/metrics_export.py`.

Levantar (desde la raíz del proyecto):
    uvicorn api.app.main:app --reload

Documentación interactiva: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import config
from .db import DATABASE_URL
from .routers import alerts, categories, forecast, health, metrics, summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Valida artefactos al arrancar (avisa con claridad si falta algo)."""
    log.info("Base de datos: %s", DATABASE_URL)
    faltantes = [p for p in (config.RUTA_METRICAS_A, config.RUTA_METRICAS_B) if not p.exists()]
    if faltantes:
        log.warning(
            "Faltan JSON de métricas: %s. Ejecuta `python -m ml.metrics_export`.",
            ", ".join(p.name for p in faltantes),
        )
    if not config.DIR_FIGURAS.exists():
        log.warning("No existe el directorio de figuras: %s", config.DIR_FIGURAS)
    yield


app = FastAPI(
    title=config.NOMBRE_SERVICIO,
    version=config.VERSION_API,
    description=(
        "API que expone los resultados y las MÉTRICAS del pronóstico del gasto "
        "público (Módulo A) y de la detección de órdenes de compra anómalas (Módulo B)."
    ),
    lifespan=lifespan,
)

# --- CORS -----------------------------------------------------------------
# Orígenes permitidos configurables por entorno (ALLOWED_ORIGINS).
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ORIGENES_CORS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Registro de routers --------------------------------------------------
app.include_router(health.router)
app.include_router(summary.router)
app.include_router(forecast.router)
app.include_router(categories.router)
app.include_router(alerts.router)
app.include_router(metrics.router)

# --- Figuras estáticas del ML (PNGs) --------------------------------------
# Se sirven en /figures para incrustarlas en la página de Métricas ML.
if config.DIR_FIGURAS.exists():
    app.mount("/figures", StaticFiles(directory=str(config.DIR_FIGURAS)), name="figures")


@app.get("/", tags=["root"], summary="Información del API")
def root() -> dict:
    """Página raíz: lista los endpoints disponibles."""
    return {
        "service": config.NOMBRE_SERVICIO,
        "version": config.VERSION_API,
        "docs": "/docs",
        "endpoints": [
            "/health", "/summary", "/forecast", "/categories", "/alerts",
            "/metrics/forecast", "/metrics/anomalies", "/models", "/figures/{archivo}",
        ],
    }
