"""
main.py
=======
Punto de entrada del API (FastAPI) del SistemaPrediccionOC.

Expone, vía REST, los resultados ya calculados por el Módulo A (pronóstico) y el
Módulo B (detección de anomalías). El backend solo lee los artefactos de
`outputs/`; no reentrena ni reprocesa nada.

Levantar (desde la raíz del proyecto):
    uvicorn backend.app.main:app --reload

Documentación interactiva: http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import config
from .routers import alerts, categories, forecast, health, summary

app = FastAPI(
    title=config.NOMBRE_SERVICIO,
    version=config.VERSION_API,
    description=(
        "API que expone los resultados del pronóstico del gasto público (Módulo A) "
        "y de la detección de órdenes de compra anómalas (Módulo B)."
    ),
)

# --- CORS -----------------------------------------------------------------
# Se habilita CORS abierto para que cualquier frontend (en desarrollo) pueda
# consumir el API. En producción conviene restringir allow_origins al dominio real.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Registro de routers (rutas en inglés) --------------------------------
app.include_router(health.router)
app.include_router(summary.router)
app.include_router(forecast.router)
app.include_router(categories.router)
app.include_router(alerts.router)


@app.get("/", tags=["root"], summary="Información del API")
def root() -> dict:
    """Página raíz: lista los endpoints disponibles."""
    return {
        "service": config.NOMBRE_SERVICIO,
        "version": config.VERSION_API,
        "docs": "/docs",
        "endpoints": ["/health", "/summary", "/forecast", "/categories", "/alerts"],
    }
