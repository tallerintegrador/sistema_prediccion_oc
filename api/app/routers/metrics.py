"""
routers/metrics.py
==================
Endpoints de MÉTRICAS del ML (para el tablero de sustentación / capstone).

Exponen, en JSON consumible por el frontend, todo lo que el ML evaluó:

    GET /metrics/forecast   -> comparación de modelos, mejor modelo, pronóstico (Módulo A)
    GET /metrics/anomalies  -> precision@k, recall por inyección, concordancia,
                               distribuciones y estadísticos de red (Módulo B)
    GET /models             -> registro compacto de modelos comparados

Los datos provienen de los JSON que genera `ml/metrics_export.py`.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import data_loader
from ..schemas import ForecastMetricsResponse, ModelsResponse

router = APIRouter(tags=["metrics"])


@router.get("/metrics/forecast", response_model=ForecastMetricsResponse,
            summary="Métricas del pronóstico (Módulo A)")
def metrics_forecast() -> dict:
    try:
        return data_loader.cargar_metricas_a()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/metrics/anomalies", summary="Métricas de detección de anomalías (Módulo B)")
def metrics_anomalies() -> dict:
    try:
        return data_loader.cargar_metricas_b()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/models", response_model=ModelsResponse, summary="Registro de modelos")
def models() -> dict:
    try:
        a = data_loader.cargar_metricas_a()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return {
        "best_model": a.get("best_model"),
        "primary_metric": a.get("primary_metric", "WAPE"),
        "models": [m.get("Modelo") for m in a.get("model_comparison", [])],
    }
