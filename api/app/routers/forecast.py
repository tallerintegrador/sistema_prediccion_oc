"""
forecast.py
===========
Endpoint /forecast: serie histórica mensual + pronóstico de los próximos meses
con su intervalo de confianza.

Reutiliza directamente los CSV del Módulo A (serie_mensual_total.csv y
pronostico_final.csv); no carga el modelo .pkl porque el pronóstico ya está
guardado como dato legible.
"""

from __future__ import annotations

from fastapi import APIRouter

from .. import config, data_loader
from ..schemas import ForecastResponse

router = APIRouter(tags=["forecast"])


@router.get("/forecast", response_model=ForecastResponse, summary="Histórico + pronóstico")
def forecast() -> ForecastResponse:
    """Devuelve la serie observada y el pronóstico con intervalos lower/upper."""
    df_hist = data_loader.cargar_serie_total()
    df_fc = data_loader.cargar_pronostico()

    return ForecastResponse(
        confidence_level=config.NIVEL_CONFIANZA,
        best_model=data_loader.leer_mejor_modelo(),
        history=data_loader.df_a_registros(df_hist),
        forecast=data_loader.df_a_registros(df_fc),
    )
