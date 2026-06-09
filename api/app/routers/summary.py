"""
summary.py
==========
Endpoint /summary: indicadores clave para el tablero.

Combina los resultados de ambos módulos:
  - Módulo A: gasto y órdenes del último periodo, totales y pronóstico del mes próximo.
  - Módulo B: número de alertas (total / alto / medio).
"""

from __future__ import annotations

from fastapi import APIRouter

from .. import data_loader
from ..schemas import (
    AlertsBreakdown,
    DataRange,
    NextMonthForecast,
    SummaryResponse,
)

router = APIRouter(tags=["summary"])


@router.get("/summary", response_model=SummaryResponse, summary="Indicadores clave")
def summary() -> SummaryResponse:
    """Devuelve los KPIs principales del sistema."""
    df_hist = data_loader.cargar_serie_total()
    df_fc = data_loader.cargar_pronostico()
    df_alertas = data_loader.cargar_alertas()

    # --- Módulo A: último periodo observado y totales históricos ----------
    ultimo = df_hist.iloc[-1]
    next_mes = df_fc.iloc[0]  # primera fila del pronóstico = próximo mes

    # --- Módulo B: conteo de alertas por nivel ----------------------------
    conteo_niveles = df_alertas["NIVEL"].value_counts()

    return SummaryResponse(
        last_period=str(ultimo["fecha"]),
        last_period_spend=float(ultimo["gasto"]),
        last_period_orders=int(ultimo["ordenes"]),
        total_spend=float(df_hist["gasto"].sum()),
        total_orders=int(df_hist["ordenes"].sum()),
        n_categories=len(
            [c for c in data_loader.cargar_serie_categoria().columns if c.lower() != "fecha"]
        ),
        alerts=AlertsBreakdown(
            total=int(len(df_alertas)),
            high=int(conteo_niveles.get("alto", 0)),
            medium=int(conteo_niveles.get("medio", 0)),
        ),
        next_month_forecast=NextMonthForecast(
            fecha=str(next_mes["fecha"]),
            pred=float(next_mes["pred"]),
            lower=float(next_mes["lower"]),
            upper=float(next_mes["upper"]),
        ),
        best_model=data_loader.leer_mejor_modelo(),
        data_range=DataRange(
            start=str(df_hist.iloc[0]["fecha"]),
            end=str(ultimo["fecha"]),
        ),
    )
