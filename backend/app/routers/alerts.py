"""
alerts.py
=========
Endpoint /alerts: lista de alertas del Módulo B, rankeadas por riesgo, con
filtros opcionales y paginación.

Filtros:
  - type  : tipo de anomalía (TIPO_ANOMALIA), coincidencia exacta sin distinguir
            mayúsculas/acentos según los valores reales del CSV.
  - level : nivel de riesgo en inglés -> high / medium (se traduce a alto / medio).

Paginación:
  - page      : número de página (1 en adelante).
  - page_size : tamaño de página (por defecto 50, máximo 500).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from .. import config, data_loader
from ..schemas import AlertItem, AlertsFilters, AlertsResponse

router = APIRouter(tags=["alerts"])


@router.get("/alerts", response_model=AlertsResponse, summary="Alertas rankeadas")
def alerts(
    type: str | None = Query(
        default=None,
        description="Filtra por tipo de anomalía (TIPO_ANOMALIA).",
    ),
    level: str | None = Query(
        default=None,
        description="Filtra por nivel de riesgo: 'high' o 'medium'.",
    ),
    page: int = Query(default=1, ge=1, description="Número de página (>=1)."),
    page_size: int = Query(default=50, ge=1, le=500, description="Tamaño de página (1-500)."),
) -> AlertsResponse:
    """Devuelve una página de alertas según los filtros indicados."""
    df = data_loader.cargar_alertas()

    # Catálogo de valores disponibles (para que el frontend arme los filtros).
    available_types = sorted(df["TIPO_ANOMALIA"].dropna().unique().tolist())
    available_levels = sorted(df["NIVEL"].dropna().unique().tolist())

    # --- Filtro por tipo (case-insensitive) -------------------------------
    if type is not None:
        df = df[df["TIPO_ANOMALIA"].str.casefold() == type.casefold()]

    # --- Filtro por nivel (high/medium -> alto/medio) ---------------------
    if level is not None:
        nivel_es = config.MAPEO_NIVEL.get(level.casefold())
        if nivel_es is None:
            raise HTTPException(
                status_code=422,
                detail=f"Nivel inválido: '{level}'. Usa uno de: {list(config.MAPEO_NIVEL.keys())}.",
            )
        df = df[df["NIVEL"] == nivel_es]

    # --- Paginación -------------------------------------------------------
    total = int(len(df))
    total_pages = (total + page_size - 1) // page_size if total else 0
    inicio = (page - 1) * page_size
    fin = inicio + page_size
    pagina = df.iloc[inicio:fin]

    return AlertsResponse(
        total=total,
        page=page,
        page_size=page_size,
        total_pages=total_pages,
        filters=AlertsFilters(type=type, level=level),
        available_types=available_types,
        available_levels=available_levels,
        items=[AlertItem(**registro) for registro in data_loader.df_a_registros(pagina)],
    )
