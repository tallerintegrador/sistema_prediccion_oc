"""
schemas.py
==========
Modelos de respuesta (Pydantic) del API.

Definen y documentan la forma exacta de cada respuesta JSON; FastAPI los usa
también para generar la documentación interactiva en /docs.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    service: str
    version: str


# ---------------------------------------------------------------------------
# /forecast
# ---------------------------------------------------------------------------
class HistoryPoint(BaseModel):
    """Un mes de la serie histórica observada."""
    fecha: str
    gasto: float
    ordenes: int
    ticket_promedio: float | None = None


class ForecastPoint(BaseModel):
    """Un mes pronosticado con su intervalo de confianza."""
    fecha: str
    pred: float
    lower: float
    upper: float


class ForecastResponse(BaseModel):
    confidence_level: float = Field(description="Nivel de confianza del intervalo (p.ej. 0.95).")
    best_model: str | None = None
    history: list[HistoryPoint]
    forecast: list[ForecastPoint]


# ---------------------------------------------------------------------------
# /categories
# ---------------------------------------------------------------------------
class CategoryItem(BaseModel):
    category: str
    total_spend: float
    share: float = Field(description="Proporción [0,1] del gasto total acumulado.")


class CategoriesResponse(BaseModel):
    total_spend: float
    n_categories: int
    categories: list[CategoryItem]


# ---------------------------------------------------------------------------
# /summary
# ---------------------------------------------------------------------------
class NextMonthForecast(BaseModel):
    fecha: str
    pred: float
    lower: float
    upper: float


class AlertsBreakdown(BaseModel):
    total: int
    high: int
    medium: int


class DataRange(BaseModel):
    start: str
    end: str


class SummaryResponse(BaseModel):
    last_period: str = Field(description="Último mes con datos observados (YYYY-MM-DD).")
    last_period_spend: float
    last_period_orders: int
    total_spend: float = Field(description="Gasto histórico acumulado.")
    total_orders: int = Field(description="Órdenes históricas acumuladas.")
    n_categories: int
    alerts: AlertsBreakdown
    next_month_forecast: NextMonthForecast
    best_model: str | None = None
    data_range: DataRange


# ---------------------------------------------------------------------------
# /alerts
# ---------------------------------------------------------------------------
class AlertItem(BaseModel):
    RANK: int
    ORDEN_ELECTRONICA: str | None = None
    RUC_ENTIDAD: int | None = None
    ENTIDAD: str | None = None
    RUC_PROVEEDOR: int | None = None
    PROVEEDOR: str | None = None
    ACUERDO_MARCO: str | None = None
    FECHA_FORMALIZACION: str | None = None
    TOTAL: float | None = None
    TIPO_ANOMALIA: str | None = None
    PUNTAJE_RIESGO: float | None = None
    NIVEL: str | None = None
    MOTIVO: str | None = None
    SCORE_IF: float | None = None
    SCORE_RED: float | None = None


class AlertsFilters(BaseModel):
    type: str | None = None
    level: str | None = None


class AlertsResponse(BaseModel):
    total: int = Field(description="Total de alertas que cumplen el filtro.")
    page: int
    page_size: int
    total_pages: int
    filters: AlertsFilters
    available_types: list[str] = Field(description="Valores válidos para el filtro 'type'.")
    available_levels: list[str] = Field(description="Valores válidos para el filtro 'level'.")
    items: list[AlertItem]


# ---------------------------------------------------------------------------
# /metrics/forecast  (métricas del Módulo A)
# ---------------------------------------------------------------------------
class ModelMetric(BaseModel):
    """Una fila de la comparación de modelos (menor es mejor en todas)."""
    Modelo: str
    WAPE: float | None = None
    MASE: float | None = None
    MAE: float | None = None
    RMSE: float | None = None
    MAPE: float | None = None
    sMAPE: float | None = None
    MPE: float | None = None


class ForecastMetricsResponse(BaseModel):
    primary_metric: str = Field(description="Métrica usada para elegir el mejor modelo (WAPE).")
    confidence_level: float
    best_model: str | None = None
    model_comparison: list[ModelMetric]
    forecast: list[ForecastPoint]


# ---------------------------------------------------------------------------
# /models  (registro de modelos)
# ---------------------------------------------------------------------------
class ModelsResponse(BaseModel):
    best_model: str | None = None
    primary_metric: str
    models: list[str] = Field(description="Nombres de los modelos comparados.")
