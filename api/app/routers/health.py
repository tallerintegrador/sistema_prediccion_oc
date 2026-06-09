"""
health.py
=========
Endpoint de salud: confirma que el API está vivo.
"""

from __future__ import annotations

from fastapi import APIRouter

from .. import config
from ..schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse, summary="Estado del servicio")
def health() -> HealthResponse:
    """Devuelve un 'ok' simple para verificar que el API responde."""
    return HealthResponse(
        status="ok",
        service=config.NOMBRE_SERVICIO,
        version=config.VERSION_API,
    )
