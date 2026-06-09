"""
categories.py
=============
Endpoint /categories: gasto por categoría (ACUERDO_MARCO).

Parte de serie_mensual_por_categoria.csv (formato ancho: una columna por
categoría). Suma cada columna para obtener el gasto acumulado por categoría y
lo devuelve ordenado de mayor a menor, con su participación sobre el total.
"""

from __future__ import annotations

from fastapi import APIRouter

from .. import data_loader
from ..schemas import CategoriesResponse, CategoryItem

router = APIRouter(tags=["categories"])


@router.get("/categories", response_model=CategoriesResponse, summary="Gasto por categoría")
def categories() -> CategoriesResponse:
    """Gasto acumulado por ACUERDO_MARCO, ordenado de mayor a menor."""
    df = data_loader.cargar_serie_categoria()

    # La primera columna es 'fecha'; el resto son categorías. Se suman a lo largo
    # del tiempo para obtener el gasto total de cada una.
    columnas_categoria = [c for c in df.columns if c.lower() != "fecha"]
    totales = df[columnas_categoria].sum().sort_values(ascending=False)

    total_global = float(totales.sum())
    items = [
        CategoryItem(
            category=str(categoria),
            total_spend=float(monto),
            # Participación sobre el gasto total (evita división por cero).
            share=float(monto / total_global) if total_global else 0.0,
        )
        for categoria, monto in totales.items()
    ]

    return CategoriesResponse(
        total_spend=total_global,
        n_categories=len(items),
        categories=items,
    )
