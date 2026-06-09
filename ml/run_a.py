"""
run_a.py
========
Punto de entrada del **Módulo A** (pronóstico del gasto público en órdenes de
compra). Ejecuta el pipeline completo, en 7 fases ordenadas y de punta a punta:

    1. Ingesta / ETL            5. Features de calendario
    2. Limpieza y validación    6. Modelado
    3. Serie temporal           7. Evaluación y pronóstico
    4. Análisis exploratorio

Toda la orquestación vive en `ml/modulo_a_forecast/pipeline.py`; aquí solo se
invoca. Los artefactos (dataset consolidado y limpio, figuras numeradas, métricas,
pronóstico e informe en Markdown) quedan en `artifacts/`.

Uso (desde la raíz del proyecto):
    python -m ml.run_a
"""

from __future__ import annotations

from ml.metrics_export import exportar_a
from ml.modulo_a_forecast.pipeline import ejecutar_pipeline

if __name__ == "__main__":
    ejecutar_pipeline()
    # Consolida las métricas del Módulo A en JSON para el API/frontend.
    exportar_a()
