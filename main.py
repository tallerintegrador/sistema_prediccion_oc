"""
main.py
=======
Punto de entrada del **Módulo A** (pronóstico del gasto público en órdenes de
compra). Ejecuta el pipeline completo, en 7 fases ordenadas y de punta a punta:

    1. Ingesta / ETL            5. Features de calendario
    2. Limpieza y validación    6. Modelado
    3. Serie temporal           7. Evaluación y pronóstico
    4. Análisis exploratorio

Toda la orquestación vive en `src/pipeline.py`; aquí solo se invoca. Los
artefactos (dataset consolidado y limpio, figuras numeradas, métricas, pronóstico
e informe en Markdown) quedan en `outputs/`.

Uso:
    python main.py
"""

from __future__ import annotations

from src.pipeline import ejecutar_pipeline

if __name__ == "__main__":
    ejecutar_pipeline()
