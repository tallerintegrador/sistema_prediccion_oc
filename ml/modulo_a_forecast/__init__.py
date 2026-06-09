"""
Paquete `src.fases` — las FASES del Módulo A, una por archivo y en ORDEN
(de la extracción al entrenamiento y pronóstico):

    f01_ingesta         -> extracción/ETL: lee y consolida los CSV mensuales
    f02_limpieza        -> limpieza y validación del consolidado
    f03_serie_temporal  -> serie mensual del gasto + panel por categoría + drivers
    f04_eda             -> análisis exploratorio (figuras 01–08)
    f05_features        -> ingeniería de características de calendario (insumo de f06)
    f06_modelado        -> catálogo de modelos de pronóstico (entrenamiento)
    f07_evaluacion      -> backtest, selección del mejor modelo y pronóstico final

El orquestador (`src.pipeline`) ejecuta estas fases en orden. Cada módulo también
puede ejecutarse de forma aislada con `python -m src.fases.fNN_<nombre>`.
"""
