"""
Paquete `src` del SistemaPrediccionOC — Módulo A (pronóstico del gasto público
en órdenes de compra de los Acuerdos Marco de PERÚ COMPRAS).

Arquitectura por fases:

    config          -> rutas y parámetros centrales (lo usa todo el proyecto)
    figuras         -> registro central de figuras (estilo + numeración ordenada)
    pipeline        -> orquestador: define y ejecuta las 7 fases en orden
    fases/          -> una fase por archivo, numeradas en orden de flujo:
        f01_ingesta         -> extracción/ETL de los CSV
        f02_limpieza        -> limpieza y validación
        f03_serie_temporal  -> serie mensual + panel por categoría + drivers
        f04_eda             -> análisis exploratorio (figuras 01–08)
        f05_features        -> features de calendario (insumo del modelado)
        f06_modelado        -> catálogo de modelos de pronóstico
        f07_evaluacion      -> backtest, selección y pronóstico final

Punto de entrada: `from src.pipeline import ejecutar_pipeline`.
"""
