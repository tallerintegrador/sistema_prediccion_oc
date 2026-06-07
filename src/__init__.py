"""
Paquete `src` del SistemaPrediccionOC — Módulo A (pronóstico del gasto público
en órdenes de compra de los Acuerdos Marco de PERÚ COMPRAS).

Submódulos del Módulo A (pronóstico):
    config          -> rutas y parámetros
    ingesta         -> lectura y consolidación de los CSV
    limpieza        -> limpieza y validación
    eda             -> análisis exploratorio y figuras
    serie_temporal  -> construcción de la serie mensual
    modelos         -> entrenamiento de los modelos
    evaluacion      -> métricas, comparación y pronóstico final

Submódulos del Módulo B (detección de anomalías; reutilizan la capa de datos):
    config_b             -> rutas y parámetros del Módulo B
    features_anomalias   -> ingeniería de variables de anomalía
    eda_anomalias        -> EDA orientado a anomalías y figuras
    modelos_anomalias    -> Isolation Forest, autoencoder y análisis de grafo
    riesgo               -> puntaje, niveles, tipo y ranking de alertas
    evaluacion_anomalias -> precision@k, inyección sintética y concordancia
"""
