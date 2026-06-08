"""
config_b.py
===========
Configuración central del **Módulo B** (detección de órdenes de compra con
comportamiento anómalo o de riesgo).

Este módulo es HERMANO e INDEPENDIENTE del Módulo A: comparte únicamente la
**capa de datos** (el dataset limpio que produjeron `ingesta.py` + `limpieza.py`).
Aquí solo se definen rutas de salida propias del B y **parámetros estructurales**
(tamaños de ventana, número de árboles, cuántas alertas mostrar, etc.).

IMPORTANTE — sobre los umbrales:
    Ningún umbral *sobre los datos* (punto de corte del fraccionamiento, tolerancia
    de IGV, cortes de monto atípico, cortes de nivel de riesgo) se fija aquí: TODOS
    se **derivan en tiempo de ejecución** a partir de la distribución real de los
    datos (ver `features_anomalias.py` y `riesgo.py`). Lo que vive aquí son
    decisiones de diseño (p. ej. "una ventana corta de fraccionamiento son 7 días"),
    que se documentan y justifican, no cifras inventadas sobre el negocio.
"""

from __future__ import annotations

from . import config  # reutiliza rutas y semilla del Módulo A (capa compartida)

# ---------------------------------------------------------------------------
# Reutilización de la capa de datos del Módulo A
# ---------------------------------------------------------------------------
# Ruta del dataset limpio generado por el Módulo A. El Módulo B lo CARGA; no
# vuelve a leer ni a limpiar los CSV.
RUTA_DATASET_LIMPIO = config.RUTA_LIMPIO_PARQUET
SEMILLA: int = config.SEMILLA  # misma semilla global -> reproducibilidad

# ---------------------------------------------------------------------------
# Rutas de salida propias del Módulo B (no pisan las del A)
# ---------------------------------------------------------------------------
DIR_FIGURAS = config.DIR_FIGURAS          # se reutiliza la carpeta de figuras
DIR_RESULTADOS = config.DIR_RESULTADOS    # se reutiliza la carpeta de resultados

RUTA_ALERTAS_CSV = DIR_RESULTADOS / "alertas.csv"
RUTA_INFORME_B = DIR_RESULTADOS / "informe_modulo_b.md"

# Prefijo de las figuras del B para no colisionar con las del A (01..11).
PREFIJO_FIG_B: str = "b"

# ---------------------------------------------------------------------------
# Columnas del dataset limpio que usa el Módulo B (nombres ya normalizados por
# la ingesta del A: MAYÚSCULAS, sin tildes en el identificador, espacios -> "_").
# ---------------------------------------------------------------------------
COL_ORDEN = config.COL_ORDEN                       # ORDEN_ELECTRONICA (id único)
COL_TOTAL = config.COL_TOTAL                       # TOTAL
COL_SUBTOTAL = config.COL_SUBTOTAL                 # SUB_TOTAL
COL_IGV = config.COL_IGV                           # IGV
COL_TIPO_PROC = config.COL_TIPO_PROC               # TIPO_PROCEDIMIENTO
COL_ACUERDO = config.COL_ACUERDO_MARCO             # ACUERDO_MARCO
COL_ENTIDAD = config.COL_ENTIDAD                   # ENTIDAD (razón social)
COL_PROVEEDOR = config.COL_PROVEEDOR               # PROVEEDOR (razón social)
COL_RUC_ENTIDAD = "RUC_ENTIDAD"
COL_RUC_PROVEEDOR = "RUC_PROVEEDOR"
COL_FECHA_FORM = config.COL_FECHA_FORMALIZACION    # FECHA_FORMALIZACION
COL_FECHA_ULT = "FECHA_ULTIMO_ESTADO"

# ---------------------------------------------------------------------------
# IGV: en Perú el IGV legal es el 18 % del subtotal. Esto es un HECHO legal, no
# un supuesto sobre los datos. La exploración lo confirma: el ratio IGV/SUB_TOTAL
# es exactamente 0.18 en ~96 % de las órdenes. La TOLERANCIA para marcar una
# orden como inconsistente NO se fija aquí: se deriva de la distribución real del
# ratio en features_anomalias.py.
IGV_LEGAL: float = 0.18

# ---------------------------------------------------------------------------
# Parámetros ESTRUCTURALES (decisiones de diseño, justificadas)
# ---------------------------------------------------------------------------
# Ventanas "cortas" para evaluar fraccionamiento: cuántas órdenes lanza un mismo
# par entidad–proveedor en pocos días. 7 y 30 días son ventanas estándar (una
# semana y un mes). Se reportan ambas; no son umbrales sobre los datos.
VENTANAS_FRACC_DIAS: tuple[int, ...] = (7, 30)

# "Cerca" del punto de corte del fraccionamiento: fracción por debajo del umbral
# (derivado de los datos) dentro de la cual se considera que una orden quedó
# "pegada" al límite. 0.90 = entre el 90 % y el 100 % del umbral. Es una banda de
# diseño; el umbral en soles SÍ se deriva de los datos.
BANDA_PROXIMIDAD_UMBRAL: float = 0.90

# Días considerados "cierre de mes" (últimos N días naturales del mes).
DIAS_CIERRE_MES: int = 5

# Isolation Forest (no supervisado). contamination='auto' deja que el algoritmo
# fije el umbral; el número de árboles y el muestreo se fijan para reproducir.
IF_N_ESTIMADORES: int = 300
IF_MAX_SAMPLES: int | str = "auto"
IF_CONTAMINACION: str | float = "auto"

# Autoencoder opcional (MLP de scikit-learn, sin dependencias extra). Si falla o
# se desactiva, el flujo continúa solo con Isolation Forest + reglas + grafo.
USAR_AUTOENCODER: bool = True
AE_CAPAS_OCULTAS: tuple[int, ...] = (8, 3, 8)   # cuello de botella en 3 dims
AE_MAX_ITER: int = 60

# Cortes de los NIVELES de riesgo (alto/medio/bajo). Son CUANTILES del propio
# puntaje combinado (se derivan de su distribución), no cortes absolutos.
# alto = por encima del cuantil 0.99 ; medio = por encima del 0.95.
CUANTIL_RIESGO_ALTO: float = 0.99
CUANTIL_RIESGO_MEDIO: float = 0.95

# Evaluación
TOP_K_ALERTAS: tuple[int, ...] = (20, 50, 100)   # precision@k a inspeccionar
N_ANOMALIAS_SINTETICAS: int = 200                # casos artificiales por tipo
