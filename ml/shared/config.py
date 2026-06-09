"""
config.py
=========
Configuración central del Módulo A (pronóstico del gasto en órdenes de compra).

Aquí se definen TODAS las rutas (relativas al proyecto, para que sea reproducible
en cualquier máquina) y los parámetros del flujo. Ningún otro módulo debe usar
rutas absolutas: todo se importa desde aquí.

Los valores de codificación, separador, formatos de fecha y estados válidos NO se
inventaron: se descubrieron explorando empíricamente los CSV reales de `data/`.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas del proyecto (todas relativas a la raíz, calculada desde este archivo)
# ---------------------------------------------------------------------------
# config.py vive en ml/shared/; la raíz del proyecto es parents[2]:
# ml/shared/config.py -> [0]=shared, [1]=ml, [2]=raíz del proyecto.
RAIZ_PROYECTO: Path = Path(__file__).resolve().parents[2]

DIR_DATOS: Path = RAIZ_PROYECTO / "data"                # CSV originales (no se modifican)
DIR_OUTPUTS: Path = RAIZ_PROYECTO / "artifacts"
DIR_FIGURAS: Path = DIR_OUTPUTS / "figuras"             # todas las gráficas
DIR_MODELOS: Path = DIR_OUTPUTS / "modelos"             # modelos entrenados serializados
DIR_RESULTADOS: Path = DIR_OUTPUTS / "resultados"       # tablas, informe EDA, métricas

# Artefactos intermedios (para no reprocesar cada vez)
RUTA_CONSOLIDADO_PARQUET: Path = DIR_RESULTADOS / "dataset_consolidado.parquet"
RUTA_LIMPIO_PARQUET: Path = DIR_RESULTADOS / "dataset_limpio.parquet"
RUTA_SERIE_MENSUAL: Path = DIR_RESULTADOS / "serie_mensual_total.csv"
RUTA_SERIE_CATEGORIA: Path = DIR_RESULTADOS / "serie_mensual_por_categoria.csv"
RUTA_INFORME_EDA: Path = DIR_RESULTADOS / "informe_resultados.md"
RUTA_METRICAS: Path = DIR_RESULTADOS / "comparacion_modelos.csv"
RUTA_PRONOSTICO: Path = DIR_RESULTADOS / "pronostico_final.csv"
RUTA_LOG_INGESTA: Path = DIR_RESULTADOS / "log_ingesta.txt"

# ---------------------------------------------------------------------------
# Lectura de los CSV (descubierto empíricamente en la exploración)
# ---------------------------------------------------------------------------
# Separador real de los reportes de PERÚ COMPRAS.
SEPARADOR_CSV: str = ";"
# Codificaciones a probar EN ORDEN. Se prueba utf-8 estricto primero: como
# falla en los archivos latin-1 (bytes no válidos en utf-8) pero NO en el único
# archivo utf-8, este orden detecta correctamente la codificación de cada CSV.
CODIFICACIONES: tuple[str, ...] = ("utf-8", "latin-1")

# Formatos de fecha encontrados (la mayoría usa el primero; 202406.csv usa el segundo).
FORMATOS_FECHA: tuple[str, ...] = ("%Y-%m-%d %H:%M:%S", "%d/%m/%Y %H:%M")

# Patrón del nombre de archivo: contiene AAAAMM (año y mes) en algún lugar.
PATRON_FECHA_ARCHIVO: str = r"(\d{6})"

# ---------------------------------------------------------------------------
# Columnas (nombres ya normalizados: MAYÚSCULAS, sin tildes, espacios -> "_")
# ---------------------------------------------------------------------------
COL_FECHA_FORMALIZACION: str = "FECHA_FORMALIZACION"   # eje temporal del pronóstico
COL_FECHA_PROCESO: str = "FECHA_PROCESO"
COL_TOTAL: str = "TOTAL"                               # variable a pronosticar (agregada)
COL_SUBTOTAL: str = "SUB_TOTAL"
COL_IGV: str = "IGV"
COL_ESTADO: str = "ESTADO_ORDEN_ELECTRONICA"
COL_ORDEN: str = "ORDEN_ELECTRONICA"
COL_ACUERDO_MARCO: str = "ACUERDO_MARCO"
COL_ENTIDAD: str = "ENTIDAD"
COL_PROVEEDOR: str = "PROVEEDOR"
COL_TIPO_PROC: str = "TIPO_PROCEDIMIENTO"

# Columnas que deben convertirse a número y a fecha.
COLUMNAS_MONTO: tuple[str, ...] = (COL_SUBTOTAL, COL_IGV, COL_TOTAL)
COLUMNAS_FECHA: tuple[str, ...] = (COL_FECHA_PROCESO, COL_FECHA_FORMALIZACION, "FECHA_ULTIMO_ESTADO")

# Estados que se EXCLUYEN del gasto (órdenes resueltas o anuladas: no representan
# gasto efectivo). El criterio se justifica en el informe.
ESTADOS_EXCLUIDOS: tuple[str, ...] = ("RESUELTA", "ORDEN DE COMPRA NULA")

# ---------------------------------------------------------------------------
# Parámetros del modelado / serie temporal
# ---------------------------------------------------------------------------
SEMILLA: int = 42                 # semilla global para reproducibilidad
PERIODO_ESTACIONAL: int = 12      # estacionalidad anual (datos mensuales)
MESES_HOLDOUT: int = 6            # meses reservados para prueba (los más recientes)
HORIZONTE_PRONOSTICO: int = 6     # meses a pronosticar hacia adelante
NIVEL_CONFIANZA: float = 0.95     # nivel para los intervalos de confianza

# Métrica primaria para seleccionar el mejor modelo (menor es mejor). Se elige
# WAPE (MAPE ponderado por monto) por ser robusta al trough de enero, donde el
# MAPE explota al dividir entre un denominador casi-cero. Ver evaluacion.py.
METRICA_PRIMARIA: str = "WAPE"

# Backtesting con origen móvil (rolling/expanding origin). Más robusto que un
# holdout único: promedia el error sobre varios orígenes de pronóstico.
BACKTEST_N_ORIGENES: int = 6      # nº de orígenes de evaluación
BACKTEST_PASO: int = 1            # meses entre orígenes consecutivos

# Objetivo de los modelos de árboles de gradiente. El gasto es no-negativo y de
# cola larga; "tweedie" optimiza un error de tipo relativo (proxy del MAPE/WAPE)
# mejor que el error cuadrático por defecto. Alternativa: "log" (entrenar sobre
# log1p del objetivo). Ver modelos.py.
OBJETIVO_GBM: str = "tweedie"
TWEEDIE_VARIANCE_POWER: float = 1.3

# País para el calendario de feriados (días hábiles por mes). 'PE' = Perú.
PAIS_FERIADOS: str = "PE"
N_ARMONICOS_FOURIER: int = 3      # nº de armónicos para la estacionalidad anual

# ---------------------------------------------------------------------------
# Cobertura de datos, modelo global/jerárquico, tuning e intervalos
# ---------------------------------------------------------------------------
# Año mínimo de cobertura esperado. NO filtra la ingesta (que lee todo data/):
# solo sirve para VERIFICAR de forma trazable que la serie usa 2022→presente.
ANIO_INICIO_DATOS: int = 2022

# Nº de categorías (ACUERDO_MARCO) que entran como series propias en el panel
# del modelo global; el resto se agrupa en el bucket 'OTRAS'. Se eligen las de
# mayor gasto acumulado (cubren ~95% del gasto) para no meter al modelo global
# series ultra-ralas que solo añaden ruido. El gasto total se reconcilia como la
# suma de estas top_n + 'OTRAS' (reconciliación bottom-up).
TOP_N_CATEGORIAS: int = 15

# Tuning de hiperparámetros de los GBM con Optuna. El informe documenta que el
# tuning no rompe el techo del 10% (los GBM pierden contra los estadísticos pase
# lo que pase), por eso se acota a pocos trials para no inflar el runtime.
USAR_OPTUNA: bool = True
N_TRIALS_OPTUNA: int = 30

# Intervalos de predicción conformales (cuantiles empíricos del error absoluto
# por paso de horizonte, medidos en el backtest) en vez del RMSE-normal. Más
# honestos cuando los residuos no son gaussianos. Fallback al RMSE-normal si se
# desactiva o no hay suficientes folds.
USAR_CONFORMAL: bool = True

# Umbral para la detección automática de meses incompletos en la cola:
# si el conteo de órdenes de un mes de la cola es menor a esta fracción de su
# valor esperado (mismo mes del año anterior), se considera incompleto.
UMBRAL_MES_INCOMPLETO: float = 0.5


def asegurar_directorios() -> None:
    """Crea las carpetas de salida si no existen (idempotente)."""
    for d in (DIR_OUTPUTS, DIR_FIGURAS, DIR_MODELOS, DIR_RESULTADOS):
        d.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# MÓDULO B — Detección de órdenes anómalas
# ===========================================================================
# El Módulo B es hermano e independiente del Módulo A: comparte la capa de datos
# (dataset limpio). Antes vivía en `config_b.py`; se fusionó aquí para que toda
# la configuración del ML viva en un solo lugar (`ml/shared/config.py`).
#
# IMPORTANTE — sobre los umbrales: ningún umbral *sobre los datos* (corte de
# fraccionamiento, tolerancia de IGV, cortes de monto, niveles de riesgo) se fija
# aquí: TODOS se derivan en tiempo de ejecución de la distribución real de los
# datos (ver features_anomalias.py y riesgo.py). Aquí solo hay decisiones de
# diseño (tamaños de ventana, nº de árboles, cuántas alertas), justificadas.

# Reutilización de la capa de datos del Módulo A.
RUTA_DATASET_LIMPIO: Path = RUTA_LIMPIO_PARQUET

# Salidas propias del Módulo B (no pisan las del A).
RUTA_ALERTAS_CSV: Path = DIR_RESULTADOS / "alertas.csv"
RUTA_INFORME_B: Path = DIR_RESULTADOS / "informe_modulo_b.md"
PREFIJO_FIG_B: str = "b"  # prefijo de las figuras del B (b01..b08)

# Columnas extra que usa el Módulo B (alias y nuevas).
COL_ACUERDO: str = COL_ACUERDO_MARCO
COL_FECHA_FORM: str = COL_FECHA_FORMALIZACION
COL_RUC_ENTIDAD: str = "RUC_ENTIDAD"
COL_RUC_PROVEEDOR: str = "RUC_PROVEEDOR"
COL_FECHA_ULT: str = "FECHA_ULTIMO_ESTADO"

# IGV legal en Perú (hecho legal, no supuesto sobre los datos). La tolerancia
# para marcar inconsistencia se deriva de los datos en features_anomalias.py.
IGV_LEGAL: float = 0.18

# Parámetros ESTRUCTURALES (decisiones de diseño, justificadas).
VENTANAS_FRACC_DIAS: tuple[int, ...] = (7, 30)   # ventanas cortas de fraccionamiento
BANDA_PROXIMIDAD_UMBRAL: float = 0.90            # banda "pegada" al umbral (90-100%)
DIAS_CIERRE_MES: int = 5                          # últimos N días naturales del mes

# Isolation Forest (no supervisado).
IF_N_ESTIMADORES: int = 300
IF_MAX_SAMPLES: int | str = "auto"
IF_CONTAMINACION: str | float = "auto"

# Autoencoder opcional (MLP de scikit-learn). Si falla, el flujo continúa.
USAR_AUTOENCODER: bool = True
AE_CAPAS_OCULTAS: tuple[int, ...] = (8, 3, 8)    # cuello de botella en 3 dims
AE_MAX_ITER: int = 60

# Cortes de los NIVELES de riesgo: CUANTILES del propio puntaje combinado.
CUANTIL_RIESGO_ALTO: float = 0.99
CUANTIL_RIESGO_MEDIO: float = 0.95

# Evaluación.
TOP_K_ALERTAS: tuple[int, ...] = (20, 50, 100)   # precision@k a inspeccionar
N_ANOMALIAS_SINTETICAS: int = 200                # casos artificiales por tipo
