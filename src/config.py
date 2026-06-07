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
# config.py vive en src/, por lo que la raíz del proyecto es el padre de src/.
RAIZ_PROYECTO: Path = Path(__file__).resolve().parent.parent

DIR_DATOS: Path = RAIZ_PROYECTO / "data"                # CSV originales (no se modifican)
DIR_OUTPUTS: Path = RAIZ_PROYECTO / "outputs"
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

# Umbral para la detección automática de meses incompletos en la cola:
# si el conteo de órdenes de un mes de la cola es menor a esta fracción de su
# valor esperado (mismo mes del año anterior), se considera incompleto.
UMBRAL_MES_INCOMPLETO: float = 0.5


def asegurar_directorios() -> None:
    """Crea las carpetas de salida si no existen (idempotente)."""
    for d in (DIR_OUTPUTS, DIR_FIGURAS, DIR_MODELOS, DIR_RESULTADOS):
        d.mkdir(parents=True, exist_ok=True)
