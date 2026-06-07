"""
figuras.py
==========
Registro **central** de figuras del Módulo A.

Antes, el estilo de las gráficas y los números de archivo ("01_", … "11_")
estaban repetidos y dispersos entre `f04_eda.py` y `f07_evaluacion.py`. Este
módulo unifica todo eso para que las figuras queden **relacionadas y en orden**:

* **Estilo único**: tema de seaborn, DPI, formato y backend "Agg" (sin ventana),
  aplicados una sola vez.
* **Orden y numeración automáticos**: el `CATALOGO` lista las figuras en el orden
  del flujo (de la extracción al pronóstico). El número de archivo sale de la
  POSICIÓN en el catálogo, así que no hay números mágicos en el resto del código
  y las gráficas siempre quedan ordenadas y consistentes con su fase.
* **Guardado uniforme** y cálculo de la ruta relativa que consume el informe.

Para añadir una figura nueva basta con insertar su `slug` en `CATALOGO`, en la
posición de orden deseada, y llamar a `guardar(fig, slug)`.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: imprescindible para guardar en lote
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from . import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Orden canónico de las figuras (de la extracción al pronóstico).
# El índice (1..N) define el prefijo numérico del archivo => orden garantizado.
# ---------------------------------------------------------------------------
CATALOGO: tuple[str, ...] = (
    # --- Fase 4: Análisis exploratorio (EDA) ---
    "evolucion_gasto_ordenes",        # 01
    "distribucion_total",             # 02
    "gasto_por_categoria",            # 03
    "gasto_por_tipo_procedimiento",   # 04
    "top_entidades_proveedores",      # 05
    "estacionalidad_por_mes",         # 06
    "descomposicion_serie",           # 07
    "deteccion_meses_incompletos",    # 08
    # --- Fase 7: Evaluación, diagnóstico y pronóstico ---
    "holdout_real_vs_modelos",        # 09  comparación real vs. modelos
    "backtest_error_por_mes",         # 10  diagnóstico del error (heatmap)
    "pronostico_final",               # 11  resultado final (va al final, a propósito)
)

_ESTILO_APLICADO = False


def aplicar_estilo() -> None:
    """Aplica (una sola vez) el estilo común a todas las figuras del proyecto."""
    global _ESTILO_APLICADO
    if _ESTILO_APLICADO:
        return
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.titlesize"] = 12
    _ESTILO_APLICADO = True


def millones(x, _pos=None) -> str:
    """Formatea un monto en soles como 'X.X M' (millones)."""
    return f"{x / 1e6:,.1f}M"


def formato_millones() -> mticker.FuncFormatter:
    """Formateador de ejes en millones de soles (para `set_major_formatter`)."""
    return mticker.FuncFormatter(millones)


def numero(slug: str) -> int:
    """Número de orden (1..N) de la figura `slug` según el `CATALOGO`."""
    try:
        return CATALOGO.index(slug) + 1
    except ValueError as exc:  # noqa: TRY003
        raise KeyError(
            f"Figura '{slug}' no registrada en figuras.CATALOGO. "
            f"Añádela en la posición de orden deseada."
        ) from exc


def nombre_archivo(slug: str) -> str:
    """Nombre de archivo numerado, p. ej. 'pronostico_final' -> '11_pronostico_final.png'."""
    return f"{numero(slug):02d}_{slug}.png"


def ruta_relativa(slug: str) -> str:
    """Ruta de la figura DESDE el informe (outputs/resultados/) hacia outputs/figuras/."""
    return f"../figuras/{nombre_archivo(slug)}"


def guardar(fig: "plt.Figure", slug: str) -> str:
    """
    Guarda `fig` en outputs/figuras con su nombre numerado y la cierra.
    Devuelve la ruta relativa lista para insertar en el informe Markdown.
    """
    config.asegurar_directorios()
    nombre = nombre_archivo(slug)
    fig.savefig(config.DIR_FIGURAS / nombre)
    plt.close(fig)
    logger.info("Figura guardada: %s", nombre)
    return ruta_relativa(slug)


def limpiar() -> None:
    """
    Borra los PNG previos de outputs/figuras antes de regenerar, para que un
    cambio de numeración/orden no deje figuras huérfanas con nombres antiguos.
    """
    if config.DIR_FIGURAS.exists():
        for p in config.DIR_FIGURAS.glob("*.png"):
            p.unlink()
        logger.info("Figuras previas limpiadas en %s", config.DIR_FIGURAS)
