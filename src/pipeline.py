"""
pipeline.py
===========
Orquestador del **Módulo A**. Define las FASES en ORDEN (de la extracción al
entrenamiento y pronóstico) y las ejecuta de punta a punta, mostrando un
**resumen por fase** en consola y dejando todos los artefactos en `outputs/`.

Fases (cada una vive en `src/fases/fNN_*.py`):

    1. Ingesta / ETL            -> extracción y consolidación de los CSV
    2. Limpieza y validación    -> tipado, fechas/montos, duplicados, filtros
    3. Serie temporal           -> gasto mensual + panel por categoría + drivers
    4. Análisis exploratorio    -> figuras 01–08 con interpretación
    5. Features de calendario   -> días hábiles, mes, Fourier (insumo de la fase 6)
    6. Modelado                 -> catálogo de modelos a entrenar y comparar
    7. Evaluación y pronóstico  -> backtest, selección y pronóstico final

La pieza pública es `ejecutar_pipeline()`. `main.py` solo la invoca.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

from . import config, figuras
from .fases import f01_ingesta as ingesta
from .fases import f02_limpieza as limpieza
from .fases import f03_serie_temporal as serie_temporal
from .fases import f04_eda as eda
from .fases import f05_features as features
from .fases import f06_modelado as modelado
from .fases import f07_evaluacion as evaluacion

log = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Estado compartido entre fases (se va llenando a medida que el flujo avanza)
# ---------------------------------------------------------------------------
@dataclass
class Contexto:
    """Artefactos que las fases producen y consumen, en orden de aparición."""

    df_crudo: Any = None
    rep_ingesta: dict = field(default_factory=dict)
    df_limpio: Any = None
    rep_limpieza: dict = field(default_factory=dict)
    serie_completa: Any = None          # serie mensual SIN recortar (para el EDA)
    deteccion: dict = field(default_factory=dict)
    serie_total: Any = None             # serie mensual recortada (objetivo)
    serie_categoria: Any = None
    panel: Any = None                   # panel largo por categoría (modelo global)
    drivers: Any = None                 # drivers internos mensuales
    tabla_calendario: Any = None        # features de calendario (fase 5)
    md_eda: str = ""
    md_modelos: str = ""
    resultados: dict = field(default_factory=dict)


@dataclass
class Fase:
    """Una fase del pipeline: número, nombre, descripción y función ejecutora."""

    numero: int
    nombre: str
    descripcion: str
    ejecutar: Callable[[Contexto], list[str]]


# ---------------------------------------------------------------------------
# Implementación de cada fase: corre la lógica de su módulo y devuelve el
# resumen (lista de líneas) que se imprime en consola.
# ---------------------------------------------------------------------------
def _fase_ingesta(ctx: Contexto) -> list[str]:
    ctx.df_crudo, ctx.rep_ingesta = ingesta.consolidar(guardar=True)
    r = ctx.rep_ingesta
    cob = r.get("cobertura", {})
    return [
        f"Archivos leídos     : {r['archivos_leidos']}/{r['archivos_encontrados']}"
        f" (omitidos: {len(r['archivos_omitidos'])})",
        f"Filas consolidadas  : {r['filas_totales']:,}",
        f"Meses cubiertos     : {r['n_meses']} ({r['meses_cubiertos'][0]} → {r['meses_cubiertos'][-1]})",
        f"Cobertura 2022→hoy  : {'OK' if cob.get('cobertura_ok') else 'REVISAR'}",
        f"Codificaciones      : {r['codificaciones']}",
        f"Artefacto           : {config.RUTA_CONSOLIDADO_PARQUET.name}",
    ]


def _fase_limpieza(ctx: Contexto) -> list[str]:
    ctx.df_limpio, ctx.rep_limpieza = limpieza.limpiar(ctx.df_crudo, guardar=True)
    r = ctx.rep_limpieza
    return [
        f"Filas iniciales     : {r['filas_iniciales']:,}",
        f"Excluidas por estado: {r.get('ordenes_excluidas_por_estado', 0):,} {config.ESTADOS_EXCLUIDOS}",
        f"Sin fecha/sin total : {r.get('sin_fecha_formalizacion', 0):,} / {r.get('sin_total', 0):,}",
        f"Duplicados por orden: {r.get('duplicados_por_orden', 0):,}",
        f"Filas válidas finales: {r['filas_finales']:,}",
        f"Artefacto           : {config.RUTA_LIMPIO_PARQUET.name}",
    ]


def _fase_serie_temporal(ctx: Contexto) -> list[str]:
    df = ctx.df_limpio
    ctx.serie_completa = serie_temporal.construir_serie_total(df)
    ctx.deteccion = serie_temporal.detectar_meses_incompletos(ctx.serie_completa)
    ctx.serie_total, ctx.serie_categoria, _ = serie_temporal.construir_series(df, guardar=True)

    # Panel por categoría y drivers internos: insumos del modelo global. Se
    # recortan a la MISMA cola que la serie total (meses incompletos fuera) para
    # que el pronóstico global arranque en el mismo mes que la serie agregada.
    ultimo_mes = ctx.serie_total.index[-1]
    panel = serie_temporal.construir_panel_categorias(df)
    ctx.panel = panel[panel["fecha"] <= ultimo_mes].copy()
    drivers = serie_temporal.construir_drivers_mensuales(df)
    ctx.drivers = drivers[drivers.index <= ultimo_mes].copy()

    incompletos = ctx.deteccion.get("meses_incompletos") or []
    incompletos_txt = ", ".join(d.strftime("%Y-%m") for d in incompletos) or "ninguno"
    return [
        f"Serie mensual       : {len(ctx.serie_total)} meses (objetivo: gasto total)",
        f"Meses incompletos   : {incompletos_txt} (recortados de la cola)",
        f"Panel de categorías : {ctx.panel['categoria'].nunique()} series (top-N + OTRAS)",
        f"Drivers internos    : {ctx.drivers.shape[1]} señales mensuales",
        f"Artefactos          : {config.RUTA_SERIE_MENSUAL.name}, {config.RUTA_SERIE_CATEGORIA.name}",
    ]


def _fase_eda(ctx: Contexto) -> list[str]:
    ctx.md_eda = eda.ejecutar_eda(
        ctx.df_limpio, ctx.serie_completa, ctx.deteccion, ctx.rep_ingesta, ctx.rep_limpieza
    )
    # Las figuras del EDA ocupan las primeras posiciones del catálogo (hasta la 1.ª
    # figura de evaluación). Se reporta cuántas se generaron y dónde.
    n_eda = figuras.numero("holdout_real_vs_modelos") - 1
    return [
        f"Figuras EDA         : {n_eda} (01–{n_eda:02d}) en {config.DIR_FIGURAS}",
        f"Entidades / proveed.: {ctx.df_limpio[config.COL_ENTIDAD].nunique():,}"
        f" / {ctx.df_limpio[config.COL_PROVEEDOR].nunique():,}",
        f"Categorías          : {ctx.df_limpio[config.COL_ACUERDO_MARCO].nunique():,}",
        "Interpretaciones añadidas a la sección 2 del informe.",
    ]


def _fase_features(ctx: Contexto) -> list[str]:
    # Materializa la tabla de features de calendario para la serie mensual, para
    # hacer VISIBLE este insumo (lo consumen los modelos basados en árboles).
    ctx.tabla_calendario = features.tabla_calendario(ctx.serie_total.index, "mes")
    cols = list(ctx.tabla_calendario.columns)
    return [
        f"Features de calendario: {len(cols)} columnas × {len(ctx.tabla_calendario)} meses",
        f"Columnas            : {', '.join(cols[:7])}…",
        f"Armónicos de Fourier: {config.N_ARMONICOS_FOURIER} | feriados: {config.PAIS_FERIADOS}",
        "Insumo de los modelos de árboles de la fase 6 (entrena enero, no lo promedia).",
    ]


def _fase_modelado(ctx: Contexto) -> list[str]:
    # La fase 6 define el CATÁLOGO de modelos; su entrenamiento real ocurre dentro
    # del backtest de la fase 7 (un fit por fold) y en el reentrenamiento final.
    base = list(modelado.MODELOS.keys())
    avanzados = []
    if config.USAR_OPTUNA:
        avanzados.append("LightGBM-Optuna")
    if ctx.drivers is not None:
        avanzados.append("LightGBM+drivers")
    if ctx.panel is not None:
        avanzados += ["Global jerárquico", "Global + drivers"]
    return [
        f"Modelos a comparar  : {len(base) + len(avanzados)}",
        f"Base                : {', '.join(base)}",
        f"Avanzados           : {', '.join(avanzados) or '—'}",
        f"Objetivo GBM        : {config.OBJETIVO_GBM} | semilla: {config.SEMILLA}",
        "El entrenamiento se realiza en el backtest (fase 7) y al reentrenar el final.",
    ]


def _fase_evaluacion(ctx: Contexto) -> list[str]:
    ctx.md_modelos, ctx.resultados = evaluacion.ejecutar_evaluacion(
        ctx.serie_total, ctx.df_limpio, panel=ctx.panel, drivers=ctx.drivers
    )
    r = ctx.resultados
    pron = r["pronostico"]
    fila = r["tabla_metricas"].loc[r["mejor_modelo"]]
    return [
        f"Backtest            : origen móvil ({config.BACKTEST_N_ORIGENES} orígenes, h={config.MESES_HOLDOUT})",
        f"Mejor modelo        : {r['mejor_modelo']}"
        f"  (WAPE {fila.get('WAPE'):.1f}% · MASE {fila.get('MASE'):.2f})",
        f"Intervalos          : {r.get('metodo_intervalo')}",
        f"Pronóstico {config.HORIZONTE_PRONOSTICO} meses : S/ {pron['pred'].sum() / 1e6:,.0f} millones"
        f" ({pron.index[0]:%Y-%m} → {pron.index[-1]:%Y-%m})",
        f"Figuras             : 09 holdout · 10 error/mes · 11 pronóstico",
        f"Artefactos          : {config.RUTA_METRICAS.name}, {config.RUTA_PRONOSTICO.name}",
    ]


# Las FASES, en el ORDEN del flujo. Esta lista ES la definición de la arquitectura.
FASES: list[Fase] = [
    Fase(1, "Ingesta / ETL", "Extracción: lectura y consolidación de los CSV mensuales.", _fase_ingesta),
    Fase(2, "Limpieza y validación", "Tipado de fechas/montos, duplicados y filtros de estado.", _fase_limpieza),
    Fase(3, "Serie temporal", "Gasto mensual + panel por categoría + drivers internos.", _fase_serie_temporal),
    Fase(4, "Análisis exploratorio (EDA)", "Figuras e interpretaciones del comportamiento del gasto.", _fase_eda),
    Fase(5, "Features de calendario", "Días hábiles, indicadores de mes y Fourier (insumo del modelado).", _fase_features),
    Fase(6, "Modelado", "Catálogo de modelos de pronóstico a entrenar y comparar.", _fase_modelado),
    Fase(7, "Evaluación y pronóstico", "Backtest de origen móvil, selección y pronóstico final.", _fase_evaluacion),
]


# ---------------------------------------------------------------------------
# Informe Markdown
# ---------------------------------------------------------------------------
def construir_encabezado_informe(reporte_ingesta: dict) -> str:
    """Genera la cabecera e introducción del informe en Markdown (sección 1)."""
    rng = reporte_ingesta["meses_cubiertos"]
    return (
        "# Informe de Resultados — Módulo A: Pronóstico del Gasto en Órdenes de Compra\n\n"
        f"_Sistema de Predicción de Órdenes de Compra (SistemaPrediccionOC). "
        f"Generado el {datetime.now():%Y-%m-%d %H:%M}._\n\n"
        "## 1. Introducción\n\n"
        "Este informe documenta el **Módulo A** del sistema: el pronóstico del gasto público "
        "mensual en las **Órdenes de Compra de los Catálogos Electrónicos de Acuerdos Marco** de "
        "la Central de Compras Públicas del Perú (PERÚ COMPRAS). Cada registro es una orden de "
        "compra de bienes emitida por una entidad pública a un proveedor, con su monto y convenio.\n\n"
        "El flujo está organizado en **7 fases ordenadas** (ingesta → limpieza → serie temporal → "
        "EDA → features → modelado → evaluación). La variable a pronosticar es el **gasto total "
        "mensual** (suma de `TOTAL`), agregado por **`FECHA_FORMALIZACION`**.\n\n"
        f"Los datos abarcan **{reporte_ingesta['n_meses']} meses** (de {rng[0]} a {rng[-1]}) e "
        f"incluyen **{reporte_ingesta['filas_totales']:,} órdenes** consolidadas desde "
        f"{reporte_ingesta['archivos_leidos']} archivos CSV mensuales.\n"
    )


# ---------------------------------------------------------------------------
# Salida en consola
# ---------------------------------------------------------------------------
_ANCHO = 70


def _configurar_logging() -> None:
    """Configura el logging a consola con formato legible (idempotente)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def _banner(texto: str) -> None:
    print("\n" + "=" * _ANCHO)
    print(texto.center(_ANCHO))
    print("=" * _ANCHO)


def _imprimir_fase(fase: Fase, total: int, resumen: list[str], segundos: float) -> None:
    print(f"\n[FASE {fase.numero}/{total}] {fase.nombre}  ({segundos:.1f} s)")
    print(f"   {fase.descripcion}")
    for linea in resumen:
        print(f"     - {linea}")


def _imprimir_resumen_final(ctx: Contexto) -> None:
    r = ctx.resultados
    cob = ctx.rep_ingesta.get("cobertura", {})
    _banner("MÓDULO A COMPLETADO")
    print(
        f"Cobertura de datos : {cob.get('primer_mes')} → {cob.get('ultimo_mes')} "
        f"({'OK' if cob.get('cobertura_ok') else 'REVISAR'})"
    )
    print(f"Mejor modelo       : {r['mejor_modelo']}")
    print(f"Intervalos         : {r.get('metodo_intervalo')}")
    print("\nMétricas (backtesting de origen móvil):")
    print(r["tabla_metricas"].round(2).to_string())
    print(f"\nArtefactos en      : {config.DIR_OUTPUTS}")
    print(f"Informe            : {config.RUTA_INFORME_EDA}")
    print("=" * _ANCHO)


# ---------------------------------------------------------------------------
# Punto de entrada
# ---------------------------------------------------------------------------
def ejecutar_pipeline() -> Contexto:
    """
    Ejecuta las 7 fases en orden, escribe el informe y muestra un resumen por
    fase. Devuelve el `Contexto` con todos los artefactos (útil en notebooks).
    """
    _configurar_logging()
    config.asegurar_directorios()
    figuras.aplicar_estilo()
    figuras.limpiar()           # regenerar figuras sin huérfanos de numeraciones previas
    modelado.fijar_semillas()

    total = len(FASES)
    _banner(f"MÓDULO A · Pipeline de pronóstico del gasto ({total} fases)")

    ctx = Contexto()
    for fase in FASES:
        log.info("== FASE %d/%d — %s ==", fase.numero, total, fase.nombre)
        inicio = time.perf_counter()
        resumen = fase.ejecutar(ctx)
        _imprimir_fase(fase, total, resumen, time.perf_counter() - inicio)

    # Informe final = encabezado (sec. 1) + EDA (sec. 2) + modelado (secs. 3–4).
    informe = (
        construir_encabezado_informe(ctx.rep_ingesta) + "\n" + ctx.md_eda + "\n" + ctx.md_modelos
    )
    config.RUTA_INFORME_EDA.write_text(informe, encoding="utf-8")
    log.info("Informe escrito en %s", config.RUTA_INFORME_EDA)

    _imprimir_resumen_final(ctx)
    return ctx


if __name__ == "__main__":
    ejecutar_pipeline()
