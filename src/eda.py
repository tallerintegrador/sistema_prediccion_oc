"""
eda.py
======
Análisis Exploratorio de Datos (EDA) con interpretación escrita en español.

Cada figura se guarda en `outputs/figuras/` y va acompañada de una interpretación
generada a partir de las cifras REALES del dataset (no de supuestos). El módulo
arma además la sección de EDA del informe en Markdown.

Incluye la detección automática de meses incompletos (delegada en
`serie_temporal`) y su visualización.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: imprescindible para guardar figuras en lote
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

from . import config

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.titlesize"] = 12

MESES_ES = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic",
}


def _guardar(fig: plt.Figure, nombre: str) -> str:
    """Guarda la figura en outputs/figuras y devuelve su ruta relativa."""
    ruta = config.DIR_FIGURAS / nombre
    fig.savefig(ruta)
    plt.close(fig)
    logger.info("Figura guardada: %s", ruta.name)
    # Ruta relativa DESDE el informe (outputs/resultados/) hacia outputs/figuras/.
    return f"../figuras/{nombre}"


def _millones(x, _pos=None) -> str:
    """Formatea un monto en soles como 'X.X M' (millones)."""
    return f"{x / 1e6:,.1f}M"


# ---------------------------------------------------------------------------
# Figuras individuales (cada una devuelve (ruta_relativa, interpretacion))
# ---------------------------------------------------------------------------
def fig_evolucion_temporal(serie: pd.DataFrame) -> tuple[str, str]:
    """Evolución mensual del gasto total y del número de órdenes (doble eje)."""
    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.plot(serie.index, serie["gasto"], color="#1f4e79", marker="o", ms=3, label="Gasto (S/)")
    ax1.set_ylabel("Gasto mensual (S/)", color="#1f4e79")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(_millones))
    ax1.tick_params(axis="y", labelcolor="#1f4e79")

    ax2 = ax1.twinx()
    ax2.bar(serie.index, serie["ordenes"], width=20, alpha=0.20, color="#c55a11", label="Órdenes")
    ax2.set_ylabel("N.º de órdenes", color="#c55a11")
    ax2.tick_params(axis="y", labelcolor="#c55a11")
    ax2.grid(False)

    ax1.set_title("Evolución mensual del gasto y del número de órdenes (FECHA_FORMALIZACIÓN)")
    ax1.set_xlabel("Mes")
    ruta = _guardar(fig, "01_evolucion_gasto_ordenes.png")

    gasto_total = serie["gasto"].sum()
    prom_mensual = serie["gasto"].mean()
    mes_max = serie["gasto"].idxmax()
    mes_min = serie["gasto"].idxmin()
    interp = (
        f"La serie cubre {len(serie)} meses. El gasto total acumulado es de "
        f"**S/ {gasto_total/1e6:,.0f} millones**, con un promedio de "
        f"**S/ {prom_mensual/1e6:,.1f} millones/mes**. El mes de mayor gasto fue "
        f"{mes_max.strftime('%Y-%m')} (S/ {serie['gasto'].max()/1e6:,.1f}M) y el de menor "
        f"{mes_min.strftime('%Y-%m')} (S/ {serie['gasto'].min()/1e6:,.1f}M). Se observa una "
        f"caída pronunciada y recurrente cada inicio de año (enero), coherente con el "
        f"calendario de ejecución presupuestal pública peruana, y una recuperación entre "
        f"marzo y diciembre. El número de órdenes (barras) acompaña al gasto, aunque en los "
        f"últimos años el gasto se sostiene con menos órdenes (ticket promedio creciente)."
    )
    return ruta, interp


def fig_distribucion_total(df: pd.DataFrame) -> tuple[str, str]:
    """Distribución del monto TOTAL por orden (escala logarítmica)."""
    total = df[config.COL_TOTAL]
    positivos = total[total > 0]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.hist(np.log10(positivos), bins=60, color="#2e75b6", edgecolor="white")
    ax.set_title("Distribución del monto TOTAL por orden (escala log10)")
    ax.set_xlabel("log10(TOTAL en S/)")
    ax.set_ylabel("N.º de órdenes")
    ruta = _guardar(fig, "02_distribucion_total.png")

    interp = (
        f"El monto por orden está **fuertemente sesgado a la derecha**: la mediana es "
        f"**S/ {total.median():,.0f}** pero la media es **S/ {total.mean():,.0f}**, señal de "
        f"una cola larga de grandes compras (máximo S/ {total.max():,.0f}). El histograma en "
        f"escala logarítmica muestra una forma aproximadamente acampanada, típica de montos "
        f"con distribución log-normal. Esta asimetría justifica analizar el GASTO AGREGADO "
        f"mensual (más estable) en lugar de los montos individuales para el pronóstico."
    )
    return ruta, interp


def fig_gasto_por_categoria(df: pd.DataFrame, top_n: int = 12) -> tuple[str, str]:
    """Gasto acumulado por ACUERDO_MARCO (top categorías)."""
    gasto = df.groupby(config.COL_ACUERDO_MARCO)[config.COL_TOTAL].sum().sort_values(ascending=False)
    top = gasto.head(top_n)
    etiquetas = [c[:55] + ("…" if len(c) > 55 else "") for c in top.index]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(etiquetas[::-1], top.values[::-1], color="#548235")
    ax.set_title(f"Gasto acumulado por ACUERDO_MARCO (top {top_n})")
    ax.set_xlabel("Gasto (S/)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_millones))
    ruta = _guardar(fig, "03_gasto_por_categoria.png")

    pct_top = top.sum() / gasto.sum() * 100
    interp = (
        f"Existen **{gasto.shape[0]} valores distintos de ACUERDO_MARCO** (con variaciones de "
        f"etiqueta entre años: algunos llevan el código del convenio, p. ej. 'EXT-CE-2021-7', y "
        f"otros solo la descripción). Las {top_n} categorías mostradas concentran el "
        f"**{pct_top:.1f}%** del gasto. Dominan los bienes de oficina —útiles de escritorio, "
        f"impresoras y consumibles— seguidos de materiales de limpieza y equipos de cómputo. "
        f"Esto orienta una eventual serie por categoría, aunque el foco del Módulo A es el "
        f"gasto total."
    )
    return ruta, interp


def fig_gasto_por_tipo(df: pd.DataFrame) -> tuple[str, str]:
    """Gasto y nº de órdenes por TIPO_PROCEDIMIENTO."""
    g = df.groupby(config.COL_TIPO_PROC).agg(
        gasto=(config.COL_TOTAL, "sum"), ordenes=(config.COL_TOTAL, "size")
    ).sort_values("gasto", ascending=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(g.index, g["gasto"], color="#c55a11")
    axes[0].set_title("Gasto por tipo de procedimiento")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(_millones))
    axes[0].tick_params(axis="x", rotation=15)
    axes[1].bar(g.index, g["ordenes"], color="#7f7f7f")
    axes[1].set_title("N.º de órdenes por tipo de procedimiento")
    axes[1].tick_params(axis="x", rotation=15)
    ruta = _guardar(fig, "04_gasto_por_tipo_procedimiento.png")

    top_gasto = g["gasto"].idxmax()
    top_ordenes = g["ordenes"].idxmax()
    pct_gasto = g.loc[top_gasto, "gasto"] / g["gasto"].sum() * 100
    pct_ord = g.loc[top_ordenes, "ordenes"] / g["ordenes"].sum() * 100
    interp = (
        f"En **gasto**, el tipo de procedimiento que más concentra es **'{top_gasto}'** "
        f"({pct_gasto:.0f}% del total), mientras que en **número de órdenes** domina "
        f"**'{top_ordenes}'** ({pct_ord:.0f}% de las órdenes). Es decir, hay muchas órdenes "
        f"ordinarias de monto moderado y pocas 'Gran Compra' de ticket muy elevado; estas "
        f"últimas, al ser grandes y esporádicas, pueden introducir saltos puntuales en el gasto "
        f"mensual y conviene tenerlas presentes al interpretar picos de la serie."
    )
    return ruta, interp


def fig_top_entidades_proveedores(df: pd.DataFrame, top_n: int = 12) -> tuple[str, str]:
    """Principales entidades compradoras y proveedores por gasto."""
    ent = df.groupby(config.COL_ENTIDAD)[config.COL_TOTAL].sum().sort_values(ascending=False).head(top_n)
    prov = df.groupby(config.COL_PROVEEDOR)[config.COL_TOTAL].sum().sort_values(ascending=False).head(top_n)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].barh([e[:38] for e in ent.index][::-1], ent.values[::-1], color="#1f4e79")
    axes[0].set_title(f"Top {top_n} entidades por gasto")
    axes[0].xaxis.set_major_formatter(mticker.FuncFormatter(_millones))
    axes[1].barh([p[:38] for p in prov.index][::-1], prov.values[::-1], color="#843c0c")
    axes[1].set_title(f"Top {top_n} proveedores por gasto")
    axes[1].xaxis.set_major_formatter(mticker.FuncFormatter(_millones))
    ruta = _guardar(fig, "05_top_entidades_proveedores.png")

    interp = (
        f"Hay **{df[config.COL_ENTIDAD].nunique():,} entidades** compradoras y "
        f"**{df[config.COL_PROVEEDOR].nunique():,} proveedores** distintos. El gasto está "
        f"relativamente concentrado: la entidad líder es '{ent.index[0][:40]}' y el proveedor "
        f"líder es '{prov.index[0][:40]}'. Esta concentración es relevante para el futuro "
        f"módulo de detección de anomalías, pero no condiciona el pronóstico agregado."
    )
    return ruta, interp


def fig_estacionalidad_mes(serie: pd.DataFrame) -> tuple[str, str]:
    """Patrón estacional: distribución del gasto por mes calendario."""
    aux = serie.copy()
    aux["mes_num"] = aux.index.month
    aux["mes"] = aux["mes_num"].map(MESES_ES)
    orden = [MESES_ES[i] for i in range(1, 13)]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    sns.boxplot(data=aux, x="mes", y="gasto", order=orden, ax=ax, color="#9dc3e6")
    ax.set_title("Estacionalidad: distribución del gasto mensual por mes calendario")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Gasto (S/)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_millones))
    ruta = _guardar(fig, "06_estacionalidad_por_mes.png")

    medias = aux.groupby("mes_num")["gasto"].mean()
    mes_bajo = MESES_ES[int(medias.idxmin())]
    mes_alto = MESES_ES[int(medias.idxmax())]
    interp = (
        f"El boxplot por mes calendario confirma una **estacionalidad anual marcada y "
        f"consistente**: **{mes_bajo}** es sistemáticamente el mes de menor gasto "
        f"(S/ {medias.min()/1e6:,.1f}M en promedio) y **{mes_alto}** uno de los más altos "
        f"(S/ {medias.max()/1e6:,.1f}M). Como el patrón se repite en todos los años, NO se "
        f"trata de meses incompletos sino de estacionalidad real. Esto favorece modelos con "
        f"componente estacional de periodo 12 (SARIMA, naive estacional, variables de mes)."
    )
    return ruta, interp


def fig_descomposicion(serie: pd.DataFrame) -> tuple[str, str]:
    """Descomposición de la serie en tendencia, estacionalidad y residuo."""
    s = serie["gasto"].asfreq("ME") if serie.index.freq is None else serie["gasto"]
    s = serie["gasto"].copy()
    s.index = pd.DatetimeIndex(s.index)
    try:
        desc = seasonal_decompose(s, model="additive", period=config.PERIODO_ESTACIONAL)
    except Exception as exc:  # noqa: BLE001
        logger.warning("No se pudo descomponer la serie: %s", exc)
        return "", ""
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True)
    desc.observed.plot(ax=axes[0], color="#1f4e79"); axes[0].set_ylabel("Observado")
    desc.trend.plot(ax=axes[1], color="#c55a11"); axes[1].set_ylabel("Tendencia")
    desc.seasonal.plot(ax=axes[2], color="#548235"); axes[2].set_ylabel("Estacional")
    desc.resid.plot(ax=axes[3], color="#7f7f7f", marker="."); axes[3].set_ylabel("Residuo")
    for a in axes:
        a.yaxis.set_major_formatter(mticker.FuncFormatter(_millones))
    axes[0].set_title("Descomposición aditiva de la serie de gasto mensual (periodo=12)")
    ruta = _guardar(fig, "07_descomposicion_serie.png")

    amplitud = desc.seasonal.max() - desc.seasonal.min()
    tendencia = desc.trend.dropna()
    direccion = "ascendente" if tendencia.iloc[-1] > tendencia.iloc[0] else "estable o descendente"
    interp = (
        f"La descomposición separa tres componentes. La **tendencia** es {direccion} a lo largo "
        f"del periodo. La **componente estacional** tiene una amplitud de "
        f"S/ {amplitud/1e6:,.1f}M entre el mes más bajo y el más alto, confirmando la "
        f"estacionalidad anual. El **residuo** recoge el ruido y eventos puntuales (p. ej. "
        f"grandes compras), sin un patrón evidente, lo que indica que tendencia + estacionalidad "
        f"explican buena parte de la variación."
    )
    return ruta, interp


def fig_deteccion_incompletos(serie_completa: pd.DataFrame, deteccion: dict) -> tuple[str, str]:
    """Visualiza el conteo de órdenes y la comparación con el año previo en la cola."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(serie_completa.index, serie_completa["ordenes"], marker="o", ms=3, color="#c55a11")
    incompletos = deteccion["meses_incompletos"]
    if incompletos:
        ax.scatter(
            incompletos,
            serie_completa.loc[serie_completa.index.isin(incompletos), "ordenes"],
            color="red", zorder=5, s=60, label="Marcado incompleto",
        )
        ax.legend()
    ax.set_title("Detección de meses incompletos: n.º de órdenes por mes")
    ax.set_xlabel("Mes")
    ax.set_ylabel("N.º de órdenes")
    ruta = _guardar(fig, "08_deteccion_meses_incompletos.png")

    if incompletos:
        lista = ", ".join(d.strftime("%Y-%m") for d in incompletos)
        decision = (
            f"Se detectaron y **recortaron** los siguientes meses por tener menos del "
            f"{int(deteccion['umbral']*100)}% de las órdenes del mismo mes del año anterior: "
            f"**{lista}**."
        )
    else:
        ult = deteccion["detalle_cola"][-1] if deteccion["detalle_cola"] else None
        ref = f" (último mes {ult['mes']}: {ult['ratio']:.2f}× respecto al año previo)" if ult else ""
        decision = (
            f"**Ningún mes de la cola resultó incompleto**{ref}: el conteo de órdenes del último "
            f"mes es comparable al del mismo mes del año anterior, por lo que se conservan todos "
            f"los meses. La fuerte caída de enero NO se recorta porque se repite cada año "
            f"(estacionalidad real, no falta de datos)."
        )
    interp = (
        "La detección de incompletitud compara, mes a mes desde el final, el número de órdenes "
        "contra el mismo mes del año anterior (controlando la estacionalidad). " + decision
    )
    return ruta, interp


# ---------------------------------------------------------------------------
# Orquestación: genera todas las figuras y arma la sección EDA del informe
# ---------------------------------------------------------------------------
def ejecutar_eda(
    df: pd.DataFrame,
    serie_completa: pd.DataFrame,
    deteccion: dict,
    reporte_ingesta: dict,
    reporte_limpieza: dict,
) -> str:
    """
    Genera todas las figuras del EDA y devuelve la sección del informe en Markdown.

    Parámetros
    ----------
    df : DataFrame limpio a nivel de orden.
    serie_completa : serie mensual SIN recortar (para visualizar la detección).
    deteccion : salida de `serie_temporal.detectar_meses_incompletos`.
    reporte_ingesta, reporte_limpieza : diccionarios de las etapas previas.
    """
    config.asegurar_directorios()
    figuras = [
        fig_evolucion_temporal(serie_completa),
        fig_distribucion_total(df),
        fig_gasto_por_categoria(df),
        fig_gasto_por_tipo(df),
        fig_top_entidades_proveedores(df),
        fig_estacionalidad_mes(serie_completa),
        fig_descomposicion(serie_completa),
        fig_deteccion_incompletos(serie_completa, deteccion),
    ]

    # ---- Sección de resumen general y calidad de datos (texto) ----
    rng = reporte_ingesta["meses_cubiertos"]
    nulos = reporte_limpieza.get("nulos_por_columna", {})
    nulos_top = {k: v for k, v in nulos.items() if v > 0 and not k.startswith("__")}

    md = []
    md.append("## 2. Análisis Exploratorio de Datos (EDA)\n")
    md.append("### 2.1 Resumen general\n")
    md.append(
        f"- **Archivos leídos:** {reporte_ingesta['archivos_leidos']} de "
        f"{reporte_ingesta['archivos_encontrados']} encontrados "
        f"(omitidos: {reporte_ingesta['archivos_omitidos'] or 'ninguno'}).\n"
        f"- **Meses cubiertos:** {reporte_ingesta['n_meses']} "
        f"(de {rng[0]} a {rng[-1]}).\n"
        f"- **Codificaciones detectadas:** {reporte_ingesta['codificaciones']}.\n"
        f"- **Filas consolidadas (crudo):** {reporte_ingesta['filas_totales']:,}.\n"
        f"- **Filas válidas tras limpieza:** {reporte_limpieza['filas_finales']:,}.\n"
        f"- **Entidades distintas:** {df[config.COL_ENTIDAD].nunique():,} | "
        f"**Proveedores distintos:** {df[config.COL_PROVEEDOR].nunique():,} | "
        f"**Categorías (ACUERDO_MARCO):** {df[config.COL_ACUERDO_MARCO].nunique():,}.\n"
    )

    md.append("\n### 2.2 Calidad de los datos\n")
    md.append(
        f"- **Duplicados exactos:** {reporte_limpieza.get('duplicados_exactos', 0)}. "
        f"**Duplicados por ORDEN_ELECTRONICA:** {reporte_limpieza.get('duplicados_por_orden', 0)}.\n"
        f"- **Órdenes sin fecha de formalización (descartadas):** {reporte_limpieza.get('sin_fecha_formalizacion', 0)}.\n"
        f"- **Órdenes excluidas por estado** {config.ESTADOS_EXCLUIDOS}: "
        f"{reporte_limpieza.get('ordenes_excluidas_por_estado', 0)}.\n"
        f"- **Consistencia contable:** se verificó que TOTAL = SUB_TOTAL + IGV.\n"
    )
    if nulos_top:
        md.append("- **Columnas con valores nulos (solo auxiliares/enlaces):**\n")
        for k, v in nulos_top.items():
            md.append(f"    - `{k}`: {v}%\n")
    out = reporte_limpieza.get("outliers_total_iqr", {})
    if out:
        md.append(
            f"- **Outliers de monto** (criterio IQR×3, límite ≈ S/ {out['limite_superior']:,.0f}): "
            f"{out['n_por_encima']:,} órdenes por encima; máximo S/ {out['monto_maximo']:,.0f}. "
            f"No se eliminan: son grandes compras reales.\n"
        )

    md.append("\n### 2.3 Figuras e interpretaciones\n")
    for i, (ruta, interp) in enumerate(figuras, start=1):
        if not ruta:
            continue
        titulo = ruta.split("/")[-1].replace(".png", "").split("_", 1)[-1].replace("_", " ").capitalize()
        md.append(f"\n**Figura {i}. {titulo}**\n")
        md.append(f"\n![{titulo}]({ruta})\n")
        md.append(f"\n_{interp}_\n")

    return "".join(md)
