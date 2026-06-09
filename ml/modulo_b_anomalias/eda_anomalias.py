"""
eda_anomalias.py
================
Análisis Exploratorio de Datos (EDA) **orientado a anomalías**, con interpretación
escrita en español. Antes de modelar, se explora cómo se ve lo "normal" en cada
variable, para que las anomalías tengan sentido.

Cada figura se guarda en `outputs/figuras/` (con prefijo `b` para no colisionar
con las del Módulo A) y va acompañada de una interpretación generada a partir de
las cifras REALES (no de supuestos). El módulo arma la sección de EDA del informe
del Módulo B en Markdown.

Figuras:
    b01  Distribución de montos (TOTAL en log) y del ratio de IGV.
    b02  Concentración de proveedor por entidad (share del top y HHI).
    b03  Patrones temporales (órdenes por día y proximidad al cierre de mes).
    b04  Vista general de la red entidad–proveedor (grados + ego-grafo).
    b05  Punto de corte del fraccionamiento: P(Gran Compra|TOTAL) y bunching.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: para guardar figuras en lote
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns

from ml.shared import config as cb

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.titlesize"] = 12


def _guardar(fig: plt.Figure, nombre: str) -> str:
    """Guarda la figura en outputs/figuras y devuelve su ruta relativa al informe."""
    ruta = cb.DIR_FIGURAS / nombre
    fig.savefig(ruta)
    plt.close(fig)
    logger.info("Figura guardada: %s", ruta.name)
    return f"../figuras/{nombre}"


# ---------------------------------------------------------------------------
# b01 — Montos y ratio de IGV
# ---------------------------------------------------------------------------
def fig_montos_igv(feats: pd.DataFrame, meta: dict) -> tuple[str, str]:
    """Distribución del log-monto y del ratio IGV/SUB_TOTAL."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    ax1.hist(feats["LOG_TOTAL"], bins=80, color="#1f4e79", alpha=0.85)
    ax1.set_title("Distribución del monto (log1p(TOTAL))")
    ax1.set_xlabel("log(1 + TOTAL)")
    ax1.set_ylabel("N.º de órdenes")

    # El ratio se concentra en 0.18; recortamos a [0, 0.25] para ver la masa.
    ratio = feats["RATIO_IGV"].clip(0, 0.25)
    ax2.hist(ratio, bins=60, color="#c55a11", alpha=0.85)
    ax2.axvline(cb.IGV_LEGAL, color="black", ls="--", lw=1.2, label="IGV legal 18%")
    ax2.set_title("Distribución del ratio IGV / SUB_TOTAL")
    ax2.set_xlabel("IGV / SUB_TOTAL")
    ax2.set_ylabel("N.º de órdenes")
    ax2.legend()
    ruta = _guardar(fig, f"{cb.PREFIJO_FIG_B}01_montos_igv.png")

    total = feats["TOTAL"]
    tol = meta["tolerancia_igv"]["tolerancia"]
    n_igv = meta["tolerancia_igv"]["n_inconsistentes"]
    n_igv0 = int((feats["RATIO_IGV"] == 0).sum())
    interp = (
        f"El monto `TOTAL` tiene una **cola extremadamente larga**: la mediana es "
        f"**S/ {total.median():,.0f}** pero el máximo llega a **S/ {total.max():,.0f}**. "
        f"Por eso el modelado usa el **logaritmo** del monto (panel izquierdo), donde la "
        f"distribución se vuelve aproximadamente acampanada y las verdaderas órdenes "
        f"gigantes quedan en el extremo derecho.\n\n"
        f"El ratio `IGV / SUB_TOTAL` (panel derecho) está **casi siempre exactamente en "
        f"0.18** (el IGV peruano del 18 %): el {100*(1-n_igv/len(feats)):.1f} % de las órdenes "
        f"cae sobre la línea. La distribución es **bimodal**: un pico nítido en 0.18 y otro "
        f"en 0 (órdenes sin IGV). Con la tolerancia derivada de los datos "
        f"(±{tol:.3f}) quedan marcadas **{n_igv:,} órdenes inconsistentes** "
        f"({100*n_igv/len(feats):.2f} %), de las cuales {n_igv0:,} tienen IGV = 0. "
        f"Lo *normal* es 0.18; apartarse de ahí es la señal de anomalía de IGV."
    )
    return ruta, interp


# ---------------------------------------------------------------------------
# b02 — Concentración de proveedor por entidad
# ---------------------------------------------------------------------------
def fig_concentracion(feats: pd.DataFrame) -> tuple[str, str]:
    """Share del proveedor principal y HHI de proveedores, por entidad."""
    # Una fila por entidad: share máximo de un proveedor y HHI.
    por_ent = feats.groupby(cb.COL_RUC_ENTIDAD).agg(
        share_top=("SHARE_GASTO_PROV", "max"),
        hhi=("HHI_ENTIDAD", "first"),
    )
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    ax1.hist(por_ent["share_top"], bins=40, color="#2e7d32", alpha=0.85)
    ax1.axvline(por_ent["share_top"].median(), color="black", ls="--", lw=1.2,
                label=f"mediana {por_ent['share_top'].median():.2f}")
    ax1.set_title("Concentración: share del proveedor principal por entidad")
    ax1.set_xlabel("Proporción del gasto en su mayor proveedor")
    ax1.set_ylabel("N.º de entidades")
    ax1.legend()

    ax2.hist(por_ent["hhi"], bins=40, color="#6a1b9a", alpha=0.85)
    ax2.axvline(por_ent["hhi"].median(), color="black", ls="--", lw=1.2,
                label=f"mediana {por_ent['hhi'].median():.2f}")
    ax2.set_title("Índice HHI de proveedores por entidad")
    ax2.set_xlabel("HHI (1 = un solo proveedor)")
    ax2.set_ylabel("N.º de entidades")
    ax2.legend()
    ruta = _guardar(fig, f"{cb.PREFIJO_FIG_B}02_concentracion.png")

    p90 = por_ent["share_top"].quantile(0.90)
    p95 = por_ent["share_top"].quantile(0.95)
    n_casi_total = int((por_ent["share_top"] >= 0.90).sum())
    interp = (
        f"Cada entidad reparte su gasto entre varios proveedores. Lo *habitual* es una "
        f"concentración moderada: la mediana del share del **proveedor principal** es "
        f"**{por_ent['share_top'].median():.2f}**. Pero la cola derecha es preocupante: el "
        f"10 % de las entidades dirige **más del {p90:.0%}** de su gasto a un único "
        f"proveedor y el 5 % supera el **{p95:.0%}**. Hay **{n_casi_total:,} entidades** "
        f"que concentran ≥90 % en un solo proveedor. El **HHI** (panel derecho) confirma el "
        f"patrón: mediana **{por_ent['hhi'].median():.2f}**, pero con entidades cuyo HHI se "
        f"acerca a 1 (mercado cautivo). Esa cola alta es la que alimenta la señal de "
        f"*concentración de proveedor*."
    )
    return ruta, interp


# ---------------------------------------------------------------------------
# b03 — Patrones temporales
# ---------------------------------------------------------------------------
def fig_temporal(feats: pd.DataFrame, df: pd.DataFrame) -> tuple[str, str]:
    """Órdenes por entidad-día y distribución dentro del mes (cierre de mes)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))

    # Órdenes por (entidad, día): histograma recortado al p99 para legibilidad.
    n_dia = feats["N_ORD_ENTIDAD_DIA"]
    tope = int(n_dia.quantile(0.99))
    ax1.hist(n_dia.clip(1, tope), bins=range(1, tope + 2), color="#1565c0", alpha=0.85)
    ax1.set_title("Órdenes de una misma entidad en un mismo día")
    ax1.set_xlabel(f"N.º de órdenes por entidad-día (recortado a p99={tope})")
    ax1.set_ylabel("N.º de órdenes")

    # Órdenes por día del mes: ¿se acumulan al cierre?
    dia_mes = df[cb.COL_FECHA_FORM].dt.day.values
    ax2.hist(dia_mes, bins=range(1, 33), color="#ad1457", alpha=0.85)
    ax2.set_title("Distribución de órdenes según el día del mes")
    ax2.set_xlabel("Día del mes de la formalización")
    ax2.set_ylabel("N.º de órdenes")
    ruta = _guardar(fig, f"{cb.PREFIJO_FIG_B}03_temporal.png")

    p_cierre = float(feats["FLAG_CIERRE_MES"].mean())
    # Comparación: media diaria en últimos 5 días vs resto.
    ult = (dia_mes >= 26).mean()
    interp = (
        f"La mayoría de las entidades emite **pocas órdenes por día** (mediana de "
        f"{int(n_dia.median())}), pero existe una cola de jornadas con decenas de órdenes "
        f"el mismo día (máximo **{int(n_dia.max())}** órdenes de una entidad en una sola "
        f"fecha): formalizar muchas órdenes de golpe puede indicar un fraccionamiento o un "
        f"cierre presupuestal apresurado. En cuanto al **día del mes** (panel derecho), "
        f"se observa una **acumulación hacia el final**: el **{p_cierre:.0%}** de las "
        f"órdenes se formaliza en los últimos {cb.DIAS_CIERRE_MES} días del mes y el "
        f"{ult:.0%} a partir del día 26. Esta concentración al cierre es un patrón temporal "
        f"que el módulo registra como factor de riesgo cuando coincide con otras señales."
    )
    return ruta, interp


# ---------------------------------------------------------------------------
# b04 — Vista general de la red entidad–proveedor
# ---------------------------------------------------------------------------
def fig_red(feats: pd.DataFrame, df: pd.DataFrame) -> tuple[str, str]:
    """Distribución de grados del grafo bipartito y un ego-grafo ilustrativo."""
    # Grado de cada entidad (nº de proveedores) y de cada proveedor (nº de entidades).
    prov_por_ent = df.groupby(cb.COL_RUC_ENTIDAD)[cb.COL_RUC_PROVEEDOR].nunique()
    ent_por_prov = df.groupby(cb.COL_RUC_PROVEEDOR)[cb.COL_RUC_ENTIDAD].nunique()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].hist(prov_por_ent, bins=50, color="#00695c", alpha=0.85)
    axes[0].set_title("Proveedores distintos por entidad (grado)")
    axes[0].set_xlabel("N.º de proveedores")
    axes[0].set_ylabel("N.º de entidades")

    axes[1].hist(ent_por_prov, bins=50, color="#ef6c00", alpha=0.85)
    axes[1].set_title("Entidades distintas por proveedor (grado)")
    axes[1].set_xlabel("N.º de entidades")
    axes[1].set_ylabel("N.º de proveedores")

    # Ego-grafo: la entidad con MAYOR concentración (share del top más alto y
    # con un volumen mínimo de órdenes) y sus proveedores. Ilustra una relación
    # potencialmente cautiva.
    cand = (
        feats.groupby(cb.COL_RUC_ENTIDAD)
        .agg(share=("SHARE_GASTO_PROV", "max"), n=("SHARE_GASTO_PROV", "size"))
        .query("n >= 20")
        .sort_values("share", ascending=False)
    )
    ent_focal = cand.index[0]
    sub = df[df[cb.COL_RUC_ENTIDAD] == ent_focal]
    gasto_prov = sub.groupby(cb.COL_RUC_PROVEEDOR)[cb.COL_TOTAL].sum().sort_values(ascending=False)
    G = nx.Graph()
    G.add_node("ENTIDAD", tipo="entidad")
    for prov, gasto in gasto_prov.head(12).items():
        G.add_node(prov, tipo="prov")
        G.add_edge("ENTIDAD", prov, weight=gasto)
    pos = nx.spring_layout(G, seed=cb.SEMILLA, k=0.9)
    pesos = np.array([G[u][v]["weight"] for u, v in G.edges()])
    anchos = 1 + 5 * pesos / pesos.max()
    nx.draw_networkx_edges(G, pos, ax=axes[2], width=anchos, edge_color="#9e9e9e")
    nx.draw_networkx_nodes(
        G, pos, ax=axes[2],
        nodelist=["ENTIDAD"], node_color="#c62828", node_size=600,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=axes[2],
        nodelist=[n for n in G.nodes if n != "ENTIDAD"],
        node_color="#1565c0", node_size=180,
    )
    axes[2].set_title("Ego-grafo: entidad más concentrada\n(grosor = gasto)")
    axes[2].axis("off")
    ruta = _guardar(fig, f"{cb.PREFIJO_FIG_B}04_red.png")

    interp = (
        f"La red entidad–proveedor es un grafo bipartito con "
        f"**{df[cb.COL_RUC_ENTIDAD].nunique():,} entidades**, "
        f"**{df[cb.COL_RUC_PROVEEDOR].nunique():,} proveedores** y "
        f"**{df.groupby([cb.COL_RUC_ENTIDAD, cb.COL_RUC_PROVEEDOR]).ngroups:,} relaciones** "
        f"distintas. La mayoría de entidades trabaja con **pocos proveedores** (mediana de "
        f"**{int(prov_por_ent.median())}**), aunque algunas alcanzan más de "
        f"{int(prov_por_ent.quantile(0.99))}. Del lado de los proveedores, la mayoría "
        f"atiende a pocas entidades (mediana **{int(ent_por_prov.median())}**), pero unos "
        f"pocos son *hubs* que sirven a cientos (máximo **{int(ent_por_prov.max())}** "
        f"entidades). El ego-grafo muestra el caso extremo: una entidad que vuelca casi "
        f"todo su gasto en un único proveedor (arista gruesa dominante), justo el tipo de "
        f"relación atípica que el análisis de grafos busca aislar."
    )
    return ruta, interp


# ---------------------------------------------------------------------------
# b05 — Punto de corte del fraccionamiento
# ---------------------------------------------------------------------------
def fig_umbral_fraccionamiento(df: pd.DataFrame, meta: dict) -> tuple[str, str]:
    """P(Gran Compra | TOTAL) y bunching de Compras ordinarias bajo el umbral."""
    info = meta["umbral_fraccionamiento"]
    umbral = info["umbral"]
    tabla = info["tabla"].reset_index()
    tabla["centro"] = np.array([iv.mid for iv in tabla["tramo"]], dtype=float)
    tabla = tabla[tabla["centro"] <= 200000]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
    ax1.plot(tabla["centro"], tabla["p_gc"], marker="o", ms=3, color="#1f4e79")
    ax1.axvline(umbral, color="red", ls="--", lw=1.2, label=f"umbral S/ {umbral:,.0f}")
    ax1.set_title("P(Gran Compra | TOTAL)")
    ax1.set_xlabel("TOTAL (S/)")
    ax1.set_ylabel("Probabilidad de ser Gran Compra")
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax1.legend()

    # Bunching: histograma fino de Compra ordinaria alrededor del umbral.
    co = df[(df[cb.COL_TIPO_PROC] == "Compra ordinaria")]
    rango = co[(co[cb.COL_TOTAL] >= 0.6 * umbral) & (co[cb.COL_TOTAL] <= 1.4 * umbral)]
    ax2.hist(rango[cb.COL_TOTAL], bins=40, color="#c55a11", alpha=0.85)
    ax2.axvline(umbral, color="red", ls="--", lw=1.2, label=f"umbral S/ {umbral:,.0f}")
    ax2.set_title("Bunching de Compras ordinarias bajo el umbral")
    ax2.set_xlabel("TOTAL (S/)")
    ax2.set_ylabel("N.º de órdenes")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    ax2.legend()
    ruta = _guardar(fig, f"{cb.PREFIJO_FIG_B}05_umbral_fraccionamiento.png")

    # Cifras del bunching: tramo justo bajo el umbral vs justo encima.
    just_below = int(((co[cb.COL_TOTAL] >= 0.98 * umbral) & (co[cb.COL_TOTAL] < umbral)).sum())
    just_above = int(((co[cb.COL_TOTAL] >= umbral) & (co[cb.COL_TOTAL] < 1.02 * umbral)).sum())
    interp = (
        f"El punto de corte del fraccionamiento **no se asumió**: se descubrió a partir de "
        f"la relación real entre `TOTAL` y `TIPO_PROCEDIMIENTO`. La probabilidad de que una "
        f"orden sea **Gran Compra** salta de **{info['p_antes']:.2f}** a **{info['p_despues']:.2f}** "
        f"justo al cruzar **S/ {umbral:,.0f}** (panel izquierdo): por encima de ese monto el "
        f"procedimiento competitivo se vuelve obligatorio. El panel derecho muestra el "
        f"**bunching**: hay **{just_below:,} Compras ordinarias** apretadas en el último 2 % "
        f"por debajo del umbral frente a solo **{just_above:,}** apenas por encima. Esa "
        f"acumulación justo debajo del límite es la huella típica de la manipulación de "
        f"umbral; por eso el módulo vigila las órdenes pegadas a S/ {umbral:,.0f}."
    )
    return ruta, interp


# ---------------------------------------------------------------------------
# Orquestación del EDA
# ---------------------------------------------------------------------------
def ejecutar_eda(feats: pd.DataFrame, df: pd.DataFrame, meta: dict) -> str:
    """
    Genera todas las figuras del EDA del Módulo B y devuelve la sección del
    informe en Markdown (figuras + interpretaciones).
    """
    logger.info("Generando EDA del Módulo B...")
    bloques: list[str] = ["## 2. Análisis exploratorio orientado a anomalías\n"]

    for titulo, (ruta, interp) in [
        ("2.1 Montos e IGV", fig_montos_igv(feats, meta)),
        ("2.2 Concentración de proveedor", fig_concentracion(feats)),
        ("2.3 Patrones temporales", fig_temporal(feats, df)),
        ("2.4 Red entidad–proveedor", fig_red(feats, df)),
        ("2.5 Punto de corte del fraccionamiento", fig_umbral_fraccionamiento(df, meta)),
    ]:
        bloques.append(f"### {titulo}\n\n![{titulo}]({ruta})\n\n{interp}\n")

    return "\n".join(bloques)
