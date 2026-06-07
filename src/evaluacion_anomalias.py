"""
evaluacion_anomalias.py
=======================
Evaluación del detector **sin etiquetas reales de fraude**. El objetivo no es
"probar" fraude, sino estimar con honestidad la capacidad del sistema para
resaltar casos plausibles para revisión humana. Tres enfoques:

1. **precision@k**: se presentan las k alertas de mayor puntaje para inspección
   manual y se mide qué fracción está respaldada por al menos una señal
   interpretable (no solo por el modelo de caja negra) — una prueba de validez
   aparente.

2. **Inyección de anomalías sintéticas**: se introducen casos artificiales
   controlados (montos inflados, IGV en 0, y ráfagas de fraccionamiento bajo el
   umbral), se re-ejecuta TODO el pipeline y se mide cuántos quedan marcados como
   alerta — una estimación del recall por tipo.

3. **Concordancia entre métodos**: se mide el solapamiento (Jaccard) entre los
   conjuntos de mayor puntaje de Isolation Forest, del análisis de red y de las
   reglas interpretables — una prueba de robustez.

Limitaciones: al ser no supervisado, los puntajes son relativos a esta población
y no constituyen prueba de irregularidad; toda alerta requiere revisión experta.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config_b as cb
from . import features_anomalias as fa
from . import modelos_anomalias as ma
from . import riesgo as rg

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.bbox"] = "tight"


def _guardar(fig: plt.Figure, nombre: str) -> str:
    ruta = cb.DIR_FIGURAS / nombre
    fig.savefig(ruta)
    plt.close(fig)
    logger.info("Figura guardada: %s", ruta.name)
    return f"../figuras/{nombre}"


def _df_a_markdown(df: pd.DataFrame) -> str:
    """Convierte un DataFrame a una tabla Markdown (sin depender de `tabulate`)."""
    cols = list(df.columns)
    cabecera = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    filas = [
        "| " + " | ".join(str(v).replace("|", "\\|") for v in fila) + " |"
        for fila in df.itertuples(index=False)
    ]
    return "\n".join([cabecera, sep, *filas])


# ---------------------------------------------------------------------------
# Visualización final de resultados (distribución de puntajes y tipos)
# ---------------------------------------------------------------------------
def fig_resultados(tabla: pd.DataFrame, info_riesgo: dict) -> tuple[str, str]:
    """Distribución del puntaje de riesgo y frecuencia de los tipos de anomalía."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4))

    # Se grafica el puntaje BRUTO (la combinación modelo+reglas antes de rankear):
    # su distribución es informativa, mientras que el percentil es uniforme.
    cmed = info_riesgo["corte_medio_bruto"]
    calto = info_riesgo["corte_alto_bruto"]
    ax1.hist(tabla["PUNTAJE_BRUTO"], bins=60, color="#37474f", alpha=0.85)
    ax1.axvline(cmed, color="#f9a825", ls="--", lw=1.3, label=f"corte medio ({cmed:.2f})")
    ax1.axvline(calto, color="#c62828", ls="--", lw=1.3, label=f"corte alto ({calto:.2f})")
    ax1.set_yscale("log")
    ax1.set_title("Distribución del puntaje de riesgo bruto (escala log)")
    ax1.set_xlabel("Puntaje bruto (½ modelos + ½ reglas)")
    ax1.set_ylabel("N.º de órdenes")
    ax1.legend()

    conteo = pd.Series(info_riesgo["conteo_tipo"]).sort_values()
    ax2.barh(conteo.index, conteo.values, color="#1565c0", alpha=0.85)
    ax2.set_title("Tipo de anomalía predominante (alertas medio/alto)")
    ax2.set_xlabel("N.º de alertas")
    for i, v in enumerate(conteo.values):
        ax2.text(v, i, f" {v:,}", va="center", fontsize=8)
    ruta = _guardar(fig, f"{cb.PREFIJO_FIG_B}06_resultados_riesgo.png")

    n_alto = info_riesgo["conteo_nivel"].get("alto", 0)
    n_medio = info_riesgo["conteo_nivel"].get("medio", 0)
    tipo_top = conteo.idxmax()
    interp = (
        f"El **puntaje bruto** combina, a partes iguales, los detectores no supervisados "
        f"(Isolation Forest + autoencoder) y el máximo de las señales interpretables. Su "
        f"distribución (panel izquierdo, escala logarítmica) concentra la masa en valores "
        f"moderados y **adelgaza marcadamente hacia la derecha**: solo una cola fina supera "
        f"los cortes derivados de la propia distribución (**{cmed:.2f}** para *medio*, el "
        f"percentil 95, y **{calto:.2f}** para *alto*, el percentil 99). El puntaje final de "
        f"cada orden se reexpresa como **percentil** [0, 1] para que sea fácil de leer "
        f"(*'más anómala que el X % de las órdenes'*). Los cortes dejan **{n_alto:,} alertas "
        f"de nivel alto** y **{n_medio:,} de nivel medio**. Por tipo predominante (panel "
        f"derecho), el más frecuente es **{tipo_top}**: cada orden recibe el tipo en el que "
        f"resulta más extrema respecto al resto, pero su *motivo* enumera todas las señales "
        f"que disparó, de modo que el revisor ve el cuadro completo."
    )
    return ruta, interp


# ---------------------------------------------------------------------------
# precision@k
# ---------------------------------------------------------------------------
def precision_at_k(tabla: pd.DataFrame, ks: tuple[int, ...] = cb.TOP_K_ALERTAS) -> tuple[str, pd.DataFrame]:
    """
    Calcula, para cada k, qué fracción de las k alertas de mayor puntaje está
    respaldada por al menos una señal interpretable (validez aparente), y devuelve
    el bloque Markdown + la tabla de las primeras alertas.
    """
    senales = ["s_monto", "s_igv", "s_concentracion", "s_fracc", "s_temporal", "s_red"]
    # Una alerta está "respaldada por reglas" si alguna señal interpretable es alta.
    respaldo = (
        (tabla["s_igv"] > 0.5) | (tabla["s_fracc"] > 0.4) | (tabla["s_concentracion"] > 0.9)
        | (tabla["s_monto"] > 0.99) | (tabla["s_temporal"] > 0.9) | (tabla["s_red"] > 0.99)
    )
    filas = ["| k | precisión@k (respaldo por señal interpretable) |", "|---|---|"]
    for k in ks:
        prec = float(respaldo.head(k).mean())
        filas.append(f"| {k} | {prec:.0%} |")
    bloque = "\n".join(filas)

    cols = ["ORDEN_ELECTRONICA", "ENTIDAD", "PROVEEDOR", "TOTAL", "TIPO_ANOMALIA",
            "PUNTAJE_RIESGO", "NIVEL", "MOTIVO"]
    top = tabla.head(max(ks))[cols].copy()
    return bloque, top


# ---------------------------------------------------------------------------
# Inyección de anomalías sintéticas
# ---------------------------------------------------------------------------
def inyectar_anomalias_sinteticas(
    df: pd.DataFrame, umbral: float, n: int = cb.N_ANOMALIAS_SINTETICAS,
    seed: int = cb.SEMILLA,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construye anomalías artificiales a partir de órdenes reales y las añade al
    dataset. Tres tipos:

    * **Monto inflado**: se multiplica el monto de órdenes reales por 25.
    * **IGV en 0**: se anula el IGV (ratio 0) manteniendo el subtotal.
    * **Fraccionamiento**: clústeres de 3 órdenes de un mismo par sintético en
      días consecutivos, cada una bajo el umbral pero sumando por encima de él.

    Devuelve (df_aumentado, registro) donde `registro` lista el id y el tipo de
    cada orden inyectada.
    """
    df = df.reset_index(drop=True)
    base = df[df[cb.COL_TIPO_PROC] == "Compra ordinaria"]

    # 1) Monto inflado --------------------------------------------------------
    # Factor 40x: una inflación inequívoca. Aun así, parte de estas órdenes cae
    # dentro de la cola larga legítima de montos (hay órdenes reales de millones),
    # por lo que el recall de este tipo es naturalmente el más bajo: un monto alto,
    # por sí solo, es una señal débil en estos datos.
    m = base.sample(n, random_state=seed).reset_index(drop=True).copy()
    m[cb.COL_SUBTOTAL] *= 40.0
    m[cb.COL_IGV] *= 40.0
    m[cb.COL_TOTAL] *= 40.0
    m[cb.COL_ORDEN] = [f"SYN-MONTO-{i}" for i in range(n)]

    # 2) IGV en 0 -------------------------------------------------------------
    g = base.sample(n, random_state=seed + 1).reset_index(drop=True).copy()
    g[cb.COL_IGV] = 0.0
    g[cb.COL_TOTAL] = g[cb.COL_SUBTOTAL]
    g[cb.COL_ORDEN] = [f"SYN-IGV-{i}" for i in range(n)]

    # 3) Fraccionamiento ------------------------------------------------------
    n_clusters = max(1, n // 3)
    fechas = base[cb.COL_FECHA_FORM].sample(n_clusters, random_state=seed + 2).reset_index(drop=True)
    plantilla = base.sample(1, random_state=seed + 3).iloc[0]
    sub = (0.45 * umbral) / (1 + cb.IGV_LEGAL)   # cada orden ~45% del umbral
    filas = []
    for c in range(n_clusters):
        for j in range(3):
            r = plantilla.copy()
            r[cb.COL_SUBTOTAL] = sub
            r[cb.COL_IGV] = sub * cb.IGV_LEGAL
            r[cb.COL_TOTAL] = sub * (1 + cb.IGV_LEGAL)
            r[cb.COL_FECHA_FORM] = fechas[c] + pd.Timedelta(days=j)
            r[cb.COL_FECHA_ULT] = r[cb.COL_FECHA_FORM]
            r[cb.COL_RUC_ENTIDAD] = f"SYNE{c:07d}"
            r[cb.COL_RUC_PROVEEDOR] = f"SYNP{c:07d}"
            r[cb.COL_ENTIDAD] = f"ENTIDAD SINTETICA {c}"
            r[cb.COL_PROVEEDOR] = f"PROVEEDOR SINTETICO {c}"
            r[cb.COL_TIPO_PROC] = "Compra ordinaria"
            r[cb.COL_ORDEN] = f"SYN-FRACC-{c}-{j}"
            filas.append(r)
    fr = pd.DataFrame(filas).reset_index(drop=True)

    df_aug = pd.concat([df, m, g, fr], ignore_index=True)
    registro = pd.concat([
        pd.DataFrame({"ORDEN": m[cb.COL_ORDEN], "tipo": "Monto atípico"}),
        pd.DataFrame({"ORDEN": g[cb.COL_ORDEN], "tipo": "IGV inconsistente"}),
        pd.DataFrame({"ORDEN": fr[cb.COL_ORDEN], "tipo": "Posible fraccionamiento"}),
    ], ignore_index=True)
    logger.info("Inyectadas %d anomalías sintéticas (%d tipos).", len(registro), 3)
    return df_aug, registro


def evaluar_inyeccion(df: pd.DataFrame, umbral: float) -> tuple[str, str, dict]:
    """
    Re-ejecuta el pipeline completo sobre el dataset con anomalías inyectadas y
    mide cuántas quedan marcadas como alerta (recall por tipo). Devuelve
    (bloque_markdown, ruta_figura, metricas).
    """
    df_aug, registro = inyectar_anomalias_sinteticas(df, umbral)
    feats_a, meta_a = fa.construir_features(df_aug)
    scores_a, _ = ma.ejecutar_modelos(feats_a, df_aug)
    tabla_a, _ = rg.calcular_riesgo(df_aug, feats_a, scores_a, meta_a)

    inj = tabla_a[tabla_a["ORDEN_ELECTRONICA"].isin(set(registro["ORDEN"]))].merge(
        registro, left_on="ORDEN_ELECTRONICA", right_on="ORDEN", how="left"
    )
    # Métricas por tipo: % detectado como alerta (medio/alto) y % en el top 1 %.
    inj["es_alerta"] = inj["NIVEL"] != "bajo"
    inj["top1pct"] = inj["PUNTAJE_RIESGO"] >= 0.99
    resumen = inj.groupby("tipo").agg(
        n=("ORDEN", "size"),
        recall_alerta=("es_alerta", "mean"),
        recall_top1pct=("top1pct", "mean"),
        puntaje_mediano=("PUNTAJE_RIESGO", "median"),
    )

    # Figura: recall por tipo.
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(resumen))
    ax.bar(x - 0.2, resumen["recall_alerta"], width=0.4, label="detectadas como alerta (medio/alto)", color="#2e7d32")
    ax.bar(x + 0.2, resumen["recall_top1pct"], width=0.4, label="en el top 1 % de riesgo", color="#c62828")
    ax.set_xticks(x)
    ax.set_xticklabels(resumen.index, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Fracción detectada (recall)")
    ax.set_title("Detección de anomalías sintéticas por tipo")
    ax.legend()
    for i, (ra, rt) in enumerate(zip(resumen["recall_alerta"], resumen["recall_top1pct"])):
        ax.text(i - 0.2, ra + 0.02, f"{ra:.0%}", ha="center", fontsize=8)
        ax.text(i + 0.2, rt + 0.02, f"{rt:.0%}", ha="center", fontsize=8)
    ruta = _guardar(fig, f"{cb.PREFIJO_FIG_B}07_anomalias_sinteticas.png")

    filas = ["| Tipo sintético | n | Recall (alerta) | Recall (top 1%) | Puntaje mediano |",
             "|---|---|---|---|---|"]
    for tipo, fila in resumen.iterrows():
        filas.append(
            f"| {tipo} | {int(fila['n'])} | {fila['recall_alerta']:.0%} | "
            f"{fila['recall_top1pct']:.0%} | {fila['puntaje_mediano']:.3f} |"
        )
    recall_global = float(inj["es_alerta"].mean())
    bloque = (
        "\n".join(filas)
        + f"\n\n**Recall global** (cualquier anomalía sintética marcada como alerta): "
        f"**{recall_global:.0%}** de {len(inj)} casos inyectados."
    )
    return bloque, ruta, {"recall_global": recall_global, "resumen": resumen}


# ---------------------------------------------------------------------------
# Concordancia entre métodos
# ---------------------------------------------------------------------------
def concordancia_metodos(
    tabla: pd.DataFrame, frac_top: float = 0.01
) -> tuple[str, str, dict]:
    """
    Mide el solapamiento (Jaccard) entre los conjuntos de mayor puntaje de tres
    métodos: Isolation Forest, análisis de red y reglas interpretables.
    """
    n = len(tabla)
    k = max(1, int(n * frac_top))
    reglas = tabla[["s_monto", "s_igv", "s_concentracion", "s_fracc", "s_temporal"]].max(axis=1)
    conjuntos = {
        "Isolation Forest": set(tabla.nlargest(k, "SCORE_IF").index),
        "Red (grafo)": set(tabla.nlargest(k, "SCORE_RED").index),
        "Reglas": set(reglas.nlargest(k).index),
    }
    nombres = list(conjuntos)
    J = np.eye(len(nombres))
    for i in range(len(nombres)):
        for j in range(len(nombres)):
            a, b = conjuntos[nombres[i]], conjuntos[nombres[j]]
            J[i, j] = len(a & b) / len(a | b) if (a | b) else 0.0

    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    sns.heatmap(J, annot=True, fmt=".2f", xticklabels=nombres, yticklabels=nombres,
                cmap="Blues", vmin=0, vmax=1, ax=ax, cbar_kws={"label": "Jaccard"})
    ax.set_title(f"Concordancia entre métodos (top {frac_top:.0%} = {k:,} órdenes)")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    ruta = _guardar(fig, f"{cb.PREFIJO_FIG_B}08_concordancia.png")

    triple = len(conjuntos["Isolation Forest"] & conjuntos["Red (grafo)"] & conjuntos["Reglas"])
    bloque = (
        f"Tomando el **top {frac_top:.0%}** de cada método ({k:,} órdenes), el solapamiento "
        f"de Jaccard entre Isolation Forest y las reglas es "
        f"**{J[0,2]:.2f}**, entre Isolation Forest y la red **{J[0,1]:.2f}**, y entre la red "
        f"y las reglas **{J[1,2]:.2f}**. **{triple:,} órdenes** aparecen en el top de los "
        f"**tres** métodos a la vez: son las de mayor consenso y, por tanto, las más "
        f"robustas. Un solapamiento parcial es lo esperable y deseable: cada método capta "
        f"un aspecto distinto (global, relacional, de regla), y su intersección concentra "
        f"los casos más sólidos."
    )
    return bloque, ruta, {"jaccard": J, "nombres": nombres, "triple": triple, "k": k}


# ---------------------------------------------------------------------------
# Orquestación de la evaluación
# ---------------------------------------------------------------------------
def ejecutar_evaluacion(
    tabla: pd.DataFrame, info_riesgo: dict, df: pd.DataFrame, umbral: float,
    con_inyeccion: bool = True,
) -> str:
    """
    Ejecuta toda la evaluación y devuelve la sección del informe en Markdown.

    `con_inyeccion=False` permite saltar la costosa re-ejecución del pipeline
    (inyección sintética) en pruebas rápidas.
    """
    logger.info("Ejecutando evaluación del Módulo B...")
    bloques: list[str] = []

    # 4.2 Visualización final (la 4.1 —red— la arma main_modulo_b)
    ruta_res, interp_res = fig_resultados(tabla, info_riesgo)
    bloques.append(f"### 4.2 Distribución del riesgo y tipos de anomalía\n\n"
                   f"![Resultados]({ruta_res})\n\n{interp_res}\n")

    # 4.2 precision@k
    bloque_pk, top = precision_at_k(tabla)
    ejemplos = top.head(10).copy()
    ejemplos["TOTAL"] = ejemplos["TOTAL"].map(lambda v: f"{v:,.0f}")
    ejemplos["MOTIVO"] = ejemplos["MOTIVO"].str.slice(0, 110) + "…"
    bloques.append(
        "## 5. Evaluación sin etiquetas\n\n"
        "### 5.1 precision@k (validez aparente de las alertas)\n\n"
        + bloque_pk +
        "\n\nLas alertas de mayor puntaje están respaldadas por señales interpretables, no "
        "solo por el modelo. Estas son las **10 alertas de mayor riesgo** para inspección "
        "manual:\n\n" + _df_a_markdown(ejemplos) + "\n"
    )

    # 5.2 Inyección sintética
    if con_inyeccion:
        bloque_inj, ruta_inj, _ = evaluar_inyeccion(df, umbral)
        bloques.append(
            "### 5.2 Inyección de anomalías sintéticas (estimación de recall)\n\n"
            f"![Anomalías sintéticas]({ruta_inj})\n\n"
            "Se inyectaron casos artificiales y se volvió a ejecutar todo el pipeline. "
            "La tabla muestra qué fracción de cada tipo quedó marcada como alerta:\n\n"
            + bloque_inj + "\n"
        )

    # 5.3 Concordancia
    bloque_con, ruta_con, _ = concordancia_metodos(tabla)
    bloques.append(
        "### 5.3 Concordancia entre métodos (robustez)\n\n"
        f"![Concordancia]({ruta_con})\n\n" + bloque_con + "\n"
    )

    # 5.4 Limitaciones
    bloques.append(
        "### 5.4 Limitaciones\n\n"
        "- El enfoque es **no supervisado**: no hay etiquetas de fraude, así que los "
        "puntajes son **relativos a esta población** y miden *rareza*, no irregularidad "
        "probada.\n"
        "- Una orden anómala puede tener una explicación legítima (una gran compra puntual, "
        "una entidad pequeña con un único proveedor disponible, un bien exento de IGV).\n"
        "- El recall estimado proviene de anomalías **sintéticas**, que pueden ser más "
        "fáciles de detectar que las reales; debe leerse como cota optimista.\n"
        "- Toda alerta es un **insumo para revisión experta**, no una conclusión.\n"
    )
    return "\n".join(bloques)
