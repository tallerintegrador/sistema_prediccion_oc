"""
riesgo.py
=========
Puntaje de riesgo y ranking de alertas.

Combina las señales del Módulo B en un único **puntaje de riesgo por orden**,
clasifica las órdenes en niveles (alto / medio / bajo) con cortes DERIVADOS de la
propia distribución del puntaje, etiqueta cada alerta con el **tipo de anomalía
predominante** y arma el **ranking de alertas** con una descripción del motivo.

Señales por tipo de anomalía (todas en [0, 1], donde 1 = más anómala):

    monto          -> desviación robusta del log-monto (vs categoría y global).
    igv            -> magnitud de la desviación del ratio IGV/SUB_TOTAL respecto a 0.18.
    concentracion  -> proporción del gasto de la entidad en el proveedor de la orden.
    fraccionamiento-> ráfaga de órdenes del par bajo el umbral derivado de los datos.
    temporal       -> órdenes de la entidad el mismo día + cierre de mes + demoras.
    red            -> puntaje del análisis de grafo (lock-in, proveedor cautivo).

El puntaje combinado mezcla los detectores no supervisados (Isolation Forest y
autoencoder) con el máximo de las señales interpretables y la señal de red.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ml.shared import config as cb

logger = logging.getLogger(__name__)

# Nombre legible de cada señal por tipo de anomalía.
TIPOS = {
    "s_monto": "Monto atípico",
    "s_igv": "IGV inconsistente",
    "s_concentracion": "Concentración de proveedor",
    "s_fracc": "Posible fraccionamiento",
    "s_temporal": "Patrón temporal",
    "s_red": "Anomalía de red",
}


def _pct(x: pd.Series | np.ndarray) -> np.ndarray:
    """Rango percentil en [0, 1] (mayor valor -> percentil más alto)."""
    return pd.Series(np.asarray(x, dtype=float)).rank(pct=True).to_numpy()


def construir_senales(feats: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """
    Construye, por orden, las seis señales de anomalía normalizadas a [0, 1].

    Devuelve un DataFrame con columnas s_monto, s_igv, s_concentracion, s_fracc,
    s_temporal y s_red, alineado al índice de `feats`.
    """
    s = pd.DataFrame(index=feats.index)

    # Monto: solo desviaciones POSITIVAS (montos altos) respecto a categoría/global.
    z_monto = np.maximum.reduce([
        feats["Z_TOTAL_CAT"].clip(lower=0).to_numpy(),
        feats["Z_TOTAL_GLOBAL"].clip(lower=0).to_numpy(),
    ])
    s["s_monto"] = _pct(z_monto)

    # IGV: la desviación máxima posible respecto a 0.18 es 0.18 (ratio = 0).
    s["s_igv"] = (feats["ABS_DESV_IGV"] / cb.IGV_LEGAL).clip(0, 1).to_numpy()

    # Concentración: proporción del gasto de la entidad en este proveedor.
    s["s_concentracion"] = feats["SHARE_GASTO_PROV"].clip(0, 1).to_numpy()

    # Fraccionamiento: ráfaga que cruza el umbral (flag) + cercanía al umbral.
    s["s_fracc"] = (
        0.6 * feats["FLAG_FRACCIONAMIENTO"] + 0.4 * feats["PROXIMIDAD_UMBRAL"]
    ).clip(0, 1).to_numpy()

    # Temporal: órdenes mismo día (graduado) + cierre de mes + demora de estado.
    s["s_temporal"] = (
        0.6 * _pct(feats["N_ORD_ENTIDAD_DIA"])
        + 0.2 * feats["FLAG_CIERRE_MES"].to_numpy()
        + 0.2 * _pct(feats["DIAS_ESTADO"])
    ).clip(0, 1)

    # Red: puntaje del análisis de grafo.
    s["s_red"] = scores["SCORE_RED"].to_numpy()

    return s


def _descripcion_motivo(fila: pd.Series, umbral: float) -> str:
    """Arma una descripción breve en español del motivo de la alerta."""
    partes: list[str] = []
    if fila["s_monto"] >= 0.99:
        partes.append(
            f"monto S/ {fila['TOTAL']:,.0f} muy por encima de lo habitual en su categoría"
        )
    if fila["FLAG_IGV"] == 1:
        partes.append(f"ratio IGV {fila['RATIO_IGV']:.3f} (esperado 0.18)")
    if fila["s_concentracion"] >= 0.80:
        partes.append(
            f"la entidad concentra el {fila['SHARE_GASTO_PROV']:.0%} de su gasto en este proveedor"
        )
    if fila["FLAG_FRACCIONAMIENTO"] == 1:
        partes.append(
            f"{int(fila['N_ORD_VENTANA_7D'])} órdenes del par en 7 días sumando "
            f"S/ {fila['SUMA_VENTANA_7D']:,.0f} (umbral S/ {umbral:,.0f})"
        )
    elif fila["FLAG_BAJO_UMBRAL"] == 1:
        partes.append(f"monto pegado al umbral de S/ {umbral:,.0f}")
    if fila["N_ORD_ENTIDAD_DIA"] >= 20:
        partes.append(f"{int(fila['N_ORD_ENTIDAD_DIA'])} órdenes de la entidad ese mismo día")
    if fila["FLAG_PROV_CAUTIVO"] == 1:
        partes.append("proveedor cautivo (atiende a una sola entidad)")
    elif fila["LOCK_IN"] >= 0.5:
        partes.append("relación entidad–proveedor bilateralmente exclusiva")
    if not partes:
        partes.append("patrón multivariado atípico detectado por el modelo")
    return "; ".join(partes).capitalize() + "."


def calcular_riesgo(
    df: pd.DataFrame, feats: pd.DataFrame, scores: pd.DataFrame, meta: dict
) -> tuple[pd.DataFrame, dict]:
    """
    Calcula el puntaje de riesgo, el nivel, el tipo de anomalía y el motivo por
    orden, y devuelve la tabla completa rankeada por riesgo.

    Parámetros
    ----------
    df : DataFrame limpio (para recuperar nombres de entidad/proveedor, fecha...).
    feats : features de anomalía.
    scores : puntajes de los modelos (SCORE_IF, SCORE_AE, SCORE_RED, red...).
    meta : metadatos con los umbrales derivados.

    Devuelve
    --------
    (tabla, info) : tuple[pd.DataFrame, dict]
        `tabla` con una fila por orden ordenada por PUNTAJE_RIESGO; `info` con los
        cortes de nivel y el conteo por nivel y por tipo.
    """
    umbral = meta["umbral_usado"]
    senales = construir_senales(feats, scores)

    # --- Puntaje combinado ----------------------------------------------------
    # Detectores no supervisados (holísticos).
    cols_modelo = [c for c in ("SCORE_IF", "SCORE_AE") if c in scores.columns]
    score_modelo = scores[cols_modelo].mean(axis=1).to_numpy()
    # Máximo de las señales interpretables (la "bandera roja" más fuerte).
    score_reglas = senales[list(TIPOS)].max(axis=1).to_numpy()
    # Mezcla mitad modelo / mitad reglas, y se reexpresa como percentil [0,1].
    puntaje = 0.5 * score_modelo + 0.5 * score_reglas
    puntaje_pct = _pct(puntaje)

    # --- Tipo de anomalía predominante ----------------------------------------
    # Se decide por la señal en la que la orden es MÁS extrema respecto a la
    # población (argmax de las señales llevadas a percentil). Así una señal que se
    # satura para muchas órdenes (p. ej. IGV=0) no acapara siempre la etiqueta
    # cuando otra señal —como el fraccionamiento— es más rara y más informativa.
    senales_pct = pd.DataFrame(
        {c: _pct(senales[c]) for c in TIPOS}, index=senales.index
    )
    arg = senales_pct.to_numpy().argmax(axis=1)
    tipo = np.array([TIPOS[list(TIPOS)[i]] for i in arg])

    # --- Niveles con cortes derivados de la distribución del puntaje ----------
    corte_alto = float(np.quantile(puntaje_pct, cb.CUANTIL_RIESGO_ALTO))
    corte_medio = float(np.quantile(puntaje_pct, cb.CUANTIL_RIESGO_MEDIO))
    nivel = np.where(
        puntaje_pct >= corte_alto, "alto",
        np.where(puntaje_pct >= corte_medio, "medio", "bajo"),
    )
    # Cortes en el espacio del puntaje BRUTO (para visualizar su distribución
    # real, que sí es asimétrica; el percentil, en cambio, es uniforme por
    # construcción y no se grafica).
    corte_alto_bruto = float(np.quantile(puntaje, cb.CUANTIL_RIESGO_ALTO))
    corte_medio_bruto = float(np.quantile(puntaje, cb.CUANTIL_RIESGO_MEDIO))

    # --- Ensamblado de la tabla -----------------------------------------------
    tabla = pd.DataFrame(index=feats.index)
    tabla["ORDEN_ELECTRONICA"] = feats[cb.COL_ORDEN].to_numpy()
    tabla["RUC_ENTIDAD"] = df[cb.COL_RUC_ENTIDAD].to_numpy()
    tabla["ENTIDAD"] = df[cb.COL_ENTIDAD].to_numpy()
    tabla["RUC_PROVEEDOR"] = df[cb.COL_RUC_PROVEEDOR].to_numpy()
    tabla["PROVEEDOR"] = df[cb.COL_PROVEEDOR].to_numpy()
    tabla["ACUERDO_MARCO"] = df[cb.COL_ACUERDO].to_numpy()
    tabla["FECHA_FORMALIZACION"] = df[cb.COL_FECHA_FORM].to_numpy()
    tabla["TOTAL"] = feats["TOTAL"].to_numpy()
    tabla["TIPO_ANOMALIA"] = tipo
    tabla["PUNTAJE_RIESGO"] = puntaje_pct.round(4)
    tabla["PUNTAJE_BRUTO"] = np.round(puntaje, 4)   # combinación antes de rankear (para graficar)
    tabla["NIVEL"] = nivel
    # Puntajes por componente (trazabilidad).
    tabla["SCORE_IF"] = scores["SCORE_IF"].round(4).to_numpy()
    if "SCORE_AE" in scores.columns:
        tabla["SCORE_AE"] = scores["SCORE_AE"].round(4).to_numpy()
    tabla["SCORE_RED"] = scores["SCORE_RED"].round(4).to_numpy()
    for c in TIPOS:
        tabla[c] = senales[c].round(4).to_numpy()

    # Columnas auxiliares para construir el motivo (se quitan al final).
    aux = pd.DataFrame(index=feats.index)
    for c in ("RATIO_IGV", "FLAG_IGV", "SHARE_GASTO_PROV", "FLAG_FRACCIONAMIENTO",
              "FLAG_BAJO_UMBRAL", "N_ORD_VENTANA_7D", "SUMA_VENTANA_7D", "N_ORD_ENTIDAD_DIA"):
        aux[c] = feats[c].to_numpy()
    aux["LOCK_IN"] = scores["LOCK_IN"].to_numpy()
    aux["FLAG_PROV_CAUTIVO"] = scores["FLAG_PROV_CAUTIVO"].to_numpy()
    base_motivo = pd.concat([tabla[["TOTAL", "s_monto", "s_concentracion"]], aux], axis=1)

    # El motivo solo se redacta para las alertas reales (nivel medio/alto) por
    # eficiencia; el resto lleva una etiqueta genérica.
    es_alerta = tabla["NIVEL"] != "bajo"
    motivos = pd.Series("Riesgo bajo: sin señales relevantes.", index=tabla.index)
    motivos.loc[es_alerta] = base_motivo.loc[es_alerta].apply(
        lambda r: _descripcion_motivo(r, umbral), axis=1
    )
    tabla["MOTIVO"] = motivos.to_numpy()

    tabla = tabla.sort_values("PUNTAJE_RIESGO", ascending=False).reset_index(drop=True)

    info = {
        "corte_alto": corte_alto,
        "corte_medio": corte_medio,
        "corte_alto_bruto": corte_alto_bruto,
        "corte_medio_bruto": corte_medio_bruto,
        "conteo_nivel": tabla["NIVEL"].value_counts().to_dict(),
        "conteo_tipo": tabla.loc[tabla["NIVEL"] != "bajo", "TIPO_ANOMALIA"]
        .value_counts()
        .to_dict(),
    }
    logger.info(
        "Riesgo: %d alto, %d medio, %d bajo. Cortes (pct): medio=%.3f, alto=%.3f.",
        info["conteo_nivel"].get("alto", 0), info["conteo_nivel"].get("medio", 0),
        info["conteo_nivel"].get("bajo", 0), corte_medio, corte_alto,
    )
    return tabla, info


def exportar_alertas(tabla: pd.DataFrame, solo_alertas: bool = True) -> pd.DataFrame:
    """
    Exporta la tabla de alertas rankeada a `outputs/resultados/alertas.csv`.

    Por defecto exporta solo las órdenes con nivel medio/alto (las que ameritan
    revisión). Devuelve el subconjunto exportado.
    """
    salida = tabla[tabla["NIVEL"] != "bajo"].copy() if solo_alertas else tabla.copy()
    columnas = [
        "ORDEN_ELECTRONICA", "RUC_ENTIDAD", "ENTIDAD", "RUC_PROVEEDOR", "PROVEEDOR",
        "ACUERDO_MARCO", "FECHA_FORMALIZACION", "TOTAL", "TIPO_ANOMALIA",
        "PUNTAJE_RIESGO", "NIVEL", "MOTIVO",
        "SCORE_IF", "SCORE_RED",
    ]
    columnas = [c for c in columnas if c in salida.columns]
    salida[columnas].to_csv(cb.RUTA_ALERTAS_CSV, index=False, encoding="utf-8-sig")
    logger.info("Alertas exportadas a %s (%d filas).", cb.RUTA_ALERTAS_CSV.name, len(salida))
    return salida
