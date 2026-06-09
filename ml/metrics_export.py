"""
metrics_export.py
=================
Exporta TODAS las métricas del ML a JSON consumible por el backend/frontend.

Motivación (capstone): el ML calcula métricas ricas (comparación de 13 modelos del
Módulo A; precision@k, recall por inyección sintética, concordancia Jaccard y
estadísticos de red del Módulo B), pero hasta ahora vivían solo en CSV sueltos y en
los informes Markdown. Este módulo las consolida en dos archivos estructurados:

    artifacts/resultados/metrics_modulo_a.json
    artifacts/resultados/metrics_modulo_b.json

que el API expone en /metrics/forecast y /metrics/anomalies.

Diseño: es un paso de POST-PROCESO independiente que lee los artefactos ya
generados por los pipelines (no reentrena nada). Las distribuciones del Módulo B se
calculan en vivo desde `alertas.csv` (siempre consistentes con los datos); las
métricas de evaluación (precision@k, recall, concordancia, red) se leen del informe
`informe_modulo_b.md`, que es la salida autoritativa de la última corrida.

Uso (desde la raíz del proyecto):
    python -m ml.metrics_export
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.shared import config

logger = logging.getLogger(__name__)

RUTA_METRICAS_A: Path = config.DIR_RESULTADOS / "metrics_modulo_a.json"
RUTA_METRICAS_B: Path = config.DIR_RESULTADOS / "metrics_modulo_b.json"


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _num(texto: str | None) -> float | None:
    """Convierte '1,214' / '74%' / '0.16' a float; None si no se puede."""
    if texto is None:
        return None
    t = texto.strip().replace(",", "").replace("%", "")
    try:
        return float(t)
    except ValueError:
        return None


def _buscar(patron: str, texto: str, grupo: int = 1) -> str | None:
    """Primer match de un patrón en el texto, o None."""
    m = re.search(patron, texto)
    return m.group(grupo) if m else None


def _escribir_json(ruta: Path, datos: dict[str, Any]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Métricas escritas en %s", ruta)


# ---------------------------------------------------------------------------
# Módulo A — pronóstico
# ---------------------------------------------------------------------------
def construir_metricas_a() -> dict[str, Any]:
    """Comparación de modelos + mejor modelo + pronóstico final (del Módulo A)."""
    datos: dict[str, Any] = {
        "primary_metric": config.METRICA_PRIMARIA,
        "confidence_level": config.NIVEL_CONFIANZA,
        "best_model": None,
        "model_comparison": [],
        "forecast": [],
    }

    # Mejor modelo (artifacts/modelos/mejor_modelo.txt: "...: <nombre>").
    ruta_mejor = config.DIR_MODELOS / "mejor_modelo.txt"
    if ruta_mejor.exists():
        txt = ruta_mejor.read_text(encoding="utf-8").strip()
        datos["best_model"] = txt.split(":")[-1].strip() if ":" in txt else txt

    # Tabla de comparación (13 modelos x 7 métricas).
    if config.RUTA_METRICAS.exists():
        df = pd.read_csv(config.RUTA_METRICAS)
        datos["model_comparison"] = json.loads(
            df.to_json(orient="records", force_ascii=False)
        )
        # Fallback del mejor modelo: el de menor métrica primaria.
        if datos["best_model"] is None and config.METRICA_PRIMARIA in df.columns:
            datos["best_model"] = df.sort_values(config.METRICA_PRIMARIA).iloc[0]["Modelo"]

    # Pronóstico final con intervalo de confianza.
    if config.RUTA_PRONOSTICO.exists():
        dfp = pd.read_csv(config.RUTA_PRONOSTICO)
        datos["forecast"] = json.loads(dfp.to_json(orient="records", force_ascii=False))

    return datos


# ---------------------------------------------------------------------------
# Módulo B — distribuciones desde alertas.csv (autoritativas)
# ---------------------------------------------------------------------------
def _distribuciones_alertas() -> dict[str, Any]:
    if not config.RUTA_ALERTAS_CSV.exists():
        return {}
    df = pd.read_csv(config.RUTA_ALERTAS_CSV)

    out: dict[str, Any] = {}
    if "NIVEL" in df.columns:
        out["level_distribution"] = df["NIVEL"].value_counts().to_dict()
    if "TIPO_ANOMALIA" in df.columns:
        out["type_distribution"] = df["TIPO_ANOMALIA"].value_counts().to_dict()

    # Histograma del puntaje de riesgo (20 bins en [0, 1]).
    if "PUNTAJE_RIESGO" in df.columns:
        valores = df["PUNTAJE_RIESGO"].dropna().to_numpy()
        if valores.size:
            counts, edges = np.histogram(valores, bins=20, range=(0.0, 1.0))
            out["risk_score_histogram"] = {
                "bin_edges": [round(float(e), 4) for e in edges],
                "counts": [int(c) for c in counts],
            }

    # Estadísticos de las señales del modelo (IF y red).
    estad: dict[str, Any] = {}
    for col in ("SCORE_IF", "SCORE_RED", "PUNTAJE_RIESGO"):
        if col in df.columns:
            s = df[col].dropna()
            if len(s):
                estad[col] = {
                    "min": round(float(s.min()), 4),
                    "mediana": round(float(s.median()), 4),
                    "media": round(float(s.mean()), 4),
                    "max": round(float(s.max()), 4),
                }
    if estad:
        out["score_stats"] = estad

    out["alertas_en_csv"] = int(len(df))
    return out


# ---------------------------------------------------------------------------
# Módulo B — métricas de evaluación desde el informe Markdown
# ---------------------------------------------------------------------------
def _evaluacion_desde_informe() -> dict[str, Any]:
    """Lee precision@k, recall por inyección, concordancia y red del informe.

    El informe `informe_modulo_b.md` lo genera el propio Módulo B con un formato
    estable, por lo que se puede extraer de forma fiable con expresiones regulares.
    Si el informe no existe, devuelve un dict vacío (el exporter no falla).
    """
    ruta = config.RUTA_INFORME_B
    if not ruta.exists():
        return {}
    md = ruta.read_text(encoding="utf-8")
    out: dict[str, Any] = {}

    # Contexto general.
    out["n_ordenes"] = int(_num(_buscar(r"Órdenes válidas analizadas\**:\s*\**([\d,]+)", md)) or 0) or None
    out["n_entidades"] = int(_num(_buscar(r"\*\*Entidades\*\*:\s*([\d,]+)", md)) or 0) or None
    out["n_proveedores"] = int(_num(_buscar(r"\*\*Proveedores\*\*:\s*([\d,]+)", md)) or 0) or None
    out["umbral_fraccionamiento"] = _num(_buscar(r"Umbral de fraccionamiento.*?S/\s*([\d,]+)", md))
    out["tolerancia_igv"] = _num(_buscar(r"Tolerancia de IGV.*?±([\d.]+)", md))

    # precision@k: filas "| 20 | 100% |", acotadas a la sub-sección 5.1 para no
    # confundirlas con las dos primeras celdas de la tabla de inyección (5.2).
    m_sec = re.search(r"###\s*5\.1.*?(?=###\s*5\.2|\Z)", md, re.DOTALL)
    seccion_pak = m_sec.group(0) if m_sec else md
    pak = []
    for k, prec in re.findall(r"\|\s*(\d+)\s*\|\s*(\d+)%\s*\|", seccion_pak):
        pak.append({"k": int(k), "precision": round(int(prec) / 100.0, 4)})
    if pak:
        out["precision_at_k"] = pak

    # Recall por inyección: "| IGV inconsistente | 200 | 74% | 12% | 0.964 |".
    recall = []
    fila_iny = re.compile(
        r"\|\s*([A-Za-zÁÉÍÓÚáéíóúñ ]+?)\s*\|\s*(\d+)\s*\|\s*(\d+)%\s*\|\s*(\d+)%\s*\|\s*([\d.]+)\s*\|"
    )
    for tipo, n, ra, rt, med in fila_iny.findall(md):
        recall.append({
            "tipo": tipo.strip(),
            "n": int(n),
            "recall_alerta": round(int(ra) / 100.0, 4),
            "recall_top1pct": round(int(rt) / 100.0, 4),
            "puntaje_mediano": float(med),
        })
    if recall:
        out["synthetic_injection_recall"] = recall
    rg = re.search(r"Recall global.*?\*\*(\d+)%\*\*\s*de\s*([\d,]+)", md)
    if rg:
        out["recall_global"] = {
            "recall": round(int(rg.group(1)) / 100.0, 4),
            "n": int(_num(rg.group(2)) or 0),
        }

    # Concordancia (Jaccard) entre Isolation Forest (IF), Red y Reglas.
    j_if_reglas = _num(_buscar(r"Isolation Forest y las reglas es \*\*([\d.]+)\*\*", md))
    j_if_red = _num(_buscar(r"Isolation Forest y la red \*\*([\d.]+)\*\*", md))
    j_red_reglas = _num(_buscar(r"la red y las reglas (?:es )?\*\*([\d.]+)\*\*", md))
    if None not in (j_if_reglas, j_if_red, j_red_reglas):
        out["concordance"] = {
            "labels": ["Isolation Forest", "Red", "Reglas"],
            "jaccard": [
                [1.0, j_if_red, j_if_reglas],
                [j_if_red, 1.0, j_red_reglas],
                [j_if_reglas, j_red_reglas, 1.0],
            ],
            "interseccion_triple": int(_num(_buscar(r"\*\*(\d+) órdenes\*\* aparecen en el top de los \*\*tres\*\*", md)) or 0) or None,
            "top_1pct_size": int(_num(_buscar(r"top 1%\*\* de cada método \(([\d,]+)", md)) or 0) or None,
        }

    # Estadísticos de la red entidad–proveedor.
    nodos = _num(_buscar(r"\*\*([\d,]+) nodos\*\*", md))
    aristas = _num(_buscar(r"\*\*([\d,]+) aristas\*\*", md))
    if nodos and aristas:
        out["network_stats"] = {
            "nodos": int(nodos),
            "aristas": int(aristas),
            "comunidades": int(_num(_buscar(r"\*\*(\d+) comunidades\*\*", md)) or 0) or None,
            "proveedores_cautivos": int(_num(_buscar(r"\*\*([\d,]+) (?:proveedores cautivos|\nproveedores cautivos)", md)) or _num(_buscar(r"([\d,]+) proveedores cautivos", md)) or 0) or None,
            "grado_prov_max": int(_num(_buscar(r"sirve a (?:\*\*)?([\d,]+) entidades", md)) or 0) or None,
            "grado_ent_max": int(_num(_buscar(r"usa (?:\*\*)?([\d,]+) proveedores", md)) or 0) or None,
        }

    # Conteo de alertas por nivel (texto del informe).
    alto = _num(_buscar(r"\*\*([\d,]+) alertas de nivel alto\*\*", md))
    medio = _num(_buscar(r"\*\*([\d,]+) de nivel medio\*\*", md))
    if alto is not None and medio is not None:
        out["alertas"] = {"alto": int(alto), "medio": int(medio), "total": int(alto + medio)}

    return out


def construir_metricas_b() -> dict[str, Any]:
    """Combina distribuciones (de alertas.csv) y evaluación (del informe)."""
    datos: dict[str, Any] = {}
    datos.update(_evaluacion_desde_informe())
    datos.update(_distribuciones_alertas())  # las distribuciones priman (datos vivos)
    return datos


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------
def exportar_a() -> Path:
    """Genera solo el JSON de métricas del Módulo A."""
    config.asegurar_directorios()
    _escribir_json(RUTA_METRICAS_A, construir_metricas_a())
    return RUTA_METRICAS_A


def exportar_b() -> Path:
    """Genera solo el JSON de métricas del Módulo B."""
    config.asegurar_directorios()
    _escribir_json(RUTA_METRICAS_B, construir_metricas_b())
    return RUTA_METRICAS_B


def exportar_todo() -> tuple[Path, Path]:
    """Genera ambos JSON de métricas. Devuelve sus rutas."""
    return exportar_a(), exportar_b()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    a, b = exportar_todo()
    print(f"OK -> {a}")
    print(f"OK -> {b}")
