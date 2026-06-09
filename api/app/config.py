"""
config.py
=========
Configuración central del BACKEND (API REST).

Aquí se resuelven TODAS las rutas a los resultados ya calculados por los
Módulos A (pronóstico) y B (detección de anomalías). El backend SOLO LEE estos
archivos: no reentrena modelos ni reprocesa los CSV crudos de `data/`.

Las rutas se calculan a partir de la ubicación de este archivo, de modo que el
API encuentra los datos sin importar desde qué carpeta se levante uvicorn.
"""

from __future__ import annotations

import os
from pathlib import Path

# api/app/config.py  ->  parents[2] = raíz del proyecto SistemaPrediccionOC
RAIZ_PROYECTO: Path = Path(__file__).resolve().parents[2]

DIR_OUTPUTS: Path = RAIZ_PROYECTO / "artifacts"
DIR_RESULTADOS: Path = DIR_OUTPUTS / "resultados"
DIR_MODELOS: Path = DIR_OUTPUTS / "modelos"
DIR_FIGURAS: Path = DIR_OUTPUTS / "figuras"   # PNGs servidos en /figures

# --- Resultados del Módulo A (pronóstico) ---------------------------------
RUTA_SERIE_TOTAL: Path = DIR_RESULTADOS / "serie_mensual_total.csv"
RUTA_SERIE_CATEGORIA: Path = DIR_RESULTADOS / "serie_mensual_por_categoria.csv"
RUTA_PRONOSTICO: Path = DIR_RESULTADOS / "pronostico_final.csv"
RUTA_MEJOR_MODELO: Path = DIR_MODELOS / "mejor_modelo.txt"

# --- Resultados del Módulo B (anomalías) ----------------------------------
RUTA_ALERTAS: Path = DIR_RESULTADOS / "alertas.csv"

# --- Métricas del ML en JSON (las genera ml/metrics_export.py) ------------
RUTA_METRICAS_A: Path = DIR_RESULTADOS / "metrics_modulo_a.json"
RUTA_METRICAS_B: Path = DIR_RESULTADOS / "metrics_modulo_b.json"

# Nivel de confianza de los intervalos del pronóstico (parámetro del Módulo A,
# NIVEL_CONFIANZA en src/config.py). Se documenta aquí para exponerlo en el API.
NIVEL_CONFIANZA: float = 0.95

# Mapeo de niveles del API (inglés, como pide el contrato de endpoints) a los
# valores reales que el Módulo B escribió en alertas.csv (español).
MAPEO_NIVEL = {
    "high": "alto",
    "medium": "medio",
    # alias en español por robustez
    "alto": "alto",
    "medio": "medio",
}

# Metadatos del servicio.
NOMBRE_SERVICIO: str = "SistemaPrediccionOC API"
VERSION_API: str = "2.0.0"

# Orígenes permitidos para CORS. Se leen de la variable de entorno
# ALLOWED_ORIGINS (lista separada por comas). Por defecto, los puertos típicos
# del frontend en desarrollo (Vite). Usar "*" solo si se desea abrir del todo.
_origenes_env = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
ORIGENES_CORS: list[str] = [o.strip() for o in _origenes_env.split(",") if o.strip()]
