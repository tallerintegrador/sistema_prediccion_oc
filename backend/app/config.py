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

from pathlib import Path

# backend/app/config.py  ->  parents[2] = raíz del proyecto SistemaPrediccionOC
RAIZ_PROYECTO: Path = Path(__file__).resolve().parents[2]

DIR_OUTPUTS: Path = RAIZ_PROYECTO / "outputs"
DIR_RESULTADOS: Path = DIR_OUTPUTS / "resultados"
DIR_MODELOS: Path = DIR_OUTPUTS / "modelos"

# --- Resultados del Módulo A (pronóstico) ---------------------------------
RUTA_SERIE_TOTAL: Path = DIR_RESULTADOS / "serie_mensual_total.csv"
RUTA_SERIE_CATEGORIA: Path = DIR_RESULTADOS / "serie_mensual_por_categoria.csv"
RUTA_PRONOSTICO: Path = DIR_RESULTADOS / "pronostico_final.csv"
RUTA_MEJOR_MODELO: Path = DIR_MODELOS / "mejor_modelo.txt"

# --- Resultados del Módulo B (anomalías) ----------------------------------
RUTA_ALERTAS: Path = DIR_RESULTADOS / "alertas.csv"

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
VERSION_API: str = "1.0.0"
