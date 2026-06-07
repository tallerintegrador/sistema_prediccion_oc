"""
ingesta.py
==========
Etapa de INGESTA / ETL: recorre `data/` de forma recursiva, lee todos los CSV
mensuales de PERÚ COMPRAS y los consolida en un único DataFrame.

Decisiones tomadas a partir de la exploración real de los datos:

* La codificación se detecta **por archivo** (52 archivos son `latin-1` y 1 es
  `utf-8`). Se prueba `utf-8` estricto primero: como falla en los archivos
  `latin-1` pero no en el `utf-8`, sirve de detector fiable.
* Los nombres de columna se **normalizan** (mayúsculas, sin tildes, espacios a
  "_") porque entre años aparecen pequeñas variaciones de codificación en la
  cabecera.
* Cualquier inconsistencia (archivo ilegible, columnas distintas, archivo vacío)
  se **registra** sin detener el proceso global.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

from .. import config

logger = logging.getLogger(__name__)


def detectar_codificacion(ruta: Path) -> str:
    """
    Detecta la codificación de un archivo probando, en orden, las codificaciones
    de `config.CODIFICACIONES`.

    Se lee el archivo COMPLETO en modo estricto: la primera codificación que lo
    decodifica sin lanzar `UnicodeDecodeError` se considera la correcta. Como
    `latin-1` nunca falla, debe ir al final de la lista (sirve de respaldo).

    Parámetros
    ----------
    ruta : Path
        Ruta al archivo CSV.

    Devuelve
    --------
    str
        Nombre de la codificación detectada.
    """
    for enc in config.CODIFICACIONES:
        try:
            with open(ruta, encoding=enc) as fh:
                fh.read()
            return enc
        except UnicodeDecodeError:
            continue
    # Respaldo final: latin-1 decodifica cualquier byte.
    return "latin-1"


def normalizar_nombre_columna(nombre: str) -> str:
    """
    Normaliza el nombre de una columna para unificar variaciones entre años.

    Pasos: quita acentos/tildes (NFKD -> ASCII), recorta espacios, pasa a
    MAYÚSCULAS y reemplaza espacios internos por guiones bajos.

    Ejemplos
    --------
    'FECHA_FORMALIZACIÓN' -> 'FECHA_FORMALIZACION'
    'Acuerdo Marco'       -> 'ACUERDO_MARCO'
    """
    sin_acentos = (
        unicodedata.normalize("NFKD", nombre)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    return re.sub(r"\s+", "_", sin_acentos.strip().upper())


def _mes_desde_nombre(nombre_archivo: str) -> str | None:
    """Extrae el 'AAAAMM' del nombre del archivo (p. ej. 'ReportePCBienes202201')."""
    m = re.search(config.PATRON_FECHA_ARCHIVO, nombre_archivo)
    return m.group(1) if m else None


def listar_archivos() -> list[Path]:
    """Devuelve, ordenados, todos los CSV bajo `data/` (búsqueda recursiva)."""
    archivos = sorted(config.DIR_DATOS.rglob("*.csv"))
    return archivos


def leer_archivo(ruta: Path) -> pd.DataFrame | None:
    """
    Lee un CSV individual de forma robusta.

    * Detecta la codificación.
    * Lee TODO como texto (`dtype=str`) para no perder ceros a la izquierda en
      RUC ni romper el parseo de montos/fechas, que se hará en la limpieza.
    * Normaliza los nombres de columna.
    * Añade columnas de trazabilidad: `__archivo` y `__mes_archivo`.

    Devuelve `None` (y registra el problema) si el archivo está vacío o es
    ilegible, para no detener la consolidación.
    """
    try:
        enc = detectar_codificacion(ruta)
        df = pd.read_csv(ruta, sep=config.SEPARADOR_CSV, encoding=enc, dtype=str)
    except pd.errors.EmptyDataError:
        logger.warning("Archivo vacío, se omite: %s", ruta.name)
        return None
    except Exception as exc:  # noqa: BLE001 - queremos continuar pese a cualquier error
        logger.error("No se pudo leer %s (%s): %s", ruta.name, type(exc).__name__, exc)
        return None

    if df.empty:
        logger.warning("Archivo sin filas, se omite: %s", ruta.name)
        return None

    df.columns = [normalizar_nombre_columna(c) for c in df.columns]
    df["__archivo"] = ruta.name
    df["__mes_archivo"] = _mes_desde_nombre(ruta.name)
    return df


def consolidar(guardar: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Recorre `data/`, lee y une todos los CSV en un único DataFrame.

    Maneja variaciones de columnas entre archivos: usa la UNIÓN de columnas y
    registra qué archivos difieren del conjunto de columnas más frecuente.

    Parámetros
    ----------
    guardar : bool
        Si True, guarda el consolidado en parquet (config.RUTA_CONSOLIDADO_PARQUET).

    Devuelve
    --------
    (df, reporte) : tuple[pd.DataFrame, dict]
        El DataFrame consolidado y un diccionario con el resumen de la ingesta
        (archivos leídos, meses, filas, inconsistencias de columnas, encodings).
    """
    config.asegurar_directorios()
    archivos = listar_archivos()
    logger.info("Archivos CSV encontrados bajo data/: %d", len(archivos))

    marcos: list[pd.DataFrame] = []
    encodings: dict[str, int] = {}
    columnas_por_archivo: dict[str, tuple[str, ...]] = {}
    leidos: list[str] = []
    omitidos: list[str] = []

    for ruta in archivos:
        enc = detectar_codificacion(ruta)
        encodings[enc] = encodings.get(enc, 0) + 1
        df = leer_archivo(ruta)
        if df is None:
            omitidos.append(ruta.name)
            continue
        # Guardamos las columnas "de negocio" (sin las internas) para comparar.
        cols_negocio = tuple(c for c in df.columns if not c.startswith("__"))
        columnas_por_archivo[ruta.name] = cols_negocio
        marcos.append(df)
        leidos.append(ruta.name)

    if not marcos:
        raise RuntimeError("No se pudo leer ningún archivo CSV de data/.")

    # Conjunto de columnas más frecuente -> referencia para detectar variaciones.
    from collections import Counter

    conteo_sets = Counter(columnas_por_archivo.values())
    columnas_referencia = conteo_sets.most_common(1)[0][0]
    archivos_distintos = {
        nombre: cols
        for nombre, cols in columnas_por_archivo.items()
        if cols != columnas_referencia
    }

    # `concat` alinea por nombre de columna; las faltantes quedan como NaN.
    df_consolidado = pd.concat(marcos, ignore_index=True, sort=False)

    meses = sorted({m for m in df_consolidado["__mes_archivo"].dropna().unique()})

    # Verificación (no filtrado) de la cobertura 2022→presente. Confirma de forma
    # trazable que la serie arranca en (o antes de) ANIO_INICIO_DATOS y que la cola
    # llega hasta hace pocos meses. La ingesta sigue leyendo TODO data/.
    cobertura = _verificar_cobertura(meses)
    if not cobertura["inicio_ok"]:
        logger.warning(
            "Cobertura: el primer mes (%s) es posterior a %s01; falta historia de %d.",
            cobertura["primer_mes"], config.ANIO_INICIO_DATOS, config.ANIO_INICIO_DATOS,
        )
    if not cobertura["fin_reciente"]:
        logger.warning(
            "Cobertura: el último mes (%s) está a %d meses de hoy; ¿faltan archivos recientes?",
            cobertura["ultimo_mes"], cobertura["meses_desde_ultimo"],
        )

    reporte = {
        "archivos_encontrados": len(archivos),
        "archivos_leidos": len(leidos),
        "archivos_omitidos": omitidos,
        "meses_cubiertos": meses,
        "n_meses": len(meses),
        "filas_totales": len(df_consolidado),
        "codificaciones": encodings,
        "n_conjuntos_columnas": len(conteo_sets),
        "columnas_referencia": list(columnas_referencia),
        "archivos_columnas_distintas": {
            n: list(c) for n, c in archivos_distintos.items()
        },
        "cobertura": cobertura,
    }

    if guardar:
        df_consolidado.to_parquet(config.RUTA_CONSOLIDADO_PARQUET, index=False)
        logger.info("Consolidado guardado en %s", config.RUTA_CONSOLIDADO_PARQUET)

    _escribir_log_ingesta(reporte)
    return df_consolidado, reporte


def _verificar_cobertura(meses: list[str]) -> dict:
    """
    Verifica (sin filtrar) que la serie cubra desde `config.ANIO_INICIO_DATOS`
    hasta un mes reciente. `meses` es la lista ordenada de 'AAAAMM' presentes.

    Devuelve un dict con el primer/último mes, banderas de cobertura y los meses
    transcurridos desde el último archivo. `cobertura_ok` resume ambas banderas.
    """
    from datetime import date

    if not meses:
        return {"cobertura_ok": False, "primer_mes": None, "ultimo_mes": None}

    primer, ultimo = meses[0], meses[-1]
    inicio_ok = primer <= f"{config.ANIO_INICIO_DATOS}01"

    hoy = date.today()
    anio_u, mes_u = int(ultimo[:4]), int(ultimo[4:6])
    meses_desde_ultimo = (hoy.year - anio_u) * 12 + (hoy.month - mes_u)
    # Tolerancia de 2 meses: el mes en curso y el previo pueden no estar cargados.
    fin_reciente = meses_desde_ultimo <= 2

    return {
        "primer_mes": primer,
        "ultimo_mes": ultimo,
        "anio_inicio_esperado": config.ANIO_INICIO_DATOS,
        "inicio_ok": inicio_ok,
        "fin_reciente": fin_reciente,
        "meses_desde_ultimo": meses_desde_ultimo,
        "cobertura_ok": inicio_ok and fin_reciente,
    }


def _escribir_log_ingesta(reporte: dict) -> None:
    """Vuelca el resumen de la ingesta a un archivo de texto legible."""
    cob = reporte.get("cobertura", {})
    lineas = [
        "==== LOG DE INGESTA (ETL) ====",
        f"Archivos encontrados : {reporte['archivos_encontrados']}",
        f"Archivos leídos      : {reporte['archivos_leidos']}",
        f"Archivos omitidos    : {reporte['archivos_omitidos'] or 'ninguno'}",
        f"Meses cubiertos      : {reporte['n_meses']} "
        f"({reporte['meses_cubiertos'][0]} a {reporte['meses_cubiertos'][-1]})",
        f"Cobertura 2022→hoy   : {'OK' if cob.get('cobertura_ok') else 'REVISAR'} "
        f"(inicio<= {cob.get('anio_inicio_esperado')}01: {cob.get('inicio_ok')}, "
        f"fin reciente: {cob.get('fin_reciente')}, "
        f"meses desde último: {cob.get('meses_desde_ultimo')})",
        f"Filas totales        : {reporte['filas_totales']:,}",
        f"Codificaciones       : {reporte['codificaciones']}",
        f"Conjuntos de columnas: {reporte['n_conjuntos_columnas']}",
    ]
    if reporte["archivos_columnas_distintas"]:
        lineas.append("Archivos con columnas distintas a la referencia:")
        for n, c in reporte["archivos_columnas_distintas"].items():
            lineas.append(f"  - {n}: {c}")
    else:
        lineas.append("Todos los archivos comparten el mismo conjunto de columnas.")

    config.RUTA_LOG_INGESTA.write_text("\n".join(lineas), encoding="utf-8")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    _df, _rep = consolidar()
    print(f"Consolidado: {_rep['filas_totales']:,} filas, {_rep['n_meses']} meses.")
