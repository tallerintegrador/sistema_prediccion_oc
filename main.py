"""
main.py
=======
Flujo completo del Módulo A (pronóstico del gasto público en órdenes de compra).

Ejecuta de punta a punta:
    ingesta -> limpieza -> serie temporal -> EDA -> modelos -> evaluación

y genera todos los artefactos en `outputs/` (dataset consolidado y limpio,
figuras, tabla de métricas, pronóstico e informe en Markdown).

Uso:
    python main.py
"""

from __future__ import annotations

import logging
from datetime import datetime

from src import config, ingesta, limpieza, serie_temporal, eda, evaluacion, modelos


def configurar_logging() -> None:
    """Configura el logging a consola con formato legible."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def construir_encabezado_informe(reporte_ingesta: dict) -> str:
    """Genera la cabecera e introducción del informe en Markdown."""
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
        "La variable a pronosticar es el **gasto total mensual** (suma de `TOTAL`), agregado por "
        "**`FECHA_FORMALIZACION`**. El informe reúne el análisis exploratorio con sus "
        "interpretaciones y la comparación de modelos con su conclusión.\n\n"
        f"Los datos abarcan **{reporte_ingesta['n_meses']} meses** (de {rng[0]} a {rng[-1]}) e "
        f"incluyen **{reporte_ingesta['filas_totales']:,} órdenes** consolidadas desde "
        f"{reporte_ingesta['archivos_leidos']} archivos CSV mensuales.\n"
    )


def main() -> None:
    """Ejecuta el flujo completo del Módulo A."""
    configurar_logging()
    log = logging.getLogger("main")
    config.asegurar_directorios()
    modelos.fijar_semillas()

    # 1) INGESTA / ETL ------------------------------------------------------
    log.info("== 1/6 Ingesta y consolidación ==")
    df_crudo, rep_ingesta = ingesta.consolidar(guardar=True)
    log.info(
        "Consolidado: %s filas, %s meses (%s a %s).",
        f"{rep_ingesta['filas_totales']:,}", rep_ingesta["n_meses"],
        rep_ingesta["meses_cubiertos"][0], rep_ingesta["meses_cubiertos"][-1],
    )

    # 2) LIMPIEZA Y VALIDACIÓN ---------------------------------------------
    log.info("== 2/6 Limpieza y validación ==")
    df_limpio, rep_limpieza = limpieza.limpiar(df_crudo, guardar=True)
    log.info("Filas válidas tras limpieza: %s", f"{rep_limpieza['filas_finales']:,}")

    # 3) SERIE TEMPORAL -----------------------------------------------------
    log.info("== 3/6 Construcción de la serie temporal ==")
    serie_completa = serie_temporal.construir_serie_total(df_limpio)
    deteccion = serie_temporal.detectar_meses_incompletos(serie_completa)
    serie_total, serie_categoria, _ = serie_temporal.construir_series(df_limpio, guardar=True)
    log.info(
        "Serie mensual: %d meses (tras recorte de incompletos: %s).",
        len(serie_total),
        deteccion["meses_incompletos"] or "ninguno",
    )

    # 4) EDA ----------------------------------------------------------------
    log.info("== 4/6 Análisis exploratorio (EDA) ==")
    md_eda = eda.ejecutar_eda(df_limpio, serie_completa, deteccion, rep_ingesta, rep_limpieza)

    # 5-6) MODELADO Y EVALUACIÓN -------------------------------------------
    log.info("== 5/6 Modelado y 6/6 Evaluación ==")
    md_modelos, resultados = evaluacion.ejecutar_evaluacion(serie_total, df_limpio)

    # INFORME FINAL ---------------------------------------------------------
    informe = (
        construir_encabezado_informe(rep_ingesta) + "\n" + md_eda + "\n" + md_modelos
    )
    config.RUTA_INFORME_EDA.write_text(informe, encoding="utf-8")
    log.info("Informe escrito en %s", config.RUTA_INFORME_EDA)

    # RESUMEN EN CONSOLA ----------------------------------------------------
    print("\n" + "=" * 70)
    print("MÓDULO A COMPLETADO".center(70))
    print("=" * 70)
    print(f"Mejor modelo: {resultados['mejor_modelo']}")
    print("\nMétricas en el holdout:")
    print(resultados["tabla_metricas"].round(2).to_string())
    print(f"\nArtefactos en: {config.DIR_OUTPUTS}")
    print(f"Informe: {config.RUTA_INFORME_EDA}")
    print("=" * 70)


if __name__ == "__main__":
    main()
