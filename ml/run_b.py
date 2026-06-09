"""
main_modulo_b.py
================
Flujo completo del **Módulo B** (detección de órdenes de compra con comportamiento
anómalo o de riesgo).

Ejecuta de punta a punta:
    cargar dataset limpio (reutiliza el Módulo A)
        -> ingeniería de features de anomalía
        -> EDA con interpretaciones
        -> modelos (Isolation Forest + autoencoder + grafo)
        -> puntaje de riesgo y ranking de alertas
        -> evaluación (precision@k, inyección sintética, concordancia)
        -> exportación de alertas e informe en Markdown

y genera todos los artefactos del B en `artifacts/` (figuras `b*.png`, `alertas.csv`,
`informe_modulo_b.md`).

Es un módulo HERMANO e INDEPENDIENTE del Módulo A: solo comparte la capa de datos.

Uso (desde la raíz del proyecto):
    python -m ml.run_b
    python -m ml.run_b --rapido     # salta la inyección sintética (más veloz)
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime

from ml.shared import config as cb
from ml.metrics_export import exportar_b
from ml.modulo_b_anomalias import (
    eda_anomalias as eda,
    evaluacion_anomalias as ev,
    features_anomalias as fa,
    modelos_anomalias as ma,
    riesgo as rg,
)


def configurar_logging() -> None:
    """Configura el logging a consola con formato legible."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def construir_encabezado(df, meta: dict) -> str:
    """Genera la cabecera e introducción del informe del Módulo B en Markdown."""
    f = df[cb.COL_FECHA_FORM]
    umbral = meta["umbral_fraccionamiento"]["umbral"]
    tol = meta["tolerancia_igv"]["tolerancia"]
    return (
        "# Informe de Resultados — Módulo B: Detección de Órdenes de Compra Anómalas\n\n"
        f"_Sistema de Predicción de Órdenes de Compra (SistemaPrediccionOC). "
        f"Generado el {datetime.now():%Y-%m-%d %H:%M}._\n\n"
        "## 1. Introducción y datos\n\n"
        "Este informe documenta el **Módulo B** del sistema: la detección, de forma **no "
        "supervisada**, de órdenes de compra con comportamiento anómalo o de riesgo, para "
        "apoyar la transparencia y la revisión de las compras públicas de los Catálogos "
        "Electrónicos de Acuerdos Marco de PERÚ COMPRAS.\n\n"
        "El Módulo B **reutiliza la capa de datos del Módulo A** (el dataset consolidado y "
        "limpio); no vuelve a leer ni a limpiar los CSV. Cada fila es **una orden completa** "
        "(no el detalle por producto), por lo que las anomalías se detectan a nivel de orden "
        "y de relaciones entre entidades y proveedores.\n\n"
        f"- **Órdenes válidas analizadas**: {len(df):,}\n"
        f"- **Rango temporal** (FECHA_FORMALIZACIÓN): {f.min():%Y-%m-%d} a {f.max():%Y-%m-%d}\n"
        f"- **Entidades**: {df[cb.COL_RUC_ENTIDAD].nunique():,} · "
        f"**Proveedores**: {df[cb.COL_RUC_PROVEEDOR].nunique():,}\n\n"
        "**Parámetros derivados de los propios datos** (no asumidos):\n\n"
        f"- **Umbral de fraccionamiento** (cambio a *Gran Compra*): **S/ {umbral:,.0f}**, "
        f"descubierto por la discontinuidad de P(Gran Compra | TOTAL).\n"
        f"- **Tolerancia de IGV**: **±{tol:.4f}** sobre el ratio 0.18, derivada de la "
        f"distribución real del ratio IGV/SUB_TOTAL.\n"
    )


def construir_seccion_features(meta: dict) -> str:
    """Documenta las variables de anomalía construidas."""
    return (
        "## 3. Ingeniería de variables de anomalía\n\n"
        "Se construyeron variables por orden que capturan cada comportamiento de riesgo. "
        "Todos los parámetros se derivan de los datos.\n\n"
        "| Variable | Comportamiento que captura |\n|---|---|\n"
        "| `LOG_TOTAL` | Monto de la orden (en log, por la cola larga). |\n"
        "| `Z_TOTAL_CAT`, `Z_TOTAL_GLOBAL` | Desviación robusta del log-monto vs su categoría y vs el global (mediana/MAD). |\n"
        "| `ABS_DESV_IGV`, `FLAG_IGV` | Distancia del ratio IGV/SUB_TOTAL respecto a 0.18. |\n"
        "| `SHARE_GASTO_PROV`, `SHARE_ORD_PROV` | Proporción del gasto/órdenes de la entidad dirigida a este proveedor. |\n"
        "| `HHI_ENTIDAD` | Concentración (Herfindahl) de proveedores de la entidad. |\n"
        "| `PROXIMIDAD_UMBRAL`, `FLAG_BAJO_UMBRAL` | Cercanía del monto al umbral de fraccionamiento, por debajo. |\n"
        "| `N_ORD_VENTANA_7D/30D`, `SUMA_VENTANA_*`, `FLAG_FRACCIONAMIENTO` | Ráfaga de órdenes del par entidad–proveedor en ventanas cortas. |\n"
        "| `N_ORD_ENTIDAD_DIA` | Órdenes de la entidad el mismo día. |\n"
        "| `DIAS_PARA_FIN_MES`, `FLAG_CIERRE_MES` | Proximidad al cierre de mes. |\n"
        "| `DIAS_ESTADO` | Días entre FECHA_FORMALIZACIÓN y FECHA_ÚLTIMO_ESTADO. |\n"
        "| `LOCK_IN`, `DEP_PROV_ENTIDAD`, `FLAG_PROV_CAUTIVO`, `TAM_COMUNIDAD` | Señales del grafo entidad–proveedor. |\n"
    )


def construir_seccion_red(info_modelos: dict) -> str:
    """Resume el análisis de grafo y lista las relaciones más atípicas."""
    r = info_modelos["red"]
    top = r["top_relaciones"].head(8)[
        ["RUC_ENTIDAD", "RUC_PROVEEDOR", "gasto", "n", "s_ent_prov", "s_prov_ent", "lock_in"]
    ].copy()
    top["gasto"] = top["gasto"].map(lambda v: f"{v:,.0f}")
    for c in ("s_ent_prov", "s_prov_ent", "lock_in"):
        top[c] = top[c].map(lambda v: f"{v:.2f}")
    tabla = ev._df_a_markdown(top)
    return (
        "### 4.1 Análisis de la red entidad–proveedor\n\n"
        f"El grafo bipartito tiene **{r['n_nodos']:,} nodos** y **{r['n_aristas']:,} aristas**, "
        f"con **{r['n_comunidades']} comunidades** (Louvain) y **{r['n_prov_cautivos']:,} "
        f"proveedores cautivos** (atienden a una sola entidad). El proveedor más conectado "
        f"sirve a {r['grado_prov_max']:,} entidades y la entidad más diversificada usa "
        f"{r['grado_ent_max']:,} proveedores.\n\n"
        "Las relaciones con mayor **candado bilateral** (lock-in) — entidad y proveedor que "
        "dependen casi exclusivamente el uno del otro — son las más atípicas:\n\n"
        + tabla + "\n"
    )


def main(con_inyeccion: bool = True) -> None:
    """Ejecuta el flujo completo del Módulo B."""
    configurar_logging()
    log = logging.getLogger("main_b")
    cb.DIR_RESULTADOS.mkdir(parents=True, exist_ok=True)
    cb.DIR_FIGURAS.mkdir(parents=True, exist_ok=True)
    ma.fijar_semillas()

    # 1) CARGA (reutiliza el Módulo A) -------------------------------------
    log.info("== 1/6 Carga del dataset limpio (Módulo A) ==")
    df = fa.cargar_dataset_limpio()

    # 2) FEATURES ----------------------------------------------------------
    log.info("== 2/6 Ingeniería de variables de anomalía ==")
    feats, meta = fa.construir_features(df)

    # 3) EDA ---------------------------------------------------------------
    log.info("== 3/6 EDA orientado a anomalías ==")
    md_eda = eda.ejecutar_eda(feats, df, meta)

    # 4) MODELOS -----------------------------------------------------------
    log.info("== 4/6 Modelos de detección (Isolation Forest + autoencoder + grafo) ==")
    scores, info_modelos = ma.ejecutar_modelos(feats, df)

    # 5) RIESGO Y RANKING --------------------------------------------------
    log.info("== 5/6 Puntaje de riesgo y ranking de alertas ==")
    tabla, info_riesgo = rg.calcular_riesgo(df, feats, scores, meta)
    alertas = rg.exportar_alertas(tabla, solo_alertas=True)

    # 6) EVALUACIÓN --------------------------------------------------------
    log.info("== 6/6 Evaluación (precision@k, inyección sintética, concordancia) ==")
    md_eval = ev.ejecutar_evaluacion(
        tabla, info_riesgo, df, meta["umbral_usado"], con_inyeccion=con_inyeccion
    )

    # INFORME FINAL --------------------------------------------------------
    informe = "\n".join([
        construir_encabezado(df, meta),
        md_eda,
        construir_seccion_features(meta),
        "## 4. Modelado de la detección\n\n"
        "Se aplicaron tres enfoques no supervisados y reproducibles (semilla fija): "
        "**Isolation Forest** y un **autoencoder** sobre las features estandarizadas, y un "
        "**análisis de grafo** de la red entidad–proveedor. Sus señales se combinan con las "
        "reglas interpretables en un puntaje de riesgo por orden.\n",
        construir_seccion_red(info_modelos),
        md_eval,
        "## 6. Salida de alertas\n\n"
        f"La tabla de alertas rankeada se exportó a `outputs/resultados/alertas.csv` "
        f"(**{len(alertas):,} alertas** de nivel medio/alto), con el id de la orden, "
        f"entidad, proveedor, tipo de anomalía, puntaje, nivel y motivo.\n",
    ])
    cb.RUTA_INFORME_B.write_text(informe, encoding="utf-8")
    log.info("Informe del Módulo B escrito en %s", cb.RUTA_INFORME_B)

    # Consolida las métricas del Módulo B en JSON para el API/frontend.
    exportar_b()

    # RESUMEN EN CONSOLA ---------------------------------------------------
    n = info_riesgo["conteo_nivel"]
    print("\n" + "=" * 70)
    print("MÓDULO B COMPLETADO".center(70))
    print("=" * 70)
    print(f"Órdenes analizadas : {len(df):,}")
    print(f"Umbral fraccionam. : S/ {meta['umbral_usado']:,.0f} (derivado de los datos)")
    print(f"Alertas            : {n.get('alto', 0):,} alto · {n.get('medio', 0):,} medio")
    print(f"Tipos predominantes: {info_riesgo['conteo_tipo']}")
    print(f"\nAlertas  : {cb.RUTA_ALERTAS_CSV}")
    print(f"Informe  : {cb.RUTA_INFORME_B}")
    print(f"Figuras  : {cb.DIR_FIGURAS} (b01..b08)")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Módulo B — detección de anomalías")
    parser.add_argument(
        "--rapido", action="store_true",
        help="Salta la inyección de anomalías sintéticas (evaluación más rápida).",
    )
    args = parser.parse_args()
    main(con_inyeccion=not args.rapido)
