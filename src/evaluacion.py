"""
evaluacion.py
=============
Evaluación, comparación de modelos y pronóstico final.

* Métricas: MAE, RMSE y MAPE sobre el holdout.
* Tabla comparativa y selección automática del mejor modelo (menor RMSE).
* Pronóstico hacia adelante con el mejor modelo, con intervalo de confianza
  (nativo si el modelo lo provee; aproximado por residuos del holdout si no).
* Figuras: real vs. predicho en el holdout, e histórico vs. pronóstico final.
* Genera la sección de modelado del informe en Markdown.
"""

from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from . import config, modelos

logger = logging.getLogger(__name__)


def _millones(x, _pos=None) -> str:
    return f"{x / 1e6:,.1f}M"


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def calcular_metricas(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Calcula MAE, RMSE y MAPE entre valores reales y predichos.

    - MAE  : error absoluto medio (en soles).
    - RMSE : raíz del error cuadrático medio (penaliza errores grandes).
    - MAPE : error porcentual absoluto medio (%), interpretable sin unidades.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def comparar_modelos(predicciones: pd.DataFrame, test: pd.Series) -> pd.DataFrame:
    """
    Construye la tabla de métricas por modelo sobre el holdout, ordenada por RMSE.
    """
    filas = []
    for modelo in predicciones.columns:
        pred = predicciones[modelo]
        if pred.isna().all():
            filas.append({"Modelo": modelo, "MAE": np.nan, "RMSE": np.nan, "MAPE": np.nan})
            continue
        m = calcular_metricas(test.to_numpy(), pred.to_numpy())
        filas.append({"Modelo": modelo, **m})
    tabla = pd.DataFrame(filas).set_index("Modelo").sort_values("RMSE")
    return tabla


def seleccionar_mejor(tabla: pd.DataFrame) -> str:
    """Devuelve el nombre del modelo con menor RMSE (ignorando los que fallaron)."""
    validos = tabla.dropna(subset=["RMSE"])
    if validos.empty:
        raise RuntimeError("Ningún modelo produjo métricas válidas.")
    return str(validos["RMSE"].idxmin())


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def figura_holdout(train: pd.Series, test: pd.Series, predicciones: pd.DataFrame, mejor: str) -> str:
    """Grafica el histórico, los valores reales del holdout y las predicciones."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(train.index, train.values, color="#404040", label="Entrenamiento")
    ax.plot(test.index, test.values, color="black", marker="o", lw=2, label="Real (holdout)")
    for modelo in predicciones.columns:
        estilo = dict(lw=2.2) if modelo == mejor else dict(lw=1, ls="--", alpha=0.7)
        ax.plot(test.index, predicciones[modelo], marker=".", label=modelo, **estilo)
    ax.set_title("Comparación en el holdout: real vs. modelos")
    ax.set_ylabel("Gasto mensual (S/)")
    ax.set_xlabel("Mes")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_millones))
    ax.legend(fontsize=8, ncol=2)
    ruta = config.DIR_FIGURAS / "09_holdout_real_vs_modelos.png"
    fig.savefig(ruta)
    plt.close(fig)
    return "../figuras/09_holdout_real_vs_modelos.png"


def figura_pronostico(serie: pd.Series, pronostico: pd.DataFrame, mejor: str) -> str:
    """Grafica el histórico real y el pronóstico futuro con su intervalo."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(serie.index, serie.values, color="#1f4e79", label="Histórico real")
    # Conectar el último punto real con el pronóstico para continuidad visual.
    eje_x = [serie.index[-1]] + list(pronostico.index)
    eje_y = [serie.values[-1]] + list(pronostico["pred"])
    ax.plot(eje_x, eje_y, color="#c00000", marker="o", lw=2, label=f"Pronóstico ({mejor})")
    if pronostico[["lower", "upper"]].notna().all().all():
        ax.fill_between(
            pronostico.index, pronostico["lower"], pronostico["upper"],
            color="#c00000", alpha=0.15,
            label=f"IC {int(config.NIVEL_CONFIANZA*100)}%",
        )
    ax.set_title("Histórico vs. pronóstico del gasto mensual")
    ax.set_ylabel("Gasto mensual (S/)")
    ax.set_xlabel("Mes")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(_millones))
    ax.legend(fontsize=9)
    ruta = config.DIR_FIGURAS / "10_pronostico_final.png"
    fig.savefig(ruta)
    plt.close(fig)
    return "../figuras/10_pronostico_final.png"


# ---------------------------------------------------------------------------
# Intervalo aproximado para modelos sin IC nativo
# ---------------------------------------------------------------------------
def _completar_intervalo(pronostico: pd.DataFrame, rmse_holdout: float) -> pd.DataFrame:
    """
    Si el modelo no provee intervalo nativo, construye uno aproximado a partir
    del RMSE del holdout: pred ± z * RMSE, con z del nivel de confianza.
    """
    if pronostico[["lower", "upper"]].notna().all().all():
        return pronostico
    from scipy.stats import norm

    z = norm.ppf(1 - (1 - config.NIVEL_CONFIANZA) / 2)
    pronostico = pronostico.copy()
    pronostico["lower"] = pronostico["pred"] - z * rmse_holdout
    pronostico["upper"] = pronostico["pred"] + z * rmse_holdout
    pronostico["lower"] = pronostico["lower"].clip(lower=0)  # el gasto no es negativo
    return pronostico


# ---------------------------------------------------------------------------
# Orquestación de la evaluación + pronóstico final
# ---------------------------------------------------------------------------
def ejecutar_evaluacion(serie: pd.DataFrame) -> tuple[str, dict]:
    """
    Evalúa todos los modelos, selecciona el mejor, genera el pronóstico final y
    las figuras, y devuelve la sección del informe en Markdown más un dict de
    resultados.
    """
    config.asegurar_directorios()
    predicciones, test, detalles = modelos.evaluar_en_holdout(serie)
    tabla = comparar_modelos(predicciones, test)
    mejor = seleccionar_mejor(tabla)
    logger.info("Mejor modelo por RMSE: %s", mejor)

    train, _ = modelos.dividir_train_test(serie)

    # Figura del holdout.
    ruta_holdout = figura_holdout(train["gasto"], test, predicciones, mejor)

    # Pronóstico final con el mejor modelo reentrenado en toda la serie.
    pronostico = modelos.pronosticar_futuro(serie, mejor, config.HORIZONTE_PRONOSTICO)
    rmse_mejor = float(tabla.loc[mejor, "RMSE"])
    pronostico = _completar_intervalo(pronostico, rmse_mejor)
    pronostico.to_csv(config.RUTA_PRONOSTICO)

    ruta_pron = figura_pronostico(serie["gasto"], pronostico, mejor)

    # Guardar tabla de métricas.
    tabla.to_csv(config.RUTA_METRICAS)

    md = _seccion_informe(tabla, mejor, detalles, pronostico, ruta_holdout, ruta_pron, test, predicciones)
    resultados = {
        "tabla_metricas": tabla,
        "mejor_modelo": mejor,
        "pronostico": pronostico,
        "detalles": detalles,
    }
    return md, resultados


def _seccion_informe(tabla, mejor, detalles, pronostico, ruta_holdout, ruta_pron, test, predicciones) -> str:
    """Arma la sección de modelado y resultados del informe (Markdown)."""
    m = tabla.loc[mejor]
    gasto_prom_holdout = float(test.mean())
    error_relativo = m["MAE"] / gasto_prom_holdout * 100

    md = []
    md.append("\n## 3. Modelado del pronóstico\n")
    md.append(
        f"Se reservaron los **últimos {config.MESES_HOLDOUT} meses** como conjunto de prueba "
        f"(holdout) y se entrenó con los anteriores. Se compararon {len(tabla)} modelos: un "
        f"naive simple y un naive estacional (referencias), un modelo estadístico SARIMA, un "
        f"modelo de árboles XGBoost con variables temporales y una red neuronal LSTM. Todas las "
        f"semillas se fijaron en {config.SEMILLA} para reproducibilidad.\n"
    )
    if "SARIMA" in detalles:
        md.append(f"\nEl SARIMA seleccionado automáticamente por AIC fue **{detalles['SARIMA']}**.\n")

    md.append("\n### 3.1 Comparación de modelos (holdout)\n")
    md.append("\n| Modelo | MAE (S/) | RMSE (S/) | MAPE (%) |\n|---|---|---|---|\n")
    for nombre, fila in tabla.iterrows():
        mae = "—" if pd.isna(fila["MAE"]) else f"{fila['MAE']:,.0f}"
        rmse = "—" if pd.isna(fila["RMSE"]) else f"{fila['RMSE']:,.0f}"
        mape = "—" if pd.isna(fila["MAPE"]) else f"{fila['MAPE']:.1f}"
        marca = " ⭐" if nombre == mejor else ""
        md.append(f"| {nombre}{marca} | {mae} | {rmse} | {mape} |\n")

    md.append(f"\n![Comparación en el holdout]({ruta_holdout})\n")
    md.append(
        f"\n_El mejor modelo es **{mejor}** (menor RMSE). Sobre el holdout alcanza un MAPE de "
        f"**{m['MAPE']:.1f}%** y un MAE de **S/ {m['MAE']/1e6:,.2f} millones**, equivalente a un "
        f"**{error_relativo:.1f}%** del gasto mensual promedio del periodo de prueba "
        f"(S/ {gasto_prom_holdout/1e6:,.1f}M). En términos prácticos, el modelo anticipa el gasto "
        f"mensual con un error típico de ese orden, suficiente para planificación presupuestal "
        f"agregada aunque no para el detalle de una orden individual._\n"
    )

    md.append("\n### 3.2 Pronóstico hacia adelante\n")
    inicio = pronostico.index[0].strftime("%Y-%m")
    fin = pronostico.index[-1].strftime("%Y-%m")
    total_pron = pronostico["pred"].sum()
    md.append(
        f"Con **{mejor}** reentrenado sobre toda la serie se pronostican los próximos "
        f"**{config.HORIZONTE_PRONOSTICO} meses** ({inicio} a {fin}), con intervalo de confianza "
        f"al {int(config.NIVEL_CONFIANZA*100)}%.\n"
    )
    md.append("\n| Mes | Pronóstico (S/) | Límite inferior | Límite superior |\n|---|---|---|---|\n")
    for fecha, fila in pronostico.iterrows():
        md.append(
            f"| {fecha.strftime('%Y-%m')} | {fila['pred']:,.0f} | "
            f"{fila['lower']:,.0f} | {fila['upper']:,.0f} |\n"
        )
    md.append(f"\n![Pronóstico final]({ruta_pron})\n")
    md.append(
        f"\n_Se proyecta un gasto acumulado de **S/ {total_pron/1e6:,.0f} millones** en los "
        f"próximos {config.HORIZONTE_PRONOSTICO} meses. El pronóstico reproduce el patrón "
        f"estacional histórico (meses altos y bajos) y el intervalo de confianza refleja la "
        f"incertidumbre: cuanto más ancho, mayor variabilidad esperada. Conviene recalibrar el "
        f"modelo a medida que ingresen nuevos meses de datos._\n"
    )

    md.append("\n## 4. Conclusiones del Módulo A\n")
    md.append(
        f"- El gasto en órdenes de compra de los Acuerdos Marco muestra una **estacionalidad "
        f"anual fuerte** (mínimos en enero–febrero) y una tendencia de fondo identificable.\n"
        f"- Entre los modelos probados, **{mejor}** ofreció el mejor equilibrio en el holdout "
        f"(MAPE {m['MAPE']:.1f}%).\n"
        f"- La principal limitación es la **longitud de la serie** (pocos años de historia "
        f"mensual), que restringe especialmente a la LSTM; con más historia el desempeño podría "
        f"mejorar.\n"
        f"- El pipeline es **reproducible** (semillas fijas, rutas relativas) y deja listos los "
        f"artefactos para los siguientes módulos del sistema.\n"
    )
    return "".join(md)
