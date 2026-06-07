"""
modelos.py
==========
Modelos de pronóstico para la serie mensual del gasto total.

Se implementan y comparan cuatro enfoques:

1. **Naive estacional** (lag-12) y naive simple — líneas base de referencia.
2. **SARIMA** — modelo estadístico con estacionalidad (statsmodels).
3. **XGBoost** — árboles de gradiente con variables derivadas del tiempo
   (rezagos, medias móviles, mes, índice de tendencia). Pronóstico recursivo.
4. **LSTM** — red neuronal recurrente (PyTorch) sobre ventanas de la serie.

Convención común: cada modelo es una función `f(historia, n_periodos)` que recibe
la serie de entrenamiento (pd.Series de 'gasto' indexada por fecha de fin de mes)
y devuelve un dict con `pred` (np.ndarray de tamaño n) y, si aplica, `lower`/
`upper` (intervalo de confianza). Esto permite usar la MISMA función tanto para
evaluar en el holdout como para el pronóstico final (reentrenando con todo).

Todas las fuentes de aleatoriedad usan `config.SEMILLA` para reproducibilidad.
"""

from __future__ import annotations

import logging
import random
import warnings

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def fijar_semillas(semilla: int = config.SEMILLA) -> None:
    """Fija las semillas de random, numpy y torch para reproducibilidad."""
    random.seed(semilla)
    np.random.seed(semilla)
    try:
        import torch

        torch.manual_seed(semilla)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # noqa: BLE001 - torch puede no estar disponible
        pass


def dividir_train_test(serie: pd.DataFrame, meses_holdout: int = config.MESES_HOLDOUT):
    """Separa la serie en entrenamiento y prueba (últimos `meses_holdout` meses)."""
    train = serie.iloc[:-meses_holdout]
    test = serie.iloc[-meses_holdout:]
    return train, test


def _indice_futuro(historia: pd.Series, n: int) -> pd.DatetimeIndex:
    """Genera n fechas de fin de mes siguientes a la última de `historia`."""
    return pd.date_range(historia.index[-1], periods=n + 1, freq="ME")[1:]


# ---------------------------------------------------------------------------
# 1) Modelos base (naive)
# ---------------------------------------------------------------------------
def pronostico_naive(historia: pd.Series, n: int) -> dict:
    """Naive simple: repite el último valor observado."""
    ultimo = float(historia.iloc[-1])
    return {"pred": np.full(n, ultimo), "lower": None, "upper": None}


def pronostico_naive_estacional(historia: pd.Series, n: int) -> dict:
    """
    Naive estacional: el pronóstico del mes futuro es el valor del mismo mes del
    año anterior (lag-12). Es una referencia exigente dada la fuerte
    estacionalidad anual de esta serie.
    """
    s = config.PERIODO_ESTACIONAL
    ultima_temporada = historia.iloc[-s:].to_numpy()
    pred = np.array([ultima_temporada[i % s] for i in range(n)])
    return {"pred": pred, "lower": None, "upper": None}


# ---------------------------------------------------------------------------
# 2) SARIMA (statsmodels) — trabaja en escala logarítmica
# ---------------------------------------------------------------------------
def _seleccionar_sarima(serie_log: pd.Series):
    """
    Selecciona el mejor orden SARIMA por AIC dentro de una rejilla pequeña
    (adecuada a una serie corta de ~50 puntos). Devuelve el modelo ajustado.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    ordenes = [(1, 1, 1), (0, 1, 1), (1, 1, 0), (2, 1, 1), (1, 0, 0)]
    estacionales = [(0, 1, 1, 12), (1, 1, 0, 12), (0, 1, 0, 12), (1, 1, 1, 12)]
    mejor = None
    mejor_aic = np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for o in ordenes:
            for so in estacionales:
                try:
                    mod = SARIMAX(
                        serie_log, order=o, seasonal_order=so,
                        enforce_stationarity=False, enforce_invertibility=False,
                    ).fit(disp=False)
                    if np.isfinite(mod.aic) and mod.aic < mejor_aic:
                        mejor_aic, mejor = mod.aic, mod
                        mejor.orden_, mejor.estacional_ = o, so
                except Exception:  # noqa: BLE001
                    continue
    if mejor is None:
        raise RuntimeError("No se pudo ajustar ningún modelo SARIMA.")
    logger.info("SARIMA elegido: %s x %s (AIC=%.1f)", mejor.orden_, mejor.estacional_, mejor_aic)
    return mejor


def pronostico_sarima(historia: pd.Series, n: int) -> dict:
    """
    Ajusta un SARIMA sobre log(gasto) y pronostica n meses con intervalo de
    confianza. Los resultados se devuelven en la escala original (soles).
    """
    serie_log = np.log(historia.astype(float))
    serie_log.index = pd.DatetimeIndex(serie_log.index).to_period("M")
    modelo = _seleccionar_sarima(serie_log)
    pred = modelo.get_forecast(steps=n)
    media = np.exp(pred.predicted_mean.to_numpy())
    ic = pred.conf_int(alpha=1 - config.NIVEL_CONFIANZA)
    lower = np.exp(ic.iloc[:, 0].to_numpy())
    upper = np.exp(ic.iloc[:, 1].to_numpy())
    return {
        "pred": media, "lower": lower, "upper": upper,
        "detalle": f"SARIMA{modelo.orden_}x{modelo.estacional_}",
        "modelo": modelo,
    }


# ---------------------------------------------------------------------------
# 3) XGBoost con variables temporales — pronóstico recursivo
# ---------------------------------------------------------------------------
def _features_desde_historia(serie: pd.Series, fecha) -> dict:
    """
    Construye el vector de variables para predecir el valor en `fecha` a partir
    de la `serie` histórica disponible hasta justo antes de esa fecha.

    Variables: rezagos 1, 2, 3 y 12; medias móviles de 3 y 12; mes (seno/coseno
    para capturar ciclicidad) e índice de tendencia.
    """
    valores = serie.to_numpy(dtype=float)
    n = len(valores)
    mes = fecha.month
    return {
        "lag1": valores[-1],
        "lag2": valores[-2] if n >= 2 else valores[-1],
        "lag3": valores[-3] if n >= 3 else valores[-1],
        "lag12": valores[-12] if n >= 12 else valores[-1],
        "media_3": valores[-3:].mean(),
        "media_12": valores[-12:].mean() if n >= 12 else valores.mean(),
        "mes_sin": np.sin(2 * np.pi * mes / 12),
        "mes_cos": np.cos(2 * np.pi * mes / 12),
        "tendencia": n,
    }


def _tabla_supervisada(serie: pd.Series, min_historia: int = 12) -> pd.DataFrame:
    """Genera la tabla (X, y) de aprendizaje supervisado a partir de la serie."""
    filas = []
    fechas = serie.index
    for i in range(min_historia, len(serie)):
        hist = serie.iloc[:i]
        feats = _features_desde_historia(hist, fechas[i])
        feats["y"] = float(serie.iloc[i])
        filas.append(feats)
    return pd.DataFrame(filas)


def pronostico_xgboost(historia: pd.Series, n: int) -> dict:
    """
    Entrena un XGBoost sobre variables temporales y pronostica n meses de forma
    recursiva (cada predicción se reincorpora a la historia para la siguiente).
    """
    from xgboost import XGBRegressor

    fijar_semillas()
    tabla = _tabla_supervisada(historia)
    columnas_x = [c for c in tabla.columns if c != "y"]
    X, y = tabla[columnas_x], tabla["y"]

    modelo = XGBRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=config.SEMILLA, n_jobs=1,
    )
    modelo.fit(X, y)

    hist = historia.copy()
    fechas_fut = _indice_futuro(historia, n)
    preds = []
    for fecha in fechas_fut:
        feats = _features_desde_historia(hist, fecha)
        x = pd.DataFrame([feats])[columnas_x]
        yhat = float(modelo.predict(x)[0])
        preds.append(yhat)
        hist = pd.concat([hist, pd.Series([yhat], index=[fecha])])
    return {"pred": np.array(preds), "lower": None, "upper": None, "modelo": modelo}


# ---------------------------------------------------------------------------
# 4) LSTM (PyTorch) — sobre log-serie escalada, pronóstico recursivo
# ---------------------------------------------------------------------------
def pronostico_lstm(historia: pd.Series, n: int, ventana: int = 12, epocas: int = 400) -> dict:
    """
    Entrena una pequeña LSTM sobre ventanas de la serie (en escala log y
    normalizada) y pronostica n meses de forma recursiva.

    Nota: la serie es corta (~50 puntos), por lo que la LSTM está limitada por
    los datos; se reporta su desempeño con honestidad en la comparación.
    """
    import torch
    import torch.nn as nn

    fijar_semillas()

    # Transformación: log para estabilizar varianza + estandarización.
    serie_log = np.log(historia.to_numpy(dtype=float))
    mu, sigma = serie_log.mean(), serie_log.std() + 1e-8
    z = (serie_log - mu) / sigma

    if len(z) <= ventana + 2:
        ventana = max(3, len(z) // 2)

    # Construcción de ventanas (X: ventana pasos -> y: paso siguiente).
    X_list, y_list = [], []
    for i in range(ventana, len(z)):
        X_list.append(z[i - ventana:i])
        y_list.append(z[i])
    X = torch.tensor(np.array(X_list), dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(-1)

    class RedLSTM(nn.Module):
        """LSTM mínima de una capa con salida densa."""

        def __init__(self, oculto: int = 24):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=oculto, batch_first=True)
            self.fc = nn.Linear(oculto, 1)

        def forward(self, x):
            salida, _ = self.lstm(x)
            return self.fc(salida[:, -1, :])

    modelo = RedLSTM()
    opt = torch.optim.Adam(modelo.parameters(), lr=0.01)
    perdida = nn.MSELoss()
    modelo.train()
    for _ in range(epocas):
        opt.zero_grad()
        salida = modelo(X)
        error = perdida(salida, y)
        error.backward()
        opt.step()

    # Pronóstico recursivo.
    modelo.eval()
    ventana_actual = list(z[-ventana:])
    preds_z = []
    with torch.no_grad():
        for _ in range(n):
            entrada = torch.tensor(
                np.array(ventana_actual[-ventana:]), dtype=torch.float32
            ).reshape(1, ventana, 1)
            yhat = float(modelo(entrada).item())
            preds_z.append(yhat)
            ventana_actual.append(yhat)

    preds = np.exp(np.array(preds_z) * sigma + mu)  # deshacer estandarización y log
    return {"pred": preds, "lower": None, "upper": None}


# ---------------------------------------------------------------------------
# Registro de modelos disponibles
# ---------------------------------------------------------------------------
MODELOS = {
    "Naive": pronostico_naive,
    "Naive estacional": pronostico_naive_estacional,
    "SARIMA": pronostico_sarima,
    "XGBoost": pronostico_xgboost,
    "LSTM": pronostico_lstm,
}


def evaluar_en_holdout(serie: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    """
    Entrena cada modelo con el tramo de entrenamiento y predice el holdout.

    Devuelve
    --------
    (predicciones, test, detalles)
        predicciones : DataFrame con una columna por modelo, indexado por las
                       fechas del holdout.
        test         : serie real del holdout (gasto).
        detalles     : dict con metadatos (p. ej. orden SARIMA elegido).
    """
    fijar_semillas()
    train, test = dividir_train_test(serie)
    historia = train["gasto"]
    n = len(test)
    predicciones = pd.DataFrame(index=test.index)
    detalles: dict = {}

    for nombre, funcion in MODELOS.items():
        try:
            res = funcion(historia, n)
            predicciones[nombre] = res["pred"]
            if "detalle" in res:
                detalles[nombre] = res["detalle"]
            logger.info("Modelo evaluado en holdout: %s", nombre)
        except Exception as exc:  # noqa: BLE001
            logger.error("Falló el modelo %s en holdout: %s", nombre, exc)
            predicciones[nombre] = np.nan

    return predicciones, test["gasto"], detalles


def pronosticar_futuro(serie: pd.DataFrame, nombre_modelo: str, horizonte: int):
    """
    Reentrena el modelo elegido con TODA la serie y pronostica `horizonte` meses.

    Devuelve un DataFrame indexado por las fechas futuras con columnas
    `pred`, `lower`, `upper` (las dos últimas pueden ser NaN si el modelo no
    provee intervalo nativo; en ese caso `evaluacion` las completará).
    """
    fijar_semillas()
    funcion = MODELOS[nombre_modelo]
    res = funcion(serie["gasto"], horizonte)
    idx = _indice_futuro(serie["gasto"], horizonte)
    salida = pd.DataFrame(index=idx)
    salida.index.name = "fecha"
    salida["pred"] = res["pred"]
    salida["lower"] = res["lower"] if res["lower"] is not None else np.nan
    salida["upper"] = res["upper"] if res["upper"] is not None else np.nan

    _persistir_modelo(nombre_modelo, res.get("modelo"))
    return salida


def _persistir_modelo(nombre_modelo: str, objeto) -> None:
    """
    Guarda el modelo entrenado en `outputs/modelos/` según su tipo. Siempre
    deja un pequeño metadato con el nombre del mejor modelo. Si el modelo no es
    serializable de forma nativa, se omite sin romper el flujo.
    """
    config.asegurar_directorios()
    (config.DIR_MODELOS / "mejor_modelo.txt").write_text(
        f"Mejor modelo del Módulo A: {nombre_modelo}\n", encoding="utf-8"
    )
    if objeto is None:
        return
    try:
        if nombre_modelo == "SARIMA":
            objeto.save(str(config.DIR_MODELOS / "sarima_final.pkl"))
        elif nombre_modelo == "XGBoost":
            objeto.save_model(str(config.DIR_MODELOS / "xgboost_final.json"))
        logger.info("Modelo final (%s) guardado en %s", nombre_modelo, config.DIR_MODELOS)
    except Exception as exc:  # noqa: BLE001
        logger.warning("No se pudo serializar el modelo %s: %s", nombre_modelo, exc)
