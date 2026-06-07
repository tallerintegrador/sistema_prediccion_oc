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

from . import config, features

logger = logging.getLogger(__name__)


def _serie_gasto(historia) -> pd.Series:
    """
    Normaliza la entrada de los modelos: acepta tanto la `pd.Series` de gasto
    (convención histórica) como el `pd.DataFrame` completo de la serie (con
    columnas gasto/ordenes/ticket_promedio, necesarias para la descomposición) y
    devuelve siempre la serie de gasto.
    """
    if isinstance(historia, pd.DataFrame):
        return historia["gasto"]
    return historia


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
    """
    Genera las n fechas futuras siguientes a la última de `historia`, infiriendo
    la frecuencia del índice (mensual/semanal/diaria). Cae a fin de mes si no se
    puede inferir (serie mensual recortada).
    """
    freq = pd.infer_freq(historia.index) or "ME"
    return pd.date_range(historia.index[-1], periods=n + 1, freq=freq)[1:]


# ---------------------------------------------------------------------------
# 1) Modelos base (naive)
# ---------------------------------------------------------------------------
def pronostico_naive(historia, n: int) -> dict:
    """Naive simple: repite el último valor observado."""
    serie_g = _serie_gasto(historia)
    ultimo = float(serie_g.iloc[-1])
    return {"pred": np.full(n, ultimo), "lower": None, "upper": None}


def pronostico_naive_estacional(historia, n: int) -> dict:
    """
    Naive estacional: el pronóstico del mes futuro es el valor del mismo mes del
    año anterior (lag-12). Es una referencia exigente dada la fuerte
    estacionalidad anual de esta serie.
    """
    serie_g = _serie_gasto(historia)
    s = config.PERIODO_ESTACIONAL
    ultima_temporada = serie_g.iloc[-s:].to_numpy()
    pred = np.array([ultima_temporada[i % s] for i in range(n)])
    return {"pred": pred, "lower": None, "upper": None}


def pronostico_estacional_drift(historia, n: int) -> dict:
    """
    Naive estacional con **ajuste de nivel** (drift interanual): toma la FORMA
    estacional del año anterior (lag-12) y la reescala por el cambio de nivel
    reciente, medido como el cociente de las sumas móviles de 12 meses
    (Σ últimos 12 / Σ 12 previos).

    Motivación: el backtest muestra que todos los modelos **sobre-predicen ~15-20%**
    (MPE negativo) porque el volumen viene cayendo (menos órdenes año a año). El
    naive estacional puro arrastra el nivel del año pasado; este ajuste corrige el
    sesgo manteniendo la estacionalidad, y es robusto (usa agregados, no meses
    sueltos volátiles como enero).
    """
    serie_g = _serie_gasto(historia)
    s = config.PERIODO_ESTACIONAL
    valores = serie_g.to_numpy(dtype=float)
    ultima_temporada = valores[-s:]
    if len(valores) >= 2 * s:
        suma_reciente = valores[-s:].sum()
        suma_previa = valores[-2 * s:-s].sum()
        factor = suma_reciente / suma_previa if suma_previa > 0 else 1.0
    else:
        factor = 1.0
    pred = np.array([ultima_temporada[i % s] * factor for i in range(n)])
    return {"pred": pred, "lower": None, "upper": None, "detalle": f"drift x{factor:.3f}"}


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


def pronostico_sarima(historia, n: int) -> dict:
    """
    Ajusta un SARIMA sobre log(gasto) y pronostica n meses con intervalo de
    confianza. Los resultados se devuelven en la escala original (soles).
    """
    serie_g = _serie_gasto(historia)
    serie_log = np.log(serie_g.astype(float))
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
# 3) Árboles de gradiente (XGBoost / LightGBM) con calendario — recursivo
# ---------------------------------------------------------------------------
# Configuración de rezagos/calendario por granularidad. Permite reutilizar el
# MISMO motor de árboles para la serie mensual, semanal y diaria.
_CONFIG_GRAN = {
    "mes":    {"lags": [1, 2, 3, 12], "medias": [3, 12], "ratio": 12, "min": 13, "calendario": "mes"},
    "semana": {"lags": [1, 2, 4, 13, 52], "medias": [4, 13], "ratio": 52, "min": 53, "calendario": None},
    "dia":    {"lags": [1, 7, 14, 30, 365], "medias": [7, 30], "ratio": 365, "min": 366, "calendario": "dia"},
}


def _features_drivers(fecha, drivers: pd.DataFrame | None) -> dict:
    """
    Variables de **drivers internos REZAGADAS** para predecir el valor en `fecha`.

    Se usa el rezago anual (lag-12): el valor de cada driver en el MISMO mes del
    año anterior. Es la única forma honesta de incorporarlos en un pronóstico
    multi-paso sin fuga (el lag-12 siempre es conocido dentro de un horizonte ≤ 12
    meses). Si no hay drivers, devuelve un dict vacío y nada cambia.
    """
    if drivers is None or len(drivers) == 0:
        return {}
    objetivo = pd.Timestamp(fecha) - pd.DateOffset(years=1)
    previos = drivers[drivers.index <= objetivo]
    fila = previos.iloc[-1] if not previos.empty else drivers.iloc[0]
    return {f"drv_{c}_lag12": float(fila[c]) for c in drivers.columns}


def _features_desde_historia(
    serie: pd.Series, fecha, granularidad: str = "mes",
    drivers: pd.DataFrame | None = None,
) -> dict:
    """
    Vector de variables para predecir el valor en `fecha` con la `serie` histórica
    disponible hasta justo antes de esa fecha. Los rezagos y el calendario se
    adaptan a la `granularidad` ('mes', 'semana', 'dia').

    * Rezagos y medias móviles propios de la frecuencia (autorregresivo).
    * `ratio_estacional` = último / valor de hace una temporada (tendencia relativa).
    * Features de **calendario** (`features.py`): días hábiles, indicadores
      enero/diciembre y armónicos de Fourier — clave para aprender el desplome de
      enero en vez de promediarlo.
    * Si se pasan `drivers`, sus valores REZAGADOS (lag-12) se añaden como señal
      exógena interna (ver `_features_drivers`).
    """
    cfg = _CONFIG_GRAN[granularidad]
    valores = serie.to_numpy(dtype=float)
    n = len(valores)
    feats: dict[str, float] = {}
    for L in cfg["lags"]:
        feats[f"lag{L}"] = valores[-L] if n >= L else valores[-1]
    for M in cfg["medias"]:
        feats[f"media_{M}"] = valores[-M:].mean() if n >= M else valores.mean()
    r = cfg["ratio"]
    feats["ratio_estacional"] = valores[-1] / valores[-r] if n >= r and valores[-r] > 0 else 1.0
    feats["tendencia"] = n
    if cfg["calendario"] == "mes":
        feats.update(features.features_calendario_mes(pd.Timestamp(fecha)))
    elif cfg["calendario"] == "dia":
        feats.update(features.features_calendario_dia(pd.Timestamp(fecha)))
    feats.update(_features_drivers(fecha, drivers))
    return feats


def _tabla_supervisada(
    serie: pd.Series, granularidad: str = "mes", drivers: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Genera la tabla (X, y) de aprendizaje supervisado a partir de la serie."""
    min_historia = _CONFIG_GRAN[granularidad]["min"]
    filas = []
    fechas = serie.index
    for i in range(min_historia, len(serie)):
        hist = serie.iloc[:i]
        feats = _features_desde_historia(hist, fechas[i], granularidad, drivers)
        feats["y"] = float(serie.iloc[i])
        filas.append(feats)
    return pd.DataFrame(filas)


def _crear_gbm(tipo: str, objetivo: str):
    """
    Crea un regresor de árboles de gradiente con un objetivo orientado al error
    relativo (proxy de WAPE/MAPE): `tweedie` (recomendado para montos no-negativos
    de cola larga) sobre el target en escala original.
    """
    if tipo == "lightgbm":
        from lightgbm import LGBMRegressor

        obj = "tweedie" if objetivo == "tweedie" else "regression"
        return LGBMRegressor(
            objective=obj, tweedie_variance_power=config.TWEEDIE_VARIANCE_POWER,
            n_estimators=400, learning_rate=0.05, num_leaves=15,
            min_child_samples=5, subsample=0.9, colsample_bytree=0.9,
            reg_lambda=1.0, random_state=config.SEMILLA, n_jobs=1, verbose=-1,
        )
    from xgboost import XGBRegressor

    obj = "reg:tweedie" if objetivo == "tweedie" else "reg:squarederror"
    return XGBRegressor(
        objective=obj, tweedie_variance_power=config.TWEEDIE_VARIANCE_POWER,
        n_estimators=400, max_depth=3, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=config.SEMILLA, n_jobs=1,
    )


def _gbm_recursivo(serie_g: pd.Series, n: int, tipo: str, objetivo: str = config.OBJETIVO_GBM,
                   log_target: bool = False, params: dict | None = None,
                   granularidad: str = "mes", drivers: pd.DataFrame | None = None) -> dict:
    """
    Motor genérico de pronóstico recursivo con árboles de gradiente. Sirve para
    cualquier serie no-negativa (gasto, nº de órdenes, ticket) y cualquier
    `granularidad` ('mes', 'semana', 'dia'): entrena sobre la tabla supervisada
    (lags + calendario [+ drivers rezagados]) y predice paso a paso, reincorporando
    cada predicción para la siguiente.
    """
    fijar_semillas()
    tabla = _tabla_supervisada(serie_g, granularidad, drivers)
    columnas_x = [c for c in tabla.columns if c != "y"]
    X = tabla[columnas_x]
    y = tabla["y"]
    if log_target:
        y = np.log1p(y)

    modelo = _crear_gbm(tipo, objetivo)
    if params:
        modelo.set_params(**params)
    modelo.fit(X, y)

    hist = serie_g.copy()
    fechas_fut = _indice_futuro(serie_g, n)
    preds = []
    for fecha in fechas_fut:
        feats = _features_desde_historia(hist, fecha, granularidad, drivers)
        x = pd.DataFrame([feats])[columnas_x]
        yhat = float(modelo.predict(x)[0])
        if log_target:
            yhat = float(np.expm1(yhat))
        yhat = max(yhat, 0.0)  # el gasto/órdenes no es negativo
        preds.append(yhat)
        hist = pd.concat([hist, pd.Series([yhat], index=[fecha])])
    return {"pred": np.array(preds), "lower": None, "upper": None, "modelo": modelo}


def pronostico_xgboost(historia, n: int) -> dict:
    """XGBoost con calendario y objetivo Tweedie, pronóstico recursivo."""
    return _gbm_recursivo(_serie_gasto(historia), n, tipo="xgboost")


def pronostico_lightgbm(historia, n: int) -> dict:
    """LightGBM con calendario y objetivo Tweedie, pronóstico recursivo."""
    return _gbm_recursivo(_serie_gasto(historia), n, tipo="lightgbm")


def pronostico_lightgbm_con_params(historia, n: int, params: dict | None = None,
                                   drivers: pd.DataFrame | None = None) -> dict:
    """
    LightGBM con hiperparámetros explícitos (p. ej. los de Optuna) y/o drivers
    internos rezagados. Pensado para registrarse vía `functools.partial` en la
    comparación de modelos sin romper la firma `f(historia, n)` del dict `MODELOS`.
    """
    return _gbm_recursivo(_serie_gasto(historia), n, tipo="lightgbm", params=params, drivers=drivers)


# ---------------------------------------------------------------------------
# 3b) Modelo DESCOMPUESTO: gasto = nº de órdenes × ticket promedio
# ---------------------------------------------------------------------------
def pronostico_descompuesto(historia, n: int) -> dict:
    """
    Pronostica por separado el **número de órdenes** y el **ticket promedio**, y
    multiplica. La caída de enero es esencialmente una caída del CONTEO de órdenes
    (~460 vs ~10 000): el conteo es muy estacional y predecible, y el ticket es casi
    estacionario. Modelar cada componente reduce la varianza y doma enero — palanca
    directa sobre WAPE/MAPE.

    Requiere el DataFrame completo (con columnas `ordenes` y `ticket_promedio`). Si
    solo se recibe la serie de gasto, cae al LightGBM directo sobre el gasto.
    """
    if not isinstance(historia, pd.DataFrame) or "ordenes" not in historia.columns:
        return _gbm_recursivo(_serie_gasto(historia), n, tipo="lightgbm")

    ordenes = historia["ordenes"].astype(float)
    ticket = historia["ticket_promedio"].astype(float)

    pred_ordenes = _gbm_recursivo(ordenes, n, tipo="lightgbm")["pred"]
    # El ticket es estable: un GBM en escala log capta su leve tendencia sin sobreajustar.
    pred_ticket = _gbm_recursivo(ticket, n, tipo="lightgbm", objetivo="regresion", log_target=True)["pred"]

    pred = pred_ordenes * pred_ticket
    return {
        "pred": pred, "lower": None, "upper": None,
        "detalle": "órdenes × ticket (LightGBM)",
        "componentes": {"ordenes": pred_ordenes, "ticket": pred_ticket},
    }


# ---------------------------------------------------------------------------
# 3c) ETS / Holt-Winters (suavizado exponencial estacional)
# ---------------------------------------------------------------------------
def pronostico_ets(historia, n: int) -> dict:
    """
    Holt-Winters (suavizado exponencial) con estacionalidad anual multiplicativa,
    sobre log(gasto). Baseline estadístico fuerte y muy barato, complementario al
    SARIMA.
    """
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    serie_g = _serie_gasto(historia).astype(float)
    serie_log = np.log(serie_g)
    serie_log.index = pd.DatetimeIndex(serie_log.index)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo = ExponentialSmoothing(
            serie_log, trend="add", seasonal="add",
            seasonal_periods=config.PERIODO_ESTACIONAL,
            initialization_method="estimated",
        ).fit()
    pred = np.exp(modelo.forecast(n).to_numpy())
    return {"pred": pred, "lower": None, "upper": None, "detalle": "ETS(A,A) log"}


# ---------------------------------------------------------------------------
# 3d) Ensemble: mediana de los modelos estadísticos robustos
# ---------------------------------------------------------------------------
def pronostico_ensemble(historia, n: int) -> dict:
    """
    Combina por **mediana** los modelos estadísticos más robustos (ETS, naive
    estacional con drift, SARIMA). La mediana reduce la varianza del pronóstico y
    es resistente a que un modelo falle puntualmente, sin sobreajustar. En series
    cortas y ruidosas como esta, combinar suele batir a cualquier modelo individual.
    """
    componentes = {
        "ETS": pronostico_ets,
        "Estacional drift": pronostico_estacional_drift,
        "SARIMA": pronostico_sarima,
    }
    preds = []
    for nombre, fn in componentes.items():
        try:
            preds.append(np.asarray(fn(historia, n)["pred"], dtype=float))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Ensemble: componente %s falló: %s", nombre, exc)
    if not preds:
        raise RuntimeError("Ensemble sin componentes válidos.")
    pred = np.median(np.vstack(preds), axis=0)
    return {"pred": pred, "lower": None, "upper": None, "detalle": "mediana(ETS, drift, SARIMA)"}


# ---------------------------------------------------------------------------
# 3e) Modelo GLOBAL / JERÁRQUICO por categoría (cross-learning + reconciliación)
# ---------------------------------------------------------------------------
def _serie_por_categoria(panel: pd.DataFrame) -> dict:
    """Dict {categoria -> serie mensual de gasto} a partir del panel largo."""
    out: dict[str, pd.Series] = {}
    for cat, g in panel.groupby("categoria"):
        out[str(cat)] = g.sort_values("fecha").set_index("fecha")["gasto"]
    return out


def _tabla_supervisada_panel(panel: pd.DataFrame, drivers: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Tabla supervisada GLOBAL: apila las filas de todas las categorías del panel.

    Cada fila reusa los features autorregresivos/calendario de
    `_features_desde_historia` (sobre la historia de ESA categoría) y añade, sin
    fuga, variables transversales: `categoria` (categórica), `share_categoria`
    (peso REZAGADO de la categoría en el total) y `antiguedad` (meses desde su
    primer gasto). Un único modelo entrenado sobre esta tabla aprende de las ~N
    categorías a la vez (cross-learning).
    """
    series = _serie_por_categoria(panel)
    total_por_fecha = panel.groupby("fecha")["gasto"].sum()
    min_historia = _CONFIG_GRAN["mes"]["min"]
    filas = []
    for cat, serie in series.items():
        valores = serie.to_numpy(dtype=float)
        nz = int(np.argmax(valores > 0)) if (valores > 0).any() else 0
        fechas = serie.index
        for i in range(min_historia, len(serie)):
            hist = serie.iloc[:i]
            feats = _features_desde_historia(hist, fechas[i], "mes", drivers)
            feats["categoria"] = cat
            tot_prev = float(total_por_fecha.get(fechas[i - 1], 0.0))
            feats["share_categoria"] = float(hist.iloc[-1] / tot_prev) if tot_prev > 0 else 0.0
            feats["antiguedad"] = max(i - nz, 0)
            feats["y"] = float(serie.iloc[i])
            filas.append(feats)
    return pd.DataFrame(filas)


def pronostico_global_jerarquico(panel: pd.DataFrame, n: int,
                                 drivers: pd.DataFrame | None = None,
                                 params: dict | None = None) -> dict:
    """
    Modelo **global** por categoría con reconciliación **bottom-up**.

    Entrena UN solo LightGBM (objetivo Tweedie) sobre el panel apilado de todas las
    categorías y pronostica cada una de forma recursiva; el gasto total es la SUMA
    de las categorías pronosticadas. Como el panel se construye con top-N + 'OTRAS'
    (que absorbe el resto), la suma reconstituye exactamente el agregado.

    El cross-learning da al modelo ~N× más muestras que la serie agregada: es la
    mayor palanca interna disponible sin datos exógenos. Devuelve `pred` (total) y
    `componentes` (pronóstico por categoría).
    """
    fijar_semillas()
    tabla = _tabla_supervisada_panel(panel, drivers)
    if tabla.empty:
        raise RuntimeError("Panel demasiado corto para el modelo global jerárquico.")

    columnas_x = [c for c in tabla.columns if c != "y"]
    categorias = sorted(panel["categoria"].astype(str).unique())
    cat_dtype = pd.CategoricalDtype(categories=categorias)

    X = tabla[columnas_x].copy()
    X["categoria"] = X["categoria"].astype(cat_dtype)
    y = tabla["y"]

    modelo = _crear_gbm("lightgbm", config.OBJETIVO_GBM)
    if params:
        modelo.set_params(**params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo.fit(X, y, categorical_feature=["categoria"])

    series = _serie_por_categoria(panel)
    hist = {cat: serie.copy() for cat, serie in series.items()}
    running_total = panel.groupby("fecha")["gasto"].sum().copy()
    nz = {
        cat: (int(np.argmax(s.to_numpy() > 0)) if (s.to_numpy() > 0).any() else 0)
        for cat, s in series.items()
    }

    fechas_fut = _indice_futuro(next(iter(series.values())), n)
    pred_por_cat: dict[str, list] = {cat: [] for cat in series}
    for fecha in fechas_fut:
        preds_mes = {}
        for cat, h in hist.items():
            feats = _features_desde_historia(h, fecha, "mes", drivers)
            feats["categoria"] = cat
            tot_prev = float(running_total.get(h.index[-1], 0.0))
            feats["share_categoria"] = float(h.iloc[-1] / tot_prev) if tot_prev > 0 else 0.0
            feats["antiguedad"] = max(len(h) - nz[cat], 0)
            x = pd.DataFrame([feats])[columnas_x]
            x["categoria"] = x["categoria"].astype(cat_dtype)
            preds_mes[cat] = max(float(modelo.predict(x)[0]), 0.0)
        for cat in series:
            hist[cat] = pd.concat([hist[cat], pd.Series([preds_mes[cat]], index=[fecha])])
            pred_por_cat[cat].append(preds_mes[cat])
        running_total = pd.concat(
            [running_total, pd.Series([sum(preds_mes.values())], index=[fecha])]
        )

    total_pred = np.sum([np.asarray(v, dtype=float) for v in pred_por_cat.values()], axis=0)
    return {
        "pred": total_pred, "lower": None, "upper": None,
        "detalle": f"global LightGBM ({len(series)} categorías, bottom-up)",
        "componentes": {c: np.asarray(v, dtype=float) for c, v in pred_por_cat.items()},
        "modelo": modelo,
    }


# ---------------------------------------------------------------------------
# 3f) Tuning de hiperparámetros del GBM con Optuna (objetivo = WAPE rolling-origin)
# ---------------------------------------------------------------------------
def optimizar_gbm(serie_g: pd.Series, granularidad: str = "mes",
                  n_trials: int | None = None, drivers: pd.DataFrame | None = None) -> dict | None:
    """
    Busca hiperparámetros del LightGBM minimizando el **WAPE medio** en validación
    de origen móvil (los mismos cortes que el backtesting). Devuelve el dict de
    `best_params` o `None` si Optuna está desactivado/ausente o la serie es corta.

    El informe documenta que el tuning no rompe el techo del 10% (los GBM pierden
    contra los estadísticos); por eso se acota a `config.N_TRIALS_OPTUNA` trials.
    """
    if not config.USAR_OPTUNA:
        return None
    try:
        import optuna
    except ImportError:
        logger.warning("Optuna no está instalado; se omite el tuning de hiperparámetros.")
        return None

    n_trials = n_trials or config.N_TRIALS_OPTUNA
    h, m, n_total = config.MESES_HOLDOUT, config.PERIODO_ESTACIONAL, len(serie_g)
    cortes = sorted(
        c for c in (n_total - h - i for i in range(config.BACKTEST_N_ORIGENES)) if c >= 2 * m
    )
    if not cortes:
        logger.warning("Serie demasiado corta para tuning con Optuna; se omite.")
        return None

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def _objetivo(trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 7, 63),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600, step=50),
            "min_child_samples": trial.suggest_int("min_child_samples", 3, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
        wapes = []
        for corte in cortes:
            train = serie_g.iloc[:corte]
            test = serie_g.iloc[corte:corte + h].to_numpy(dtype=float)
            try:
                pred = _gbm_recursivo(
                    train, len(test), tipo="lightgbm", params=params,
                    granularidad=granularidad, drivers=drivers,
                )["pred"]
            except Exception:  # noqa: BLE001
                return float("inf")
            den = np.abs(test).sum()
            wapes.append(np.abs(test - pred).sum() / den * 100 if den > 0 else np.nan)
        valor = float(np.nanmean(wapes))
        return valor if np.isfinite(valor) else float("inf")

    estudio = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=config.SEMILLA),
    )
    estudio.optimize(_objetivo, n_trials=n_trials, show_progress_bar=False)
    logger.info("Optuna: mejor WAPE(CV)=%.2f%% con %s", estudio.best_value, estudio.best_params)
    return estudio.best_params


# ---------------------------------------------------------------------------
# 4) LSTM (PyTorch) — sobre log-serie escalada, pronóstico recursivo
# ---------------------------------------------------------------------------
def pronostico_lstm(historia, n: int, ventana: int = 12, epocas: int = 400) -> dict:
    """
    Entrena una pequeña LSTM sobre ventanas de la serie (en escala log y
    normalizada) y pronostica n meses de forma recursiva.

    Nota: la serie es corta (~50 puntos), por lo que la LSTM está limitada por
    los datos; se reporta su desempeño con honestidad en la comparación.
    """
    import torch
    import torch.nn as nn

    fijar_semillas()
    serie_g = _serie_gasto(historia)

    # Transformación: log para estabilizar varianza + estandarización.
    serie_log = np.log(serie_g.to_numpy(dtype=float))
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
# 5) Transformer (PyTorch) — encoder de auto-atención, recursivo
# ---------------------------------------------------------------------------
def pronostico_transformer(historia, n: int, ventana: int = 12, epocas: int = 300) -> dict:
    """
    Transformer mínimo (un bloque de auto-atención) sobre ventanas de la serie en
    escala log estandarizada. Se incluye para **documentar el trade-off**: sin
    sesgo inductivo temporal fuerte, un Transformer necesita miles de secuencias;
    con ~50 puntos sobreajusta y rinde peor que los modelos estadísticos a mayor
    costo. La comparación lo evidencia con honestidad.
    """
    import torch
    import torch.nn as nn

    fijar_semillas()
    serie_g = _serie_gasto(historia)
    serie_log = np.log(serie_g.to_numpy(dtype=float))
    mu, sigma = serie_log.mean(), serie_log.std() + 1e-8
    z = (serie_log - mu) / sigma
    if len(z) <= ventana + 2:
        ventana = max(3, len(z) // 2)

    X_list, y_list = [], []
    for i in range(ventana, len(z)):
        X_list.append(z[i - ventana:i])
        y_list.append(z[i])
    X = torch.tensor(np.array(X_list), dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(-1)

    class RedTransformer(nn.Module):
        """Embedding lineal + 1 capa TransformerEncoder + cabeza densa."""

        def __init__(self, d_model: int = 32, n_heads: int = 4):
            super().__init__()
            self.proj = nn.Linear(1, d_model)
            capa = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=64,
                dropout=0.1, batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(capa, num_layers=1)
            self.fc = nn.Linear(d_model, 1)

        def forward(self, x):
            h = self.encoder(self.proj(x))
            return self.fc(h[:, -1, :])

    modelo = RedTransformer()
    opt = torch.optim.Adam(modelo.parameters(), lr=0.005)
    perdida = nn.MSELoss()
    modelo.train()
    for _ in range(epocas):
        opt.zero_grad()
        error = perdida(modelo(X), y)
        error.backward()
        opt.step()

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

    preds = np.exp(np.array(preds_z) * sigma + mu)
    return {"pred": preds, "lower": None, "upper": None}


# ---------------------------------------------------------------------------
# Registro de modelos disponibles
# ---------------------------------------------------------------------------
MODELOS = {
    "Naive": pronostico_naive,
    "Naive estacional": pronostico_naive_estacional,
    "Estacional drift": pronostico_estacional_drift,
    "SARIMA": pronostico_sarima,
    "ETS": pronostico_ets,
    "Ensemble": pronostico_ensemble,
    "XGBoost": pronostico_xgboost,
    "LightGBM": pronostico_lightgbm,
    "Descompuesto": pronostico_descompuesto,
    "LSTM": pronostico_lstm,
    "Transformer": pronostico_transformer,
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
    historia = train  # DataFrame completo (gasto/ordenes/ticket) para descomposición
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
    res = funcion(serie, horizonte)
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
