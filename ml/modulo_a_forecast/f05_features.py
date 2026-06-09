"""
features.py
===========
Ingeniería de características de **calendario** para el pronóstico.

El gran cuello de botella del MAPE es enero: su gasto se desploma porque combina
(1) feriados de inicio de año y (2) el congelamiento de la ejecución presupuestal
pública. Ninguna de esas señales está en los rezagos puros; sí en el calendario.
Este módulo las hace explícitas para que el modelo APRENDA el efecto enero en vez
de promediarlo.

Funciones por granularidad:
* `features_calendario_mes(fecha)`  -> dict de features mensuales.
* `features_calendario_dia(fecha)`  -> dict de features diarias.

Diseño común: cada función recibe una `Timestamp` y devuelve un dict de columnas
numéricas, de modo que se integre con la tabla supervisada de `modelos.py` tanto
en entrenamiento como en pronóstico recursivo.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd

from ml.shared import config


@lru_cache(maxsize=None)
def _feriados(anios: tuple[int, ...]):
    """Calendario de feriados de Perú para los años dados (cacheado)."""
    import holidays

    return holidays.country_holidays(config.PAIS_FERIADOS, years=list(anios))


def dias_habiles_mes(fecha: pd.Timestamp) -> int:
    """
    Número de días hábiles (lun–vie menos feriados de Perú) del mes de `fecha`.

    Es el predictor más directo del volumen mensual: enero pierde días por feriados
    y por el arranque tardío de la ejecución presupuestal.
    """
    inicio = fecha.replace(day=1)
    fin = inicio + pd.offsets.MonthEnd(0)
    rango = pd.bdate_range(inicio, fin)
    fer = _feriados((fecha.year,))
    return int(sum(1 for d in rango if d.date() not in fer))


def fourier_terminos(posicion: float, periodo: float, n_armonicos: int) -> dict:
    """
    Términos de Fourier (seno/coseno) para una estacionalidad de `periodo` pasos.

    Varios armónicos capturan una estacionalidad anual no sinusoidal (p. ej. la
    caída brusca y aislada de enero) mejor que un único par seno/coseno.
    """
    out: dict[str, float] = {}
    for k in range(1, n_armonicos + 1):
        ang = 2 * np.pi * k * posicion / periodo
        out[f"fourier_sin_{k}"] = float(np.sin(ang))
        out[f"fourier_cos_{k}"] = float(np.cos(ang))
    return out


def features_calendario_mes(fecha: pd.Timestamp) -> dict:
    """Features de calendario para un mes (indexado por fin de mes)."""
    mes = int(fecha.month)
    feats = {
        "mes": mes,
        "trimestre": (mes - 1) // 3 + 1,
        "es_enero": int(mes == 1),
        "es_febrero": int(mes == 2),
        "es_diciembre": int(mes == 12),
        "es_inicio_anio": int(mes in (1, 2)),
        "dias_habiles": dias_habiles_mes(fecha),
    }
    feats.update(fourier_terminos(mes, 12, config.N_ARMONICOS_FOURIER))
    return feats


def features_calendario_dia(fecha: pd.Timestamp) -> dict:
    """Features de calendario para un día (granularidad diaria)."""
    fer = _feriados((fecha.year,))
    dia_semana = int(fecha.dayofweek)  # 0=lunes
    es_feriado = int(fecha.date() in fer)
    feats = {
        "dia_semana": dia_semana,
        "dia_mes": int(fecha.day),
        "mes": int(fecha.month),
        "trimestre": int(fecha.quarter),
        "es_finde": int(dia_semana >= 5),
        "es_feriado": es_feriado,
        "es_habil": int(dia_semana < 5 and not es_feriado),
        "es_enero": int(fecha.month == 1),
        "es_diciembre": int(fecha.month == 12),
        "es_fin_de_mes": int(fecha.day >= 25),
    }
    feats.update(fourier_terminos(fecha.dayofyear, 365.25, config.N_ARMONICOS_FOURIER))
    feats.update(fourier_terminos(dia_semana, 7, 2))  # estacionalidad semanal
    return feats


def tabla_calendario(indice: pd.DatetimeIndex, granularidad: str = "mes") -> pd.DataFrame:
    """
    Construye un DataFrame de features de calendario para todo un índice de fechas.
    `granularidad` ∈ {"mes", "dia"}.
    """
    constructor = features_calendario_dia if granularidad == "dia" else features_calendario_mes
    filas = [constructor(f) for f in indice]
    return pd.DataFrame(filas, index=indice)
