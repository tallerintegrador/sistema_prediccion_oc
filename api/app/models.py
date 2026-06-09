"""
models.py
=========
Modelos ORM (SQLAlchemy) de las tablas que consume el backend.

Las tablas las puebla el ETL `ml/load_to_db.py` a partir de los artefactos del ML.
Estos modelos documentan el esquema y permiten consultas tipadas; la capa de
lectura (`data_loader.py`) usa el mismo engine vía pandas para devolver registros.

Tablas (nombres):
    ordenes                  -> dataset_limpio (no se mapea aquí; ~40 columnas)
    serie_mensual_total      -> serie histórica mensual del gasto
    serie_mensual_categoria  -> serie mensual por categoría (formato ancho; no ORM)
    pronostico               -> pronóstico final con intervalo de confianza
    comparacion_modelos      -> métricas de comparación de modelos (Módulo A)
    alertas                  -> alertas rankeadas (Módulo B)
"""

from __future__ import annotations

from sqlalchemy import Column, Float, Integer, String

from .db import Base


class Alerta(Base):
    __tablename__ = "alertas"

    RANK = Column(Integer, primary_key=True)
    ORDEN_ELECTRONICA = Column(String, index=True)
    RUC_ENTIDAD = Column(Integer)
    ENTIDAD = Column(String)
    RUC_PROVEEDOR = Column(Integer)
    PROVEEDOR = Column(String)
    ACUERDO_MARCO = Column(String, index=True)
    FECHA_FORMALIZACION = Column(String)
    TOTAL = Column(Float)
    TIPO_ANOMALIA = Column(String, index=True)
    PUNTAJE_RIESGO = Column(Float, index=True)
    NIVEL = Column(String, index=True)
    MOTIVO = Column(String)
    SCORE_IF = Column(Float)
    SCORE_RED = Column(Float)


class SerieMensualTotal(Base):
    __tablename__ = "serie_mensual_total"

    fecha = Column(String, primary_key=True)
    gasto = Column(Float)
    ordenes = Column(Integer)
    ticket_promedio = Column(Float)


class Pronostico(Base):
    __tablename__ = "pronostico"

    fecha = Column(String, primary_key=True)
    pred = Column(Float)
    lower = Column(Float)
    upper = Column(Float)


class ComparacionModelo(Base):
    __tablename__ = "comparacion_modelos"

    Modelo = Column(String, primary_key=True)
    WAPE = Column(Float)
    MASE = Column(Float)
    MAE = Column(Float)
    RMSE = Column(Float)
    MAPE = Column(Float)
    sMAPE = Column(Float)
    MPE = Column(Float)
