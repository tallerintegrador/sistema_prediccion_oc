"""
db.py
=====
Capa de conexión a la base de datos (SQLAlchemy).

La URL se toma de la variable de entorno `DATABASE_URL`. Por defecto usa un SQLite
local en `artifacts/sistema.db`, de modo que el sistema funcione sin infraestructura
extra en desarrollo. En despliegue (Docker Compose) se apunta a PostgreSQL con, por
ejemplo:

    DATABASE_URL=postgresql+psycopg2://user:pass@db:5432/sistema_oc

El ETL (`ml/load_to_db.py`) escribe las tablas y el backend solo las LEE.
"""

from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# api/app/db.py -> parents[2] = raíz del proyecto.
RAIZ_PROYECTO: Path = Path(__file__).resolve().parents[2]
_RUTA_SQLITE = (RAIZ_PROYECTO / "artifacts" / "sistema.db").as_posix()

DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{_RUTA_SQLITE}")

# SQLite necesita check_same_thread=False para usarse desde varios hilos (uvicorn).
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True, connect_args=_connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()


def get_engine():
    """Devuelve el engine compartido (lo usa también el ETL)."""
    return engine


def get_db():
    """Dependencia de FastAPI: entrega una sesión y la cierra al terminar."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
