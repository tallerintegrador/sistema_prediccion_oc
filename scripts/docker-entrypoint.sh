#!/bin/sh
# Entrypoint del contenedor del API:
# 1) carga los artefactos del ML a la base de datos (idempotente),
# 2) arranca el servidor uvicorn.
set -e

echo "[entrypoint] Cargando artefactos a la base de datos..."
python -m ml.metrics_export || echo "[entrypoint] aviso: metrics_export falló (¿faltan artefactos?)"
python -m ml.load_to_db || echo "[entrypoint] aviso: load_to_db falló (¿DB lista?)"

echo "[entrypoint] Iniciando API..."
exec uvicorn api.app.main:app --host 0.0.0.0 --port 8000
