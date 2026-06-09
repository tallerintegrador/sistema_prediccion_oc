#!/bin/sh
# Entrypoint del contenedor del API:
# 1) regenera los JSON de métricas desde los artefactos (no toca la BD),
# 2) opcionalmente carga los artefactos a la BD (solo si RUN_ETL_ON_BOOT=1),
# 3) arranca uvicorn en el puerto que indique el entorno ($PORT, p.ej. Render).
set -e

echo "[entrypoint] Regenerando JSON de métricas..."
python -m ml.metrics_export || echo "[entrypoint] aviso: metrics_export falló (¿faltan artefactos?)"

# La BD se pobla una sola vez. Recargar ~483k filas en cada arranque (cold start
# de Render) es lento e innecesario. Por eso el ETL es opt-in:
#   - Docker Compose lo activa (Postgres local vacío) con RUN_ETL_ON_BOOT=1.
#   - En Render (Neon ya poblada) se deja sin definir -> se salta.
if [ "${RUN_ETL_ON_BOOT:-0}" = "1" ]; then
  echo "[entrypoint] RUN_ETL_ON_BOOT=1 -> cargando artefactos a la base de datos..."
  python -m ml.load_to_db || echo "[entrypoint] aviso: load_to_db falló (¿DB lista?)"
else
  echo "[entrypoint] ETL omitido (RUN_ETL_ON_BOOT!=1); se asume BD ya poblada."
fi

echo "[entrypoint] Iniciando API en el puerto ${PORT:-8000}..."
exec uvicorn api.app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
