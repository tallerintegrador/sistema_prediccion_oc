# Makefile — Orquestador de desarrollo (Linux/macOS).
# Uso: make <objetivo>   (ver `make help`)

PY ?= python

.PHONY: help setup ml-a ml-b metrics load-db api web pipeline test docker-up docker-down

help:
	@echo "Objetivos disponibles:"
	@echo "  setup      Instala dependencias de Python y de Node (web/)"
	@echo "  ml-a       Pipeline del Módulo A (pronóstico)"
	@echo "  ml-b       Pipeline del Módulo B (anomalías)"
	@echo "  metrics    Regenera los JSON de métricas"
	@echo "  load-db    ETL: artefactos -> base de datos"
	@echo "  pipeline   ml-a + ml-b + metrics + load-db"
	@echo "  api        Levanta el API (uvicorn, puerto 8000)"
	@echo "  web        Levanta el frontend (Vite, puerto 5173)"
	@echo "  test       Ejecuta pytest"
	@echo "  docker-up  Levanta todo con Docker Compose"
	@echo "  docker-down Detiene Docker Compose"

setup:
	$(PY) -m pip install -r requirements.txt
	cd web && npm install

ml-a:
	$(PY) -m ml.run_a

ml-b:
	$(PY) -m ml.run_b

metrics:
	$(PY) -m ml.metrics_export

load-db:
	$(PY) -m ml.load_to_db

pipeline: ml-a ml-b metrics load-db

api:
	$(PY) -m uvicorn api.app.main:app --reload --port 8000

web:
	cd web && npm run dev

test:
	$(PY) -m pytest -q

docker-up:
	docker compose up --build

docker-down:
	docker compose down
