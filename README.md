# SistemaPrediccionOC — Pronóstico y Detección de Anomalías en Órdenes de Compra

Sistema full-stack sobre las **Órdenes de Compra de los Catálogos Electrónicos de
Acuerdos Marco** de **PERÚ COMPRAS**. Integra, en una sola arquitectura:

- **Módulo A — Pronóstico del gasto**: pronostica el **gasto público mensual** con
  13 modelos comparados por backtesting (2022–2026).
- **Módulo B — Detección de anomalías**: detecta, de forma **no supervisada**,
  **órdenes con comportamiento de riesgo** (montos atípicos, IGV inconsistente,
  concentración de proveedor, fraccionamiento, patrones temporales, anomalías de red).
- **API (FastAPI)** que sirve resultados y **todas las métricas del ML** desde una
  **base de datos** y JSON estructurados.
- **Frontend (React + Vite)** con dos vistas: el **Dashboard** de negocio y una
  página **Métricas ML** que sustenta todo lo evaluado (capstone).

```
   data/ (CSV)                 ┌──────────────┐
        │                      │   Frontend   │  Dashboard + Métricas ML
        ▼                      │ React + Vite │  (web/, puerto 5173)
  ┌─────────────┐              └──────┬───────┘
  │     ML      │  artifacts/         │ HTTP
  │ Módulo A/B  │ ───────────►  ┌─────▼───────┐
  │  (ml/)      │  *.csv/.json  │     API     │  /forecast /alerts /metrics/* ...
  └─────┬───────┘  + figuras    │   FastAPI   │  (api/, puerto 8000)
        │ ETL (load_to_db)      └─────┬───────┘
        ▼                             │ SQLAlchemy
  ┌─────────────┐  ◄──────────────────┘
  │ Base de datos│  Postgres (Docker) o SQLite (local)
  └─────────────┘
```

---

## 1. Estructura del repositorio (monorepo)

```
sistema_prediccion_oc/
├── ml/                          # TODO el Machine Learning
│   ├── shared/                  # configuración y utilidades comunes (A y B)
│   │   ├── config.py            # rutas + parámetros (fusión de config A y B)
│   │   └── figuras.py           # registro central de figuras
│   ├── modulo_a_forecast/       # Módulo A: f01_ingesta … f07_evaluacion + pipeline
│   ├── modulo_b_anomalias/      # Módulo B: features / modelos / riesgo / evaluacion
│   ├── metrics_export.py        # consolida las métricas del ML en JSON
│   ├── load_to_db.py            # ETL: artefactos limpios → base de datos
│   ├── run_a.py                 # entrypoint Módulo A   (python -m ml.run_a)
│   └── run_b.py                 # entrypoint Módulo B   (python -m ml.run_b)
├── api/                         # Backend (FastAPI)
│   └── app/
│       ├── main.py              # app, CORS, lifespan, /figures estáticas
│       ├── db.py                # engine/sesión SQLAlchemy (DATABASE_URL)
│       ├── models.py            # modelos ORM de las tablas
│       ├── data_loader.py       # lectura desde DB + JSON de métricas
│       ├── schemas.py           # modelos de respuesta (Pydantic)
│       └── routers/             # health, summary, forecast, categories, alerts, metrics
├── web/                         # Frontend (React + Vite)
│   └── src/
│       ├── pages/               # Dashboard.jsx, MetricasML.jsx
│       ├── components/          # gráficos y tablas (Recharts)
│       ├── api/client.js        # cliente HTTP del API
│       └── hooks/ utils/ styles/
├── data/                        # CSV originales por año (no se modifican)
├── artifacts/                   # salidas del ML: figuras, modelos, resultados, métricas
├── scripts/                     # run.ps1 (Windows), docker-entrypoint.sh
├── tests/                       # pruebas (pytest): API + metrics_export
├── docker-compose.yml · Dockerfile.api · Dockerfile.web
├── Makefile · requirements.txt · requirements-api.txt · .env.example
```

---

## 2. Cómo ejecutar

Hay dos formas: **local** (scripts) para desarrollo y **Docker** para un arranque
portátil de todo el sistema.

### 2.1 Local (Windows / PowerShell)

```powershell
# 1) Instalar dependencias (Python + Node)
./scripts/run.ps1 setup

# 2) Generar artefactos del ML y cargarlos a la base de datos (SQLite por defecto)
./scripts/run.ps1 pipeline      # = ml-a + ml-b + metrics + load-db

# 3) En dos terminales:
./scripts/run.ps1 api           # API  -> http://127.0.0.1:8000/docs
./scripts/run.ps1 web           # Web  -> http://localhost:5173
```

En Linux/macOS los mismos pasos están en el **Makefile**: `make setup`,
`make pipeline`, `make api`, `make web`.

> Si los artefactos del ML ya existen en `artifacts/` y solo quieres servirlos,
> basta con `./scripts/run.ps1 metrics` + `./scripts/run.ps1 load-db` y luego `api`/`web`.

### 2.2 Docker Compose

```bash
cp .env.example .env
docker compose up --build
#   Web  -> http://localhost:5173
#   API  -> http://localhost:8000/docs
#   DB   -> Postgres (servicio 'db')
```

El contenedor del API ejecuta el ETL al arrancar (`docker-entrypoint.sh`:
`metrics_export` + `load_to_db`) y luego levanta uvicorn.

### 2.3 Base de datos

La conexión se controla con `DATABASE_URL` (variable de entorno):

- **Local sin Docker** (por defecto): SQLite en `artifacts/sistema.db`.
- **Docker / producción**: `postgresql+psycopg2://usuario:clave@db:5432/sistema_oc`.

La capa es la misma (SQLAlchemy); cambiar de motor es cambiar la URL.

---

## 3. API — endpoints

| Endpoint | Descripción |
|---|---|
| `GET /health` | Liveness del servicio. |
| `GET /summary` | KPIs del tablero (gasto, órdenes, alertas, próximo mes). |
| `GET /forecast` | Serie histórica + pronóstico con intervalo de confianza. |
| `GET /categories` | Gasto por categoría (ACUERDO_MARCO). |
| `GET /alerts` | Alertas rankeadas (filtros `type`, `level`; paginación). |
| `GET /metrics/forecast` | **Métricas del Módulo A**: comparación de modelos, mejor modelo. |
| `GET /metrics/anomalies` | **Métricas del Módulo B**: precision@k, recall, concordancia, distribuciones. |
| `GET /models` | Registro compacto de modelos comparados. |
| `GET /figures/{archivo}` | Figuras PNG del ML (estáticas). |

Documentación interactiva en `/docs`.

---

## 4. Diccionario de métricas (para la sustentación)

**Módulo A — pronóstico** (todas: *menor es mejor*; promediadas en backtesting de
origen móvil con 6 orígenes y horizonte 6 meses):

| Métrica | Qué mide |
|---|---|
| **WAPE** | Error absoluto **ponderado por monto** (`Σ|e|/Σ|y|`). **Métrica primaria**: robusta al trough de enero. |
| **MASE** | Error vs. el naive estacional. **< 1 ⇒ vence al naive**. |
| **MAE / RMSE** | Error absoluto medio / raíz del error cuadrático (en soles). |
| **MAPE / sMAPE** | Error porcentual (simétrico el segundo). Inestables cuando `y ≈ 0`. |
| **MPE** | Sesgo con signo (>0 subestima, <0 sobreestima). |

**Módulo B — detección (sin etiquetas):**

| Métrica | Qué mide |
|---|---|
| **precision@k** | De las *k* alertas de mayor riesgo, qué fracción está respaldada por una señal interpretable (validez aparente). |
| **Recall por inyección sintética** | Se inyectan anomalías artificiales (monto, IGV, fraccionamiento) y se mide cuántas se detectan (recall por tipo y en el top 1%). |
| **Concordancia (Jaccard)** | Solapamiento del top 1% entre Isolation Forest, red y reglas (robustez/consenso). |
| **Distribuciones** | Conteo por nivel (alto/medio) y por tipo de anomalía; histograma del puntaje de riesgo; estadísticos de red. |

---

## 5. Significado de los datos

Cada registro es **una orden de compra de bienes** (no producto). Columnas clave:

| Columna | Significado |
|---|---|
| `FECHA_FORMALIZACION` | Fecha de formalización. **Eje temporal del pronóstico.** |
| `TOTAL` | Monto total (`SUB_TOTAL` + `IGV`). **Variable a pronosticar (agregada).** |
| `SUB_TOTAL` / `IGV` | Monto antes de IGV / impuesto (soles). |
| `RUC_ENTIDAD` / `ENTIDAD` | Entidad pública compradora. |
| `RUC_PROVEEDOR` / `PROVEEDOR` | Proveedor. |
| `ACUERDO_MARCO` | Convenio/categoría de bienes. |
| `TIPO_PROCEDIMIENTO` | "Compra ordinaria" o "Gran Compra". |
| `ORDEN_ELECTRONICA` | Identificador único de la orden. |
| `ESTADO_ORDEN_ELECTRONICA` | Estado (ACEPTADA, PAGADA, RESUELTA, …). |

Datos reales: **53 CSV** (2022–2026), **483,889 órdenes** consolidadas; separador
`;`, codificación mixta (52 `latin-1`, 1 `utf-8`), dos formatos de fecha. Tras
limpiar: 0 fechas nulas, sin duplicados, `TOTAL = SUB_TOTAL + IGV` exacto.

---

## 6. Resultados del Módulo A

Sobre 53 meses (ene-2022 a may-2026), gasto acumulado ~**S/ 8,078 M**. Métricas
medias del backtest (extracto):

| Modelo | WAPE (%) | MASE | MAPE (%) |
|---|---|---|---|
| **Ensemble** ⭐ (mediana ETS+drift+SARIMA) | **16.5** | **0.75** | 28.0 |
| ETS (Holt-Winters) | 17.4 | 0.78 | 24.4 |
| SARIMA | 19.5 | 0.89 | 33.0 |
| LightGBM (calendario, Tweedie) | 20.9 | 0.93 | 46.7 |
| LSTM | 38.9 | 1.66 | 66.2 |
| Transformer | 47.5 | 2.05 | 206.5 |

El mejor modelo es un **Ensemble** (WAPE 16.5%, MASE 0.75). Los modelos complejos
pierden frente a los simples por la poca historia (~50 puntos mensuales). Detalle:
[`artifacts/resultados/informe_resultados.md`](artifacts/resultados/informe_resultados.md).

---

## 7. Resultados del Módulo B

Sobre **483,692 órdenes válidas** (2,336 entidades · 3,563 proveedores):

- **Umbral de fraccionamiento = S/ 100,000**, descubierto (no asumido) por la
  discontinuidad de P(Gran Compra | TOTAL) y el *bunching* bajo el umbral.
- **IGV** 0.18 en el 96.1 %; **18,888 (3.90 %)** inconsistentes (casi todas IGV = 0).
- **Red**: grafo de **5,899 nodos** y **189,825 aristas**, **15 comunidades**,
  **309 proveedores cautivos**.
- **Alertas**: **24,185** medio/alto (**4,837 alto**), rankeadas.
- **precision@k = 100 %** (k = 20/50/100). **Recall** inyección: fraccionamiento
  100 %, IGV 74 %, monto 28 %, global 67 %. **Concordancia**: 517 órdenes en el top
  1 % de los tres métodos.

Detalle: [`artifacts/resultados/informe_modulo_b.md`](artifacts/resultados/informe_modulo_b.md).

---

## 8. Pruebas y reproducibilidad

```bash
python -m pytest          # pruebas del API y del export de métricas
```

- **Semillas fijas** (`config.SEMILLA = 42`) en numpy, random, torch, Isolation
  Forest, autoencoder y detección de comunidades.
- **Rutas relativas** desde `ml/shared/config.py`; artefactos reproducibles en
  `artifacts/`.
- Flujo único: `./scripts/run.ps1 pipeline` (o `make pipeline`) regenera todo.

> **Nota de entorno**: el ML pesado (torch/sklearn/xgboost/lightgbm) requiere un
> entorno con esas librerías correctamente instaladas. El API y el ETL son ligeros
> (pandas/SQLAlchemy) y corren con `requirements-api.txt`.
