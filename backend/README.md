# Backend — SistemaPrediccionOC (API REST)

API construida con **FastAPI** que expone, mediante endpoints en inglés, los
resultados ya calculados por:

- **Módulo A** — pronóstico del gasto público en órdenes de compra.
- **Módulo B** — detección de órdenes de compra anómalas.

El backend **solo lee** los artefactos de `outputs/`; no reentrena modelos ni
reprocesa los CSV crudos de `data/`.

## Estructura

```
backend/
└── app/
    ├── main.py            # App FastAPI + CORS + registro de routers
    ├── config.py          # Rutas a outputs/ y parámetros del API
    ├── data_loader.py     # Lectura y caché en memoria de los CSV
    ├── schemas.py         # Modelos de respuesta (Pydantic)
    └── routers/
        ├── health.py      # GET /health
        ├── summary.py     # GET /summary
        ├── forecast.py    # GET /forecast
        ├── categories.py  # GET /categories
        └── alerts.py      # GET /alerts
```

## Cómo levantarlo

Desde la **raíz del proyecto** (`SistemaPrediccionOC/`):

```bash
pip install -r requirements.txt          # solo la primera vez
uvicorn backend.app.main:app --reload
```

El API queda disponible en **http://127.0.0.1:8000**
Documentación interactiva (Swagger UI): **http://127.0.0.1:8000/docs**

## Endpoints

| Método | Ruta          | Descripción |
|--------|---------------|-------------|
| GET    | `/health`     | Confirma que el API está vivo. |
| GET    | `/summary`    | Indicadores clave: gasto del último periodo, nº de órdenes, nº de alertas, pronóstico del próximo mes. |
| GET    | `/forecast`   | Serie histórica mensual + pronóstico de los próximos meses + intervalo de confianza. |
| GET    | `/categories` | Gasto acumulado por categoría (`ACUERDO_MARCO`). |
| GET    | `/alerts`     | Alertas rankeadas, con filtros y paginación. |

### Filtros de `/alerts`

| Parámetro   | Valores | Por defecto |
|-------------|---------|-------------|
| `type`      | tipo de anomalía: `IGV inconsistente`, `Anomalía de red`, `Concentración de proveedor`, `Posible fraccionamiento`, `Patrón temporal`, `Monto atípico` | (todos) |
| `level`     | `high` (→ alto) · `medium` (→ medio) | (todos) |
| `page`      | nº de página (≥ 1) | `1` |
| `page_size` | tamaño de página (1–500) | `50` |

Ejemplos:

```
GET /alerts?level=high&page=1&page_size=20
GET /alerts?type=Posible%20fraccionamiento
GET /alerts?level=medium&type=IGV%20inconsistente&page=2
```

> Nota: todas las respuestas son `application/json` en **UTF-8** (las tildes y
> la "ñ" se devuelven correctamente).
