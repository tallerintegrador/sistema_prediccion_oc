# Frontend — Sistema de Predicción OC

Tablero web (React + Vite) que consume el API (FastAPI) del proyecto
`SistemaPrediccionOC`. Muestra los indicadores clave, el pronóstico del gasto,
el gasto por categoría y las alertas de órdenes anómalas.

> Arquitectura: **Frontend (React)** ⟷ **API (FastAPI)** ⟷ **Backend/Módulos**.
> El frontend NO tiene datos quemados: todo proviene del API por HTTP.

## Requisitos

- Node.js 18 o superior.
- El **backend corriendo** (ver más abajo).

## Configuración

La URL del API se define en el archivo `.env` (ya incluido):

```env
VITE_API_URL=http://127.0.0.1:8000
```

Cámbiala si tu backend corre en otra dirección.

## Puesta en marcha

```bash
# 1) Instalar dependencias (solo la primera vez)
npm install

# 2) Levantar el servidor de desarrollo
npm run dev
```

El tablero queda disponible en **http://localhost:5173**.

> ⚠️ Recuerda tener el **backend corriendo en paralelo**. Desde la raíz del
> proyecto:
>
> ```bash
> uvicorn backend.app.main:app --reload
> ```

## Estructura

```
frontend/
├── .env                      # VITE_API_URL (URL base del API)
├── index.html
├── vite.config.js
└── src/
    ├── main.jsx              # Punto de entrada
    ├── App.jsx               # Layout del tablero
    ├── api/
    │   └── client.js         # Capa única de llamadas al API (fetch)
    ├── hooks/
    │   └── useApi.js         # Hook de carga/error/datos
    ├── utils/
    │   └── format.js         # Formato de moneda, números y fechas (es-PE)
    ├── components/
    │   ├── Estado.jsx              # Cargando / Error / Tarjeta
    │   ├── TarjetasIndicadores.jsx # /summary  → KPIs
    │   ├── GraficoPronostico.jsx   # /forecast → línea + intervalo
    │   ├── GraficoCategorias.jsx   # /categories → barras
    │   └── TablaAlertas.jsx        # /alerts → tabla con filtros
    └── styles/
        └── index.css
```

## Build de producción

```bash
npm run build     # genera dist/
npm run preview   # sirve dist/ localmente para revisarlo
```
