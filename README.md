# SistemaPrediccionOC — Módulo A: Pronóstico del Gasto en Órdenes de Compra

Sistema de análisis y predicción sobre las **Órdenes de Compra de los Catálogos
Electrónicos de Acuerdos Marco** de la Central de Compras Públicas del Perú
(**PERÚ COMPRAS**).

Este repositorio implementa el **Módulo A**: el **pronóstico del gasto público
mensual** en órdenes de compra de bienes, a partir de datos históricos reales
(2022–2026). Otros módulos previstos (p. ej. detección de anomalías) **no** forman
parte de esta fase.

---

## 1. Descripción del proyecto y del módulo

Cada registro de los datos es **una orden de compra de bienes** emitida por una
entidad pública a un proveedor bajo un convenio (acuerdo marco): contiene quién
compra, a quién, cuándo, por qué monto y bajo qué convenio. Los datos están a
**nivel de orden completa** (no de producto individual).

El **Módulo A** construye, a partir de esos datos:

1. Un proceso de **ingesta y consolidación** (ETL) de todos los CSV mensuales.
2. Una etapa de **limpieza y validación**.
3. Un **análisis exploratorio de datos (EDA)** con interpretaciones escritas.
4. La **serie temporal mensual** del gasto total (variable a pronosticar).
5. El **entrenamiento y comparación de varios modelos** de pronóstico.
6. La **evaluación** (MAE, RMSE, MAPE) y el **pronóstico final** con intervalo de
   confianza, con sus visualizaciones e interpretaciones.

La variable objetivo es el **gasto total mensual** = suma de `TOTAL`, agregada por
**`FECHA_FORMALIZACION`** (fecha de aceptación/formalización de la orden).

---

## 2. Significado de los datos

Los archivos provienen de reportes mensuales (`ReportePCBienesAAAAMM.csv`).
Columnas principales:

| Columna | Significado |
|---|---|
| `FECHA_PROCESO` | Fecha en que se generó el registro/reporte. |
| `RUC_PROVEEDOR` / `PROVEEDOR` | RUC y razón social del proveedor. |
| `RUC_ENTIDAD` / `ENTIDAD` | RUC y razón social de la entidad pública compradora. |
| `TIPO_PROCEDIMIENTO` | Tipo de compra ("Compra ordinaria" o "Gran Compra"). |
| `ORDEN_ELECTRONICA` | Identificador único de la orden de compra. |
| `ESTADO_ORDEN_ELECTRONICA` | Estado de la orden (ACEPTADA, PAGADA, RESUELTA, …). |
| `FECHA_FORMALIZACION` | Fecha de formalización. **Eje temporal del pronóstico.** |
| `FECHA_ULTIMO_ESTADO` | Fecha del último estado. |
| `SUB_TOTAL` | Monto antes de IGV (incluye envío), en soles. |
| `IGV` | Impuesto General a las Ventas, en soles. |
| `TOTAL` | Monto total (`SUB_TOTAL` + `IGV`). **Variable a pronosticar (agregada).** |
| `ACUERDO_MARCO` | Convenio/categoría de bienes (sirve como categoría). |

Otras columnas (`ORDEN_ELECTRONICA_GENERADA`, `DOCUMENTO_ESTADO_OCAM`,
`ORDEN_DIGITALIZADA`, `DESCRIPCION_ESTADO`, `DESCRIPCION_CESION_DERECHOS`) son
enlaces o descripciones auxiliares, en su mayoría vacías, y no se usan en el
pronóstico.

### Hechos descubiertos al explorar los datos reales

> Nada se asumió: todo se verificó leyendo los archivos.

- **53 archivos** CSV (subcarpetas 2022–2026; 2026 llega hasta mayo) con
  **483,889 órdenes** consolidadas.
- **Separador `;`** y **codificación mixta**: 52 archivos en `latin-1` y **1 en
  `utf-8`** (`2024/ReportePCBienes202403_0.csv`). La codificación se detecta por
  archivo.
- **Suciedad real corregida:** tabuladores (`\t`) al inicio de campos en el
  archivo UTF-8, y **dos formatos de fecha** (`YYYY-MM-DD HH:MM:SS` y, solo en
  `202406.csv`, `DD/MM/YYYY HH:MM`). Tras limpiar: **0 fechas nulas**.
- **Calidad:** sin duplicados; `TOTAL = SUB_TOTAL + IGV` exacto; columnas clave
  sin nulos. Variaciones de etiqueta en nombres y categorías (espacios/tildes) se
  normalizan.
- **Estacionalidad anual fuerte:** enero–febrero son sistemáticamente bajos todos
  los años (calendario de ejecución presupuestal), no por falta de datos.

---

## 3. Estructura del repositorio

La arquitectura está organizada **por fases**: una fase por archivo, numeradas en
orden de flujo dentro de `src/fases/`. Un orquestador (`src/pipeline.py`) las
ejecuta en orden, y un registro central (`src/figuras.py`) numera y ordena todas
las figuras.

```
SistemaPrediccionOC/
├── data/                       # CSVs originales por año (no se modifican)
├── src/
│   ├── config.py               # rutas y parámetros centrales (lo usa todo)
│   ├── figuras.py              # registro central de figuras (estilo + numeración 01..11)
│   ├── pipeline.py             # ORQUESTADOR: define y ejecuta las 7 fases en orden
│   └── fases/                  # una fase por archivo, en orden de flujo:
│       ├── f01_ingesta.py          # 1. extracción/ETL: lectura y consolidación de los CSV
│       ├── f02_limpieza.py         # 2. limpieza y validación
│       ├── f03_serie_temporal.py   # 3. serie mensual + panel por categoría + drivers
│       ├── f04_eda.py              # 4. análisis exploratorio y figuras 01–08
│       ├── f05_features.py         # 5. features de calendario (días hábiles, feriados, Fourier)
│       ├── f06_modelado.py         # 6. naive(+drift), SARIMA, ETS, Ensemble, XGBoost/LightGBM, descompuesto, LSTM, Transformer
│       └── f07_evaluacion.py       # 7. métricas (WAPE/MASE/…), backtest de origen móvil y pronóstico
├── notebooks/
│   └── flujo_completo.ipynb    # recorrido guiado de las 7 fases (figuras inline)
├── outputs/
│   ├── figuras/                # todas las gráficas (.png) numeradas 01..11
│   ├── modelos/                # modelo entrenado serializado
│   └── resultados/             # datasets, métricas, pronóstico e informe
├── main.py                     # punto de entrada: invoca pipeline.ejecutar_pipeline()
├── requirements.txt
└── README.md
```

---

## 4. Requisitos

- **Python 3.11+** (desarrollado y probado en Python 3.14, Windows 11).
- Las librerías de `requirements.txt` (pandas, numpy, matplotlib, seaborn,
  scikit-learn, statsmodels, xgboost, lightgbm, optuna, holidays, scipy, torch,
  pyarrow).

---

## 5. Instalación y ejecución paso a paso

```bash
# 1) Clonar el repositorio y entrar en la carpeta
git clone <URL-del-repo>
cd SistemaPrediccionOC

# 2) (Recomendado) crear y activar un entorno virtual
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux/Mac:
# source .venv/bin/activate

# 3) Instalar dependencias
pip install -r requirements.txt

# 4) Verificar que los datos están en data/  (subcarpetas 2022..2026 con los CSV)

# 5) Ejecutar el flujo completo
python main.py
```

`main.py` ejecuta el pipeline en **7 fases ordenadas**: **ingesta → limpieza →
serie temporal → EDA → features → modelado → evaluación**, imprime un **resumen
por fase** en consola y un resumen final. Al terminar, todos los artefactos quedan
en `outputs/`.

```
[FASE 1/7] Ingesta / ETL  (2.1 s)
   Extracción: lectura y consolidación de los CSV mensuales.
     - Archivos leídos     : 53/53 (omitidos: 0)
     - Filas consolidadas  : 483,889
     ...
[FASE 7/7] Evaluación y pronóstico  (...)
     - Mejor modelo        : Ensemble  (WAPE 16.5% · MASE 0.75)
```

> Cada fase también puede ejecutarse de forma aislada, p. ej.
> `python -m src.fases.f01_ingesta` o `python -m src.fases.f02_limpieza`.

### Salidas generadas

- `outputs/figuras/` — 11 gráficas numeradas (01–08 EDA; 09–11 evaluación y
  pronóstico), ordenadas por el registro central `src/figuras.py`.
- `outputs/resultados/informe_resultados.md` — **informe completo** (EDA +
  comparación de modelos + conclusiones).
- `outputs/resultados/comparacion_modelos.csv` — tabla de métricas.
- `outputs/resultados/pronostico_final.csv` — pronóstico con intervalo.
- `outputs/resultados/dataset_consolidado.parquet` y `dataset_limpio.parquet`.
- `outputs/resultados/serie_mensual_total.csv` y `serie_mensual_por_categoria.csv`.
- `outputs/modelos/` — mejor modelo entrenado serializado.

---

## 6. Resumen de resultados y conclusiones del Módulo A

Sobre los **53 meses** (ene-2022 a may-2026), el gasto total acumulado es de
**~S/ 8,078 millones** (~**S/ 152 millones/mes** en promedio). La serie tiene una
**estacionalidad anual marcada** (enero el más bajo) y una tendencia de fondo
identificable.

La evaluación usa **validación de origen móvil** (rolling-origin, 6 orígenes,
horizonte 6 meses) y como métrica primaria el **WAPE** (error absoluto ponderado
por monto) y el **MASE**, ambos robustos al *trough* de enero (gasto casi-cero
donde el MAPE explota). Métricas medias del backtest (extracto):

| Modelo | WAPE (%) | MASE | MAPE (%) |
|---|---|---|---|
| **Ensemble** ⭐ (mediana ETS+drift+SARIMA) | **16.5** | **0.75** | 28.0 |
| ETS (Holt-Winters) | 17.4 | 0.78 | 24.4 |
| Estacional drift | 18.0 | 0.84 | 29.4 |
| Naive estacional | 19.0 | 0.86 | 32.3 |
| SARIMA | 19.5 | 0.89 | 33.0 |
| LightGBM (calendario, Tweedie) | 20.9 | 0.93 | 46.7 |
| LSTM | 38.9 | 1.66 | 66.2 |
| Transformer | 47.5 | 2.05 | 206.5 |

**Conclusiones:**

- El mejor modelo es un **Ensemble** (mediana de ETS, naive-estacional-con-drift y
  SARIMA): **WAPE 16.5%, MASE 0.75** (un MASE < 1 supera al naive estacional).
- **Los modelos complejos pierden frente a los simples** (LightGBM/XGBoost y, sobre
  todo, LSTM/Transformer): con ~50 puntos mensuales hay inanición de datos. No es
  un problema de implementación sino de **información disponible**.
- **Granularidad fina no ayuda**: modelar a nivel diario o semanal y agregar a
  mensual da WAPE 28.8% / 35.1% (peor), porque la agregación reacumula la
  incertidumbre. Útil para el calendario, no para bajar el error mensual.
- **El objetivo de error < 10% no se alcanza** con los datos disponibles (techo
  empírico ~16.5%, igual en mensual/semanal/diario). Para romperlo hace falta más
  **señal**: variables exógenas (presupuesto PIA/PIM, calendario de ejecución),
  más historia o un modelo global por categoría. Detalle en el informe.
- El **pronóstico a 6 meses** (jun–nov 2026) proyecta ~**S/ 1,038 millones**, con
  intervalo de confianza al 95%.

> El detalle completo, con todas las figuras e interpretaciones, está en
> [`outputs/resultados/informe_resultados.md`](outputs/resultados/informe_resultados.md).

---

## 7. Reproducibilidad

- **Semillas fijas** (`config.SEMILLA = 42`) en numpy, random, torch y XGBoost.
- **Rutas relativas** calculadas desde el propio proyecto (`src/config.py`).
- Flujo único (`python main.py`) que regenera todos los artefactos desde los CSV.
