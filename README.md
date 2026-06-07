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

```
SistemaPrediccionOC/
├── data/                      # CSVs originales por año (no se modifican)
├── src/
│   ├── config.py              # rutas y parámetros centrales
│   ├── ingesta.py             # lectura y consolidación de los CSV (ETL)
│   ├── limpieza.py            # limpieza y validación
│   ├── serie_temporal.py      # construcción de la serie mensual
│   ├── eda.py                 # análisis exploratorio y figuras
│   ├── modelos.py             # naive, SARIMA, XGBoost, LSTM
│   └── evaluacion.py          # métricas, comparación y pronóstico final
├── notebooks/
│   └── flujo_completo.ipynb   # (opcional) recorrido guiado del flujo
├── outputs/
│   ├── figuras/               # todas las gráficas (.png)
│   ├── modelos/               # modelo entrenado serializado
│   └── resultados/            # datasets, métricas, pronóstico e informe
├── main.py                    # ejecuta el flujo completo de punta a punta
├── requirements.txt
└── README.md
```

---

## 4. Requisitos

- **Python 3.11+** (desarrollado y probado en Python 3.14, Windows 11).
- Las librerías de `requirements.txt` (pandas, numpy, matplotlib, seaborn,
  scikit-learn, statsmodels, xgboost, scipy, torch, pyarrow).

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

`main.py` ejecuta en orden: **ingesta → limpieza → serie temporal → EDA →
modelos → evaluación**, e imprime un resumen final. Al terminar, todos los
artefactos quedan en `outputs/`.

> También puede ejecutarse cada etapa por separado, p. ej.
> `python -m src.ingesta` o `python -m src.limpieza`.

### Salidas generadas

- `outputs/figuras/` — 10 gráficas del EDA y del pronóstico.
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

Se reservaron los **últimos 6 meses** como prueba (holdout) y se compararon cinco
modelos. Métricas sobre el holdout:

| Modelo | MAE (S/) | RMSE (S/) | MAPE (%) |
|---|---|---|---|
| **SARIMA** ⭐ | 16.6 M | 20.1 M | **30.8** |
| Naive estacional | 18.8 M | 24.6 M | 33.1 |
| XGBoost | 29.3 M | 32.8 M | 50.2 |
| LSTM | 72.3 M | 88.3 M | 81.8 |
| Naive | 113.4 M | 124.6 M | 435.0 |

**Conclusiones:**

- El mejor modelo es **SARIMA** (menor RMSE y MAPE), que captura bien la
  estacionalidad anual. Supera incluso al *naive estacional*, una referencia
  exigente en esta serie.
- El **naive simple** falla por completo (MAPE 435%): al repetir el último valor,
  no anticipa la caída de enero. Esto evidencia la importancia del componente
  estacional.
- **XGBoost** y **LSTM** quedan por detrás: con solo ~4 años de historia mensual
  hay muy pocos ejemplos para que un árbol de gradiente o una red neuronal
  generalicen; es una limitación de **datos**, no de implementación.
- El **pronóstico a 6 meses** (jun–nov 2026) proyecta un gasto acumulado de
  **~S/ 1,043 millones**, reproduciendo el patrón estacional, con su intervalo de
  confianza al 95%.

En términos prácticos, el error del mejor modelo (~15% del gasto mensual promedio
en MAE) es **adecuado para planificación presupuestal agregada**, no para predecir
órdenes individuales. El desempeño debería mejorar conforme se acumule más
historia mensual.

> El detalle completo, con todas las figuras e interpretaciones, está en
> [`outputs/resultados/informe_resultados.md`](outputs/resultados/informe_resultados.md).

---

## 7. Reproducibilidad

- **Semillas fijas** (`config.SEMILLA = 42`) en numpy, random, torch y XGBoost.
- **Rutas relativas** calculadas desde el propio proyecto (`src/config.py`).
- Flujo único (`python main.py`) que regenera todos los artefactos desde los CSV.
