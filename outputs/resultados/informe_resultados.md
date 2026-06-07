# Informe de Resultados — Módulo A: Pronóstico del Gasto en Órdenes de Compra

_Sistema de Predicción de Órdenes de Compra (SistemaPrediccionOC). Generado el 2026-06-07 00:09._

## 1. Introducción

Este informe documenta el **Módulo A** del sistema: el pronóstico del gasto público mensual en las **Órdenes de Compra de los Catálogos Electrónicos de Acuerdos Marco** de la Central de Compras Públicas del Perú (PERÚ COMPRAS). Cada registro es una orden de compra de bienes emitida por una entidad pública a un proveedor, con su monto y convenio.

La variable a pronosticar es el **gasto total mensual** (suma de `TOTAL`), agregado por **`FECHA_FORMALIZACION`**. El informe reúne el análisis exploratorio con sus interpretaciones y la comparación de modelos con su conclusión.

Los datos abarcan **53 meses** (de 202201 a 202605) e incluyen **483,889 órdenes** consolidadas desde 53 archivos CSV mensuales.

## 2. Análisis Exploratorio de Datos (EDA)
### 2.1 Resumen general
- **Archivos leídos:** 53 de 53 encontrados (omitidos: ninguno).
- **Meses cubiertos:** 53 (de 202201 a 202605).
- **Codificaciones detectadas:** {'latin-1': 52, 'utf-8': 1}.
- **Filas consolidadas (crudo):** 483,889.
- **Filas válidas tras limpieza:** 483,692.
- **Entidades distintas:** 3,244 | **Proveedores distintos:** 3,586 | **Categorías (ACUERDO_MARCO):** 72.

### 2.2 Calidad de los datos
- **Duplicados exactos:** 0. **Duplicados por ORDEN_ELECTRONICA:** 0.
- **Órdenes sin fecha de formalización (descartadas):** 0.
- **Órdenes excluidas por estado** ('RESUELTA', 'ORDEN DE COMPRA NULA'): 197.
- **Consistencia contable:** se verificó que TOTAL = SUB_TOTAL + IGV.
- **Columnas con valores nulos (solo auxiliares/enlaces):**
    - `DESCRIPCION_CESION_DERECHOS`: 100.0%
    - `DOCUMENTO_ESTADO_OCAM`: 99.41%
    - `DESCRIPCION_ESTADO`: 81.24%
- **Outliers de monto** (criterio IQR×3, límite ≈ S/ 29,310): 42,048 órdenes por encima; máximo S/ 44,580,303. No se eliminan: son grandes compras reales.

### 2.3 Figuras e interpretaciones

**Figura 1. Evolucion gasto ordenes**

![Evolucion gasto ordenes](../figuras/01_evolucion_gasto_ordenes.png)

_La serie cubre 53 meses. El gasto total acumulado es de **S/ 8,078 millones**, con un promedio de **S/ 152.4 millones/mes**. El mes de mayor gasto fue 2023-11 (S/ 276.7M) y el de menor 2026-01 (S/ 9.7M). Se observa una caída pronunciada y recurrente cada inicio de año (enero), coherente con el calendario de ejecución presupuestal pública peruana, y una recuperación entre marzo y diciembre. El número de órdenes (barras) acompaña al gasto, aunque en los últimos años el gasto se sostiene con menos órdenes (ticket promedio creciente)._

**Figura 2. Distribucion total**

![Distribucion total](../figuras/02_distribucion_total.png)

_El monto por orden está **fuertemente sesgado a la derecha**: la mediana es **S/ 2,690** pero la media es **S/ 16,701**, señal de una cola larga de grandes compras (máximo S/ 44,580,303). El histograma en escala logarítmica muestra una forma aproximadamente acampanada, típica de montos con distribución log-normal. Esta asimetría justifica analizar el GASTO AGREGADO mensual (más estable) en lugar de los montos individuales para el pronóstico._

**Figura 3. Gasto por categoria**

![Gasto por categoria](../figuras/03_gasto_por_categoria.png)

_Existen **72 valores distintos de ACUERDO_MARCO** (con variaciones de etiqueta entre años: algunos llevan el código del convenio, p. ej. 'EXT-CE-2021-7', y otros solo la descripción). Las 12 categorías mostradas concentran el **82.6%** del gasto. Dominan los bienes de oficina —útiles de escritorio, impresoras y consumibles— seguidos de materiales de limpieza y equipos de cómputo. Esto orienta una eventual serie por categoría, aunque el foco del Módulo A es el gasto total._

**Figura 4. Gasto por tipo procedimiento**

![Gasto por tipo procedimiento](../figuras/04_gasto_por_tipo_procedimiento.png)

_En **gasto**, el tipo de procedimiento que más concentra es **'Gran Compra'** (58% del total), mientras que en **número de órdenes** domina **'Compra ordinaria'** (93% de las órdenes). Es decir, hay muchas órdenes ordinarias de monto moderado y pocas 'Gran Compra' de ticket muy elevado; estas últimas, al ser grandes y esporádicas, pueden introducir saltos puntuales en el gasto mensual y conviene tenerlas presentes al interpretar picos de la serie._

**Figura 5. Top entidades proveedores**

![Top entidades proveedores](../figuras/05_top_entidades_proveedores.png)

_Hay **3,244 entidades** compradoras y **3,586 proveedores** distintos. El gasto está relativamente concentrado: la entidad líder es 'SEGURO SOCIAL DE SALUD' y el proveedor líder es 'OK COMPUTER E.I.R.L.'. Esta concentración es relevante para el futuro módulo de detección de anomalías, pero no condiciona el pronóstico agregado._

**Figura 6. Estacionalidad por mes**

![Estacionalidad por mes](../figuras/06_estacionalidad_por_mes.png)

_El boxplot por mes calendario confirma una **estacionalidad anual marcada y consistente**: **Ene** es sistemáticamente el mes de menor gasto (S/ 16.1M en promedio) y **Nov** uno de los más altos (S/ 237.4M). Como el patrón se repite en todos los años, NO se trata de meses incompletos sino de estacionalidad real. Esto favorece modelos con componente estacional de periodo 12 (SARIMA, naive estacional, variables de mes)._

**Figura 7. Descomposicion serie**

![Descomposicion serie](../figuras/07_descomposicion_serie.png)

_La descomposición separa tres componentes. La **tendencia** es estable o descendente a lo largo del periodo. La **componente estacional** tiene una amplitud de S/ 221.8M entre el mes más bajo y el más alto, confirmando la estacionalidad anual. El **residuo** recoge el ruido y eventos puntuales (p. ej. grandes compras), sin un patrón evidente, lo que indica que tendencia + estacionalidad explican buena parte de la variación._

**Figura 8. Deteccion meses incompletos**

![Deteccion meses incompletos](../figuras/08_deteccion_meses_incompletos.png)

_La detección de incompletitud compara, mes a mes desde el final, el número de órdenes contra el mismo mes del año anterior (controlando la estacionalidad). **Ningún mes de la cola resultó incompleto** (último mes 2026-05: 1.12× respecto al año previo): el conteo de órdenes del último mes es comparable al del mismo mes del año anterior, por lo que se conservan todos los meses. La fuerte caída de enero NO se recorta porque se repite cada año (estacionalidad real, no falta de datos)._


## 3. Modelado del pronóstico
Se reservaron los **últimos 6 meses** como conjunto de prueba (holdout) y se entrenó con los anteriores. Se compararon 5 modelos: un naive simple y un naive estacional (referencias), un modelo estadístico SARIMA, un modelo de árboles XGBoost con variables temporales y una red neuronal LSTM. Todas las semillas se fijaron en 42 para reproducibilidad.

El SARIMA seleccionado automáticamente por AIC fue **SARIMA(1, 0, 0)x(0, 1, 0, 12)**.

### 3.1 Comparación de modelos (holdout)

| Modelo | MAE (S/) | RMSE (S/) | MAPE (%) |
|---|---|---|---|
| SARIMA ⭐ | 16,645,734 | 20,131,214 | 30.8 |
| Naive estacional | 18,822,269 | 24,562,604 | 33.1 |
| XGBoost | 29,297,718 | 32,753,480 | 50.2 |
| LSTM | 72,282,351 | 88,277,845 | 81.8 |
| Naive | 113,419,603 | 124,580,430 | 435.0 |

![Comparación en el holdout](../figuras/09_holdout_real_vs_modelos.png)

_El mejor modelo es **SARIMA** (menor RMSE). Sobre el holdout alcanza un MAPE de **30.8%** y un MAE de **S/ 16.65 millones**, equivalente a un **15.3%** del gasto mensual promedio del periodo de prueba (S/ 109.0M). En términos prácticos, el modelo anticipa el gasto mensual con un error típico de ese orden, suficiente para planificación presupuestal agregada aunque no para el detalle de una orden individual._

### 3.2 Pronóstico hacia adelante
Con **SARIMA** reentrenado sobre toda la serie se pronostican los próximos **6 meses** (2026-06 a 2026-11), con intervalo de confianza al 95%.

| Mes | Pronóstico (S/) | Límite inferior | Límite superior |
|---|---|---|---|
| 2026-06 | 131,392,861 | 79,835,591 | 216,245,458 |
| 2026-07 | 161,601,233 | 97,332,717 | 268,306,065 |
| 2026-08 | 129,890,892 | 78,209,353 | 215,724,120 |
| 2026-09 | 184,289,296 | 110,962,271 | 306,072,905 |
| 2026-10 | 200,796,261 | 120,901,219 | 333,488,271 |
| 2026-11 | 234,933,174 | 141,455,357 | 390,183,855 |

![Pronóstico final](../figuras/10_pronostico_final.png)

_Se proyecta un gasto acumulado de **S/ 1,043 millones** en los próximos 6 meses. El pronóstico reproduce el patrón estacional histórico (meses altos y bajos) y el intervalo de confianza refleja la incertidumbre: cuanto más ancho, mayor variabilidad esperada. Conviene recalibrar el modelo a medida que ingresen nuevos meses de datos._

## 4. Conclusiones del Módulo A
- El gasto en órdenes de compra de los Acuerdos Marco muestra una **estacionalidad anual fuerte** (mínimos en enero–febrero) y una tendencia de fondo identificable.
- Entre los modelos probados, **SARIMA** ofreció el mejor equilibrio en el holdout (MAPE 30.8%).
- La principal limitación es la **longitud de la serie** (pocos años de historia mensual), que restringe especialmente a la LSTM; con más historia el desempeño podría mejorar.
- El pipeline es **reproducible** (semillas fijas, rutas relativas) y deja listos los artefactos para los siguientes módulos del sistema.
