// =============================================================================
// MetricasML.jsx — Página de Métricas del ML (sustentación / capstone).
//
// Consolida TODO lo que el ML evaluó:
//   Módulo A (pronóstico): comparación de modelos, mejor modelo, WAPE/MASE.
//   Módulo B (anomalías):  precision@k, recall por inyección, concordancia,
//                          distribución por tipo, estadísticos de red.
//   + Galería de figuras generadas por los pipelines (servidas por el API).
//
// Datos: /metrics/forecast, /metrics/anomalies (ver api/client.js).
// =============================================================================

import { obtenerMetricasAnomalias, obtenerMetricasPronostico } from '../api/client'
import { useApi } from '../hooks/useApi'
import { Cargando, MensajeError, Tarjeta } from '../components/Estado'
import TablaModelos from '../components/TablaModelos'
import GraficoBarrasMetrica from '../components/GraficoBarrasMetrica'
import GraficoPrecisionK from '../components/GraficoPrecisionK'
import GraficoRecallInyeccion from '../components/GraficoRecallInyeccion'
import HeatmapConcordancia from '../components/HeatmapConcordancia'
import DistribucionTipos from '../components/DistribucionTipos'
import GaleriaFiguras from '../components/GaleriaFiguras'
import { formatoEntero, formatoNumero, formatoPorcentaje, formatoPct } from '../utils/format'

// Figuras relevantes para la sustentación de métricas (generadas por el ML).
const FIGURAS = [
  { archivo: '09_holdout_real_vs_modelos.png', titulo: 'A · Holdout: real vs. modelos' },
  { archivo: '10_backtest_error_por_mes.png', titulo: 'A · Error de backtest por mes' },
  { archivo: '11_pronostico_final.png', titulo: 'A · Pronóstico final' },
  { archivo: 'b06_resultados_riesgo.png', titulo: 'B · Distribución del riesgo' },
  { archivo: 'b07_anomalias_sinteticas.png', titulo: 'B · Inyección sintética (recall)' },
  { archivo: 'b08_concordancia.png', titulo: 'B · Concordancia entre métodos' },
]

/** Tarjeta KPI pequeña y reutilizable. */
function Kpi({ etiqueta, valor, detalle, tono = 'azul' }) {
  return (
    <div className={`kpi kpi--${tono}`}>
      <div className="kpi__texto">
        <span className="kpi__etiqueta">{etiqueta}</span>
        <span className="kpi__valor">{valor}</span>
        {detalle && <span className="kpi__detalle">{detalle}</span>}
      </div>
    </div>
  )
}

export default function MetricasML() {
  const a = useApi(obtenerMetricasPronostico, [])
  const b = useApi(obtenerMetricasAnomalias, [])

  return (
    <>
      {/* ============================ MÓDULO A ============================ */}
      <Tarjeta
        titulo="Módulo A — Pronóstico del gasto"
        subtitulo="Comparación de modelos por backtesting de origen móvil (menor es mejor)"
      >
        {a.cargando && <Cargando texto="Cargando métricas del pronóstico…" />}
        {a.error && <MensajeError mensaje={a.error} onReintentar={a.recargar} />}
        {a.datos && <SeccionA datos={a.datos} />}
      </Tarjeta>

      {/* ============================ MÓDULO B ============================ */}
      <Tarjeta
        titulo="Módulo B — Detección de anomalías"
        subtitulo="Evaluación sin etiquetas: precision@k, recall por inyección y concordancia"
      >
        {b.cargando && <Cargando texto="Cargando métricas de anomalías…" />}
        {b.error && <MensajeError mensaje={b.error} onReintentar={b.recargar} />}
        {b.datos && <SeccionB datos={b.datos} />}
      </Tarjeta>

      {/* ============================ FIGURAS ============================= */}
      <Tarjeta titulo="Figuras del ML" subtitulo="Salidas visuales de los pipelines (clic para ampliar)">
        <GaleriaFiguras figuras={FIGURAS} />
      </Tarjeta>
    </>
  )
}

function SeccionA({ datos }) {
  const mejor = datos.best_model
  const filaMejor = (datos.model_comparison || []).find((m) => m.Modelo === mejor) || {}

  return (
    <>
      <div className="kpis">
        <Kpi etiqueta="Mejor modelo" valor={mejor || '—'} detalle={`métrica primaria: ${datos.primary_metric}`} tono="verde" />
        <Kpi etiqueta={`${datos.primary_metric} (mejor)`} valor={formatoPct(filaMejor.WAPE)} detalle="error porcentual ponderado" tono="azul" />
        <Kpi etiqueta="MASE (mejor)" valor={formatoNumero(filaMejor.MASE, 3)} detalle="<1 vence al naive estacional" tono="indigo" />
        <Kpi etiqueta="Modelos comparados" valor={formatoEntero((datos.model_comparison || []).length)} detalle={`IC ${formatoPorcentaje(datos.confidence_level, 0)}`} tono="ambar" />
      </div>

      <h3 className="subtitulo-seccion">Comparación de modelos</h3>
      <TablaModelos modelos={datos.model_comparison} mejor={mejor} />

      <div className="grid-2">
        <div>
          <h3 className="subtitulo-seccion">WAPE por modelo (%)</h3>
          <GraficoBarrasMetrica modelos={datos.model_comparison} metrica="WAPE" mejor={mejor} sufijo="%" />
        </div>
        <div>
          <h3 className="subtitulo-seccion">MASE por modelo</h3>
          <GraficoBarrasMetrica modelos={datos.model_comparison} metrica="MASE" mejor={mejor} color="#7c3aed" decimales={3} />
        </div>
      </div>
    </>
  )
}

function SeccionB({ datos }) {
  const net = datos.network_stats || {}
  const al = datos.alertas || {}
  const rg = datos.recall_global

  return (
    <>
      <div className="kpis">
        <Kpi etiqueta="Órdenes analizadas" valor={formatoEntero(datos.n_ordenes)} detalle={`${formatoEntero(datos.n_entidades)} entidades · ${formatoEntero(datos.n_proveedores)} proveedores`} tono="azul" />
        <Kpi etiqueta="Alertas" valor={formatoEntero(al.total)} detalle={`${formatoEntero(al.alto)} alto · ${formatoEntero(al.medio)} medio`} tono="ambar" />
        {rg && <Kpi etiqueta="Recall global" valor={formatoPorcentaje(rg.recall, 0)} detalle={`${formatoEntero(rg.n)} casos inyectados`} tono="verde" />}
        <Kpi etiqueta="Red entidad–proveedor" valor={`${formatoEntero(net.nodos)} nodos`} detalle={`${formatoEntero(net.aristas)} aristas · ${formatoEntero(net.comunidades)} comunidades · ${formatoEntero(net.proveedores_cautivos)} cautivos`} tono="indigo" />
      </div>

      <div className="grid-2">
        <div>
          <h3 className="subtitulo-seccion">precision@k (validez aparente)</h3>
          <GraficoPrecisionK datos={datos.precision_at_k || []} />
        </div>
        <div>
          <h3 className="subtitulo-seccion">Recall por inyección sintética</h3>
          <GraficoRecallInyeccion datos={datos.synthetic_injection_recall || []} />
        </div>
      </div>

      <div className="grid-2">
        <div>
          <h3 className="subtitulo-seccion">Concordancia entre métodos (Jaccard)</h3>
          <HeatmapConcordancia concordancia={datos.concordance} />
        </div>
        <div>
          <h3 className="subtitulo-seccion">Alertas por tipo de anomalía</h3>
          <DistribucionTipos distribucion={datos.type_distribution || {}} />
        </div>
      </div>
    </>
  )
}
