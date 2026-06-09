// =============================================================================
// TarjetasIndicadores.jsx — Tarjetas con los indicadores clave (KPIs).
//
// Consume /summary y muestra: gasto del último mes, órdenes del mes, alertas
// detectadas y el pronóstico del próximo mes. Debajo, una franja con metadatos
// (modelo usado, rango de datos y número de categorías).
// =============================================================================

import { obtenerResumen } from '../api/client'
import { useApi } from '../hooks/useApi'
import {
  formatoSoles,
  formatoSolesCompacto,
  formatoEntero,
  mesLargo,
} from '../utils/format'
import { Cargando, MensajeError } from './Estado'

/** Tarjeta individual de KPI: valor grande + etiqueta + detalle. */
function Kpi({ etiqueta, valor, detalle, icono, acento }) {
  return (
    <div className={`kpi kpi--${acento}`}>
      <div className="kpi__icono" aria-hidden="true">{icono}</div>
      <div className="kpi__contenido">
        <p className="kpi__etiqueta">{etiqueta}</p>
        <p className="kpi__valor">{valor}</p>
        {detalle && <p className="kpi__detalle">{detalle}</p>}
      </div>
    </div>
  )
}

export default function TarjetasIndicadores() {
  const { datos, cargando, error, recargar } = useApi(obtenerResumen, [])

  if (cargando) return <Cargando texto="Cargando indicadores…" />
  if (error) return <MensajeError mensaje={error} onReintentar={recargar} />

  const fc = datos.next_month_forecast

  return (
    <>
      <div className="kpis">
        <Kpi
          acento="azul"
          icono="S/"
          etiqueta={`Gasto en ${mesLargo(datos.last_period)}`}
          valor={formatoSoles(datos.last_period_spend)}
          detalle={`Acumulado histórico: ${formatoSolesCompacto(datos.total_spend)}`}
        />
        <Kpi
          acento="indigo"
          icono="🧾"
          etiqueta={`Órdenes en ${mesLargo(datos.last_period)}`}
          valor={formatoEntero(datos.last_period_orders)}
          detalle={`Total histórico: ${formatoEntero(datos.total_orders)} órdenes`}
        />
        <Kpi
          acento="ambar"
          icono="⚠️"
          etiqueta="Alertas detectadas"
          valor={formatoEntero(datos.alerts.total)}
          detalle={`${formatoEntero(datos.alerts.high)} de nivel alto · ${formatoEntero(datos.alerts.medium)} medio`}
        />
        <Kpi
          acento="verde"
          icono="📈"
          etiqueta={`Pronóstico para ${mesLargo(fc.fecha)}`}
          valor={formatoSoles(fc.pred)}
          detalle={`Intervalo 95%: ${formatoSolesCompacto(fc.lower)} – ${formatoSolesCompacto(fc.upper)}`}
        />
      </div>

      <p className="kpis__meta">
        Modelo de pronóstico: <strong>{datos.best_model ?? 'No disponible'}</strong>
        {' · '}Categorías analizadas: <strong>{datos.n_categories}</strong>
        {' · '}Datos del <strong>{mesLargo(datos.data_range.start)}</strong> al{' '}
        <strong>{mesLargo(datos.data_range.end)}</strong>
      </p>
    </>
  )
}
