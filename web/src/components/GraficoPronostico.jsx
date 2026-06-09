// =============================================================================
// GraficoPronostico.jsx — Histórico mensual + pronóstico + intervalo de confianza.
//
// Consume /forecast y dibuja, en un mismo gráfico:
//   1) La serie histórica observada (línea sólida).
//   2) El pronóstico de los próximos meses (línea punteada).
//   3) El intervalo de confianza del pronóstico (banda sombreada lower–upper).
// Las tres capas se distinguen por color y estilo para que la lectura sea clara.
// =============================================================================

import {
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts'
import { obtenerPronostico } from '../api/client'
import { useApi } from '../hooks/useApi'
import { Tarjeta, Cargando, MensajeError } from './Estado'
import { formatoSoles, formatoSolesCompacto, etiquetaMes, mesLargo } from '../utils/format'

const COLOR_HISTORICO = '#2563eb' // azul
const COLOR_PRONOSTICO = '#16a34a' // verde
const COLOR_BANDA = '#16a34a'

/**
 * Une la serie histórica y el pronóstico en un solo arreglo para Recharts.
 * El último punto histórico se "puentea" hacia el pronóstico para que la línea
 * punteada y la banda arranquen pegadas a la serie observada (sin saltos).
 */
function combinarSeries(history, forecast) {
  const historia = history.map((h) => ({
    etiqueta: etiquetaMes(h.fecha),
    fecha: h.fecha,
    historico: h.gasto,
  }))

  const pron = forecast.map((f) => ({
    etiqueta: etiquetaMes(f.fecha),
    fecha: f.fecha,
    pronostico: f.pred,
    // Recharts dibuja una banda cuando el dataKey es un par [min, max].
    banda: [f.lower, f.upper],
  }))

  // Puente: el último mes observado también es el origen del pronóstico.
  if (historia.length > 0) {
    const ultimo = historia[historia.length - 1]
    ultimo.pronostico = ultimo.historico
    ultimo.banda = [ultimo.historico, ultimo.historico]
  }

  return { datos: [...historia, ...pron], inicioPronostico: pron[0]?.etiqueta }
}

/** Tooltip personalizado: muestra el valor según la capa (histórico o pronóstico). */
function TooltipPersonalizado({ active, payload, label }) {
  if (!active || !payload || payload.length === 0) return null
  const fila = payload[0].payload

  return (
    <div className="tooltip">
      <p className="tooltip__titulo">{mesLargo(fila.fecha)}</p>
      {fila.historico != null && (
        <p>
          <span className="punto" style={{ background: COLOR_HISTORICO }} />
          Gasto observado: <strong>{formatoSoles(fila.historico)}</strong>
        </p>
      )}
      {fila.pronostico != null && (
        <p>
          <span className="punto" style={{ background: COLOR_PRONOSTICO }} />
          Pronóstico: <strong>{formatoSoles(fila.pronostico)}</strong>
        </p>
      )}
      {Array.isArray(fila.banda) && fila.banda[0] !== fila.banda[1] && (
        <p className="tooltip__intervalo">
          Intervalo 95%: {formatoSoles(fila.banda[0])} – {formatoSoles(fila.banda[1])}
        </p>
      )}
    </div>
  )
}

export default function GraficoPronostico() {
  const { datos, cargando, error, recargar } = useApi(obtenerPronostico, [])

  const subtitulo = datos
    ? `Modelo: ${datos.best_model ?? 'N/D'} · Intervalo de confianza ${Math.round(
        datos.confidence_level * 100,
      )}%`
    : 'Evolución mensual del gasto y proyección'

  return (
    <Tarjeta titulo="Pronóstico del gasto mensual" subtitulo={subtitulo} className="tarjeta--ancha">
      {cargando && <Cargando texto="Cargando pronóstico…" />}
      {error && <MensajeError mensaje={error} onReintentar={recargar} />}

      {datos && (() => {
        const { datos: series, inicioPronostico } = combinarSeries(datos.history, datos.forecast)
        // Intervalo de etiquetas en el eje X para no saturarlo (~15 visibles).
        const paso = Math.max(0, Math.ceil(series.length / 15) - 1)

        return (
          <ResponsiveContainer width="100%" height={380}>
            <ComposedChart data={series} margin={{ top: 10, right: 20, left: 10, bottom: 30 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#eef1f5" />
              <XAxis
                dataKey="etiqueta"
                interval={paso}
                angle={-45}
                textAnchor="end"
                height={50}
                tick={{ fontSize: 11, fill: '#64748b' }}
              />
              <YAxis
                tickFormatter={formatoSolesCompacto}
                tick={{ fontSize: 11, fill: '#64748b' }}
                width={80}
              />
              <Tooltip content={<TooltipPersonalizado />} />
              <Legend verticalAlign="top" height={36} />

              {/* Banda del intervalo de confianza (lower–upper). */}
              <Area
                type="monotone"
                dataKey="banda"
                name="Intervalo de confianza 95%"
                stroke="none"
                fill={COLOR_BANDA}
                fillOpacity={0.15}
                connectNulls
                activeDot={false}
              />
              {/* Línea histórica observada. */}
              <Line
                type="monotone"
                dataKey="historico"
                name="Gasto observado"
                stroke={COLOR_HISTORICO}
                strokeWidth={2}
                dot={false}
                connectNulls
              />
              {/* Línea de pronóstico (punteada para diferenciarla). */}
              <Line
                type="monotone"
                dataKey="pronostico"
                name="Pronóstico"
                stroke={COLOR_PRONOSTICO}
                strokeWidth={2.5}
                strokeDasharray="6 4"
                dot={{ r: 2.5 }}
                connectNulls
              />
              {/* Marca vertical donde termina el histórico y empieza el pronóstico. */}
              {inicioPronostico && (
                <ReferenceLine
                  x={inicioPronostico}
                  stroke="#94a3b8"
                  strokeDasharray="4 4"
                  label={{ value: 'Inicio del pronóstico', position: 'top', fontSize: 10, fill: '#94a3b8' }}
                />
              )}
            </ComposedChart>
          </ResponsiveContainer>
        )
      })()}
    </Tarjeta>
  )
}
