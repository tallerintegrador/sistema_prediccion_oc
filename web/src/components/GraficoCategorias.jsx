// =============================================================================
// GraficoCategorias.jsx — Gasto por categoría (ACUERDO_MARCO).
//
// Consume /categories y dibuja un gráfico de barras horizontales ordenado de
// mayor a menor gasto. Se usan barras horizontales porque los nombres de las
// categorías son largos y así se leen cómodamente.
// =============================================================================

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
  ResponsiveContainer,
} from 'recharts'
import { obtenerCategorias } from '../api/client'
import { useApi } from '../hooks/useApi'
import { Tarjeta, Cargando, MensajeError } from './Estado'
import { formatoSoles, formatoSolesCompacto, formatoPorcentaje } from '../utils/format'

// Paleta secuencial: las categorías de mayor gasto se ven más intensas.
const COLORES = [
  '#1d4ed8', '#2563eb', '#3b82f6', '#60a5fa', '#93c5fd',
  '#7dd3fc', '#a5b4fc', '#c7d2fe', '#e0e7ff',
]

/** Recorta un texto largo para usarlo como etiqueta del eje. */
function recortar(texto, max = 32) {
  return texto.length > max ? `${texto.slice(0, max - 1)}…` : texto
}

/** Tooltip con el nombre completo, el monto y la participación. */
function TooltipCategoria({ active, payload }) {
  if (!active || !payload || payload.length === 0) return null
  const c = payload[0].payload
  return (
    <div className="tooltip">
      <p className="tooltip__titulo">{c.category}</p>
      <p>Gasto: <strong>{formatoSoles(c.total_spend)}</strong></p>
      <p>Participación: <strong>{formatoPorcentaje(c.share)}</strong></p>
    </div>
  )
}

export default function GraficoCategorias() {
  const { datos, cargando, error, recargar } = useApi(obtenerCategorias, [])

  const subtitulo = datos
    ? `${datos.n_categories} categorías · Gasto total ${formatoSolesCompacto(datos.total_spend)}`
    : 'Distribución del gasto por acuerdo marco'

  return (
    <Tarjeta titulo="Gasto por categoría (Acuerdo Marco)" subtitulo={subtitulo}>
      {cargando && <Cargando texto="Cargando categorías…" />}
      {error && <MensajeError mensaje={error} onReintentar={recargar} />}

      {datos && (
        <ResponsiveContainer width="100%" height={Math.max(320, datos.categories.length * 42)}>
          <BarChart
            data={datos.categories}
            layout="vertical"
            margin={{ top: 5, right: 30, left: 10, bottom: 5 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#eef1f5" horizontal={false} />
            <XAxis
              type="number"
              tickFormatter={formatoSolesCompacto}
              tick={{ fontSize: 11, fill: '#64748b' }}
            />
            <YAxis
              type="category"
              dataKey="category"
              tickFormatter={recortar}
              width={230}
              tick={{ fontSize: 11, fill: '#334155' }}
            />
            <Tooltip content={<TooltipCategoria />} cursor={{ fill: 'rgba(37,99,235,0.06)' }} />
            <Bar dataKey="total_spend" radius={[0, 4, 4, 0]}>
              {datos.categories.map((_, i) => (
                <Cell key={i} fill={COLORES[i % COLORES.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </Tarjeta>
  )
}
