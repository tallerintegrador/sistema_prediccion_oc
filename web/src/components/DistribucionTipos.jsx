// =============================================================================
// DistribucionTipos.jsx — Distribución de alertas por tipo de anomalía.
//
// Recibe un objeto { tipo: conteo } y dibuja barras horizontales ordenadas.
// =============================================================================

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { formatoEntero } from '../utils/format'

const COLORES = ['#1d4ed8', '#2563eb', '#3b82f6', '#60a5fa', '#93c5fd', '#bfdbfe']

function recortar(texto, max = 26) {
  return texto && texto.length > max ? `${texto.slice(0, max - 1)}…` : texto
}

export default function DistribucionTipos({ distribucion = {} }) {
  const datos = Object.entries(distribucion)
    .map(([tipo, conteo]) => ({ tipo, conteo }))
    .sort((a, b) => b.conteo - a.conteo)

  return (
    <ResponsiveContainer width="100%" height={Math.max(260, datos.length * 44)}>
      <BarChart data={datos} layout="vertical" margin={{ top: 5, right: 40, left: 10, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#eef1f5" horizontal={false} />
        <XAxis type="number" tickFormatter={formatoEntero} tick={{ fontSize: 11, fill: '#64748b' }} />
        <YAxis type="category" dataKey="tipo" width={180} tickFormatter={recortar} tick={{ fontSize: 11, fill: '#334155' }} />
        <Tooltip formatter={(v) => [formatoEntero(v), 'alertas']} cursor={{ fill: 'rgba(37,99,235,0.06)' }} />
        <Bar dataKey="conteo" radius={[0, 4, 4, 0]}>
          {datos.map((_, i) => (
            <Cell key={i} fill={COLORES[i % COLORES.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
