// =============================================================================
// GraficoRecallInyeccion.jsx — Recall por inyección de anomalías sintéticas.
//
// Por cada tipo inyectado (monto, IGV, fraccionamiento) muestra dos barras:
// recall como alerta (medio/alto) y recall dentro del top 1% de riesgo.
// =============================================================================

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

function recortar(texto, max = 16) {
  return texto && texto.length > max ? `${texto.slice(0, max - 1)}…` : texto
}

export default function GraficoRecallInyeccion({ datos = [] }) {
  const filas = datos.map((d) => ({
    tipo: d.tipo,
    alerta: Math.round((d.recall_alerta ?? 0) * 100),
    top1: Math.round((d.recall_top1pct ?? 0) * 100),
  }))

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={filas} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#eef1f5" />
        <XAxis dataKey="tipo" tickFormatter={recortar} tick={{ fontSize: 11, fill: '#334155' }} />
        <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} tick={{ fontSize: 11, fill: '#64748b' }} />
        <Tooltip formatter={(v, n) => [`${v}%`, n === 'alerta' ? 'Recall (alerta)' : 'Recall (top 1%)']} />
        <Legend formatter={(n) => (n === 'alerta' ? 'Recall (alerta)' : 'Recall (top 1%)')} />
        <Bar dataKey="alerta" fill="#2563eb" radius={[4, 4, 0, 0]} />
        <Bar dataKey="top1" fill="#f59e0b" radius={[4, 4, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}
