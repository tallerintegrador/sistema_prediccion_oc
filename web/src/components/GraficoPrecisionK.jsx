// =============================================================================
// GraficoPrecisionK.jsx — precision@k del Módulo B.
//
// Muestra qué fracción de las k alertas de mayor riesgo está respaldada por una
// señal interpretable. Eje X = k (categórico); eje Y = precisión en %.
// =============================================================================

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

export default function GraficoPrecisionK({ datos = [] }) {
  const puntos = datos.map((d) => ({ k: `k=${d.k}`, precision: Math.round(d.precision * 100) }))

  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={puntos} margin={{ top: 10, right: 20, left: 0, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#eef1f5" />
        <XAxis dataKey="k" tick={{ fontSize: 12, fill: '#334155' }} />
        <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} tick={{ fontSize: 11, fill: '#64748b' }} />
        <Tooltip formatter={(v) => [`${v}%`, 'precisión']} />
        <Line type="monotone" dataKey="precision" stroke="#2563eb" strokeWidth={2.5} dot={{ r: 5 }} />
      </LineChart>
    </ResponsiveContainer>
  )
}
