// =============================================================================
// GraficoBarrasMetrica.jsx — Barras horizontales de una métrica por modelo.
//
// Reutilizable: recibe los modelos, la clave de la métrica a graficar y un color.
// Ordena de menor a mayor (menor = mejor) y resalta el mejor modelo.
// =============================================================================

import {
  Bar,
  BarChart,
  Cell,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

function recortar(texto, max = 18) {
  return texto && texto.length > max ? `${texto.slice(0, max - 1)}…` : texto
}

export default function GraficoBarrasMetrica({
  modelos = [],
  metrica = 'WAPE',
  color = '#2563eb',
  mejor,
  sufijo = '',
  decimales = 1,
}) {
  const datos = [...modelos]
    .filter((m) => m[metrica] != null)
    .sort((a, b) => a[metrica] - b[metrica])
    .map((m) => ({ modelo: m.Modelo, valor: m[metrica] }))

  return (
    <ResponsiveContainer width="100%" height={Math.max(280, datos.length * 30)}>
      <BarChart data={datos} layout="vertical" margin={{ top: 5, right: 40, left: 10, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#eef1f5" horizontal={false} />
        <XAxis type="number" tick={{ fontSize: 11, fill: '#64748b' }} tickFormatter={(v) => `${v.toFixed(0)}${sufijo}`} />
        <YAxis
          type="category"
          dataKey="modelo"
          width={140}
          tickFormatter={recortar}
          tick={{ fontSize: 11, fill: '#334155' }}
        />
        <Tooltip
          formatter={(v) => [`${Number(v).toFixed(decimales)}${sufijo}`, metrica]}
          cursor={{ fill: 'rgba(37,99,235,0.06)' }}
        />
        <Bar dataKey="valor" radius={[0, 4, 4, 0]}>
          {datos.map((d) => (
            <Cell key={d.modelo} fill={d.modelo === mejor ? '#16a34a' : color} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
