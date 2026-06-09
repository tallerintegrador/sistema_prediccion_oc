// =============================================================================
// HeatmapConcordancia.jsx — Matriz de concordancia (Jaccard) entre métodos.
//
// Mide el solapamiento del top 1% de Isolation Forest, Red y Reglas. Un mapa de
// calor 3x3 con CSS grid (sin dependencia extra). Más oscuro = mayor solapamiento.
// =============================================================================

import { formatoNumero } from '../utils/format'

// Devuelve un color de fondo proporcional al valor [0,1] (azul).
function colorCelda(v) {
  const alpha = 0.12 + Math.min(v, 1) * 0.78
  return `rgba(37, 99, 235, ${alpha.toFixed(3)})`
}

export default function HeatmapConcordancia({ concordancia }) {
  if (!concordancia || !concordancia.jaccard) {
    return <p className="vacio">Sin datos de concordancia.</p>
  }
  const { labels, jaccard } = concordancia

  return (
    <div className="heatmap-wrap">
      <div className="heatmap" style={{ gridTemplateColumns: `120px repeat(${labels.length}, 1fr)` }}>
        <div className="heatmap__esquina" />
        {labels.map((l) => (
          <div key={`col-${l}`} className="heatmap__cabecera">{l}</div>
        ))}
        {jaccard.map((fila, i) => (
          <FilaHeatmap key={`fila-${i}`} etiqueta={labels[i]} valores={fila} diagonal={i} />
        ))}
      </div>
      {concordancia.interseccion_triple != null && (
        <p className="heatmap__nota">
          <strong>{concordancia.interseccion_triple.toLocaleString('es-PE')}</strong> órdenes
          aparecen en el top 1% de los <strong>tres</strong> métodos (mayor consenso).
        </p>
      )}
    </div>
  )
}

function FilaHeatmap({ etiqueta, valores, diagonal }) {
  return (
    <>
      <div className="heatmap__cabecera heatmap__cabecera--fila">{etiqueta}</div>
      {valores.map((v, j) => (
        <div
          key={j}
          className="heatmap__celda"
          style={{ background: j === diagonal ? '#e2e8f0' : colorCelda(v) }}
          title={`Jaccard = ${formatoNumero(v, 2)}`}
        >
          {formatoNumero(v, 2)}
        </div>
      ))}
    </>
  )
}
