// =============================================================================
// GaleriaFiguras.jsx — Galería de figuras del ML servidas por el API (/figures).
//
// Muestra los PNGs generados por los Módulos A y B como respaldo visual de las
// métricas. Recibe una lista de { archivo, titulo }.
// =============================================================================

import { useState } from 'react'
import { urlFigura } from '../api/client'

export default function GaleriaFiguras({ figuras = [] }) {
  const [zoom, setZoom] = useState(null)

  return (
    <>
      <div className="galeria">
        {figuras.map((f) => (
          <figure key={f.archivo} className="galeria__item" onClick={() => setZoom(f)}>
            <img
              src={urlFigura(f.archivo)}
              alt={f.titulo}
              loading="lazy"
              onError={(e) => {
                e.currentTarget.parentElement.style.display = 'none'
              }}
            />
            <figcaption>{f.titulo}</figcaption>
          </figure>
        ))}
      </div>

      {zoom && (
        <div className="lightbox" role="dialog" aria-label={zoom.titulo} onClick={() => setZoom(null)}>
          <img src={urlFigura(zoom.archivo)} alt={zoom.titulo} />
        </div>
      )}
    </>
  )
}
