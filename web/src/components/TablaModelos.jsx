// =============================================================================
// TablaModelos.jsx — Comparación de modelos del Módulo A.
//
// Tabla con las 7 métricas de backtesting por modelo (menor es mejor en todas).
// Resalta el mejor modelo. Recibe la lista `modelos` y el nombre del `mejor`.
// =============================================================================

import { formatoPct, formatoNumero, formatoSolesCompacto } from '../utils/format'

// Columnas: clave en el JSON, etiqueta, y cómo formatear el valor.
const COLUMNAS = [
  { clave: 'WAPE', etiqueta: 'WAPE', fmt: formatoPct },
  { clave: 'MASE', etiqueta: 'MASE', fmt: (v) => formatoNumero(v, 3) },
  { clave: 'MAE', etiqueta: 'MAE', fmt: formatoSolesCompacto },
  { clave: 'RMSE', etiqueta: 'RMSE', fmt: formatoSolesCompacto },
  { clave: 'MAPE', etiqueta: 'MAPE', fmt: formatoPct },
  { clave: 'sMAPE', etiqueta: 'sMAPE', fmt: formatoPct },
  { clave: 'MPE', etiqueta: 'MPE', fmt: formatoPct },
]

export default function TablaModelos({ modelos = [], mejor }) {
  // Orden ascendente por WAPE (la métrica primaria): el mejor arriba.
  const filas = [...modelos].sort((a, b) => (a.WAPE ?? Infinity) - (b.WAPE ?? Infinity))

  return (
    <div className="tabla-scroll">
      <table className="tabla">
        <thead>
          <tr>
            <th className="tabla__izq">Modelo</th>
            {COLUMNAS.map((c) => (
              <th key={c.clave}>{c.etiqueta}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {filas.map((m) => {
            const esMejor = m.Modelo === mejor
            return (
              <tr key={m.Modelo} className={esMejor ? 'fila--mejor' : ''}>
                <td className="tabla__izq">
                  {m.Modelo}
                  {esMejor && <span className="chip chip--mejor">mejor</span>}
                </td>
                {COLUMNAS.map((c) => (
                  <td key={c.clave}>{c.fmt(m[c.clave])}</td>
                ))}
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
