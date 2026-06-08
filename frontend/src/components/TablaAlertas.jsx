// =============================================================================
// TablaAlertas.jsx — Tabla de alertas rankeadas por riesgo (Módulo B).
//
// Consume /alerts con filtros por tipo de anomalía y por nivel (alto/medio) y
// paginación. Cada fila muestra el puntaje de riesgo de forma visible (valor +
// barra) y el nivel como una etiqueta de color.
// =============================================================================

import { useEffect, useMemo, useState } from 'react'
import { obtenerAlertas } from '../api/client'
import { useApi } from '../hooks/useApi'
import { Tarjeta, Cargando, MensajeError } from './Estado'
import { formatoSoles, formatoEntero } from '../utils/format'

const TAMANO_PAGINA = 20

// Opciones de nivel: la UI muestra español, pero el API espera high/medium.
const OPCIONES_NIVEL = [
  { etiqueta: 'Todos los niveles', valor: '' },
  { etiqueta: 'Alto', valor: 'high' },
  { etiqueta: 'Medio', valor: 'medium' },
]

/** Etiqueta de color según el nivel de riesgo. */
function BadgeNivel({ nivel }) {
  const clave = (nivel || '').toLowerCase()
  const clase = clave === 'alto' ? 'badge badge--alto' : 'badge badge--medio'
  return <span className={clase}>{nivel || '—'}</span>
}

/** Puntaje de riesgo (0–1) como número + barra proporcional. */
function BarraRiesgo({ puntaje }) {
  if (puntaje == null) return <span className="texto-tenue">—</span>
  const pct = Math.round(puntaje * 100)
  const color = puntaje >= 0.66 ? '#dc2626' : puntaje >= 0.33 ? '#f59e0b' : '#16a34a'
  return (
    <div className="riesgo">
      <span className="riesgo__valor">{puntaje.toFixed(2)}</span>
      <span className="riesgo__pista">
        <span className="riesgo__relleno" style={{ width: `${pct}%`, background: color }} />
      </span>
    </div>
  )
}

export default function TablaAlertas() {
  const [tipo, setTipo] = useState('')
  const [nivel, setNivel] = useState('')
  const [pagina, setPagina] = useState(1)

  // Catálogo de tipos: se conserva entre cargas para que el filtro no parpadee.
  const [tiposDisponibles, setTiposDisponibles] = useState([])

  // La llamada se rehace cuando cambian filtros o página.
  const peticion = useMemo(
    () => () => obtenerAlertas({ type: tipo, level: nivel, page: pagina, page_size: TAMANO_PAGINA }),
    [tipo, nivel, pagina],
  )
  const { datos, cargando, error, recargar } = useApi(peticion, [tipo, nivel, pagina])

  // Al cambiar un filtro, se vuelve a la primera página.
  const cambiarTipo = (e) => { setTipo(e.target.value); setPagina(1) }
  const cambiarNivel = (e) => { setNivel(e.target.value); setPagina(1) }

  // Guarda el catálogo de tipos la primera vez que llega del API.
  useEffect(() => {
    if (datos?.available_types?.length) setTiposDisponibles(datos.available_types)
  }, [datos])

  const totalPaginas = datos?.total_pages ?? 0

  const filtros = (
    <div className="filtros">
      <label className="filtro">
        <span>Tipo de anomalía</span>
        <select value={tipo} onChange={cambiarTipo}>
          <option value="">Todos los tipos</option>
          {tiposDisponibles.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>
      </label>
      <label className="filtro">
        <span>Nivel de riesgo</span>
        <select value={nivel} onChange={cambiarNivel}>
          {OPCIONES_NIVEL.map((o) => (
            <option key={o.valor} value={o.valor}>{o.etiqueta}</option>
          ))}
        </select>
      </label>
    </div>
  )

  const subtitulo = datos
    ? `${formatoEntero(datos.total)} alertas que cumplen el filtro`
    : 'Órdenes de compra con riesgo de anomalía'

  return (
    <Tarjeta
      titulo="Alertas de órdenes anómalas"
      subtitulo={subtitulo}
      acciones={filtros}
      className="tarjeta--ancha"
    >
      {error && <MensajeError mensaje={error} onReintentar={recargar} />}

      {!error && (
        <>
          <div className="tabla-scroll">
            <table className="tabla">
              <thead>
                <tr>
                  <th>#</th>
                  <th>Orden</th>
                  <th>Entidad</th>
                  <th>Proveedor</th>
                  <th>Tipo de anomalía</th>
                  <th className="num">Monto</th>
                  <th>Riesgo</th>
                  <th>Nivel</th>
                  <th>Motivo</th>
                </tr>
              </thead>
              <tbody>
                {cargando && (
                  <tr>
                    <td colSpan={9}><Cargando texto="Cargando alertas…" /></td>
                  </tr>
                )}

                {!cargando && datos?.items?.length === 0 && (
                  <tr>
                    <td colSpan={9} className="tabla__vacio">
                      No hay alertas que coincidan con los filtros seleccionados.
                    </td>
                  </tr>
                )}

                {!cargando &&
                  datos?.items?.map((a) => (
                    <tr key={`${a.RANK}-${a.ORDEN_ELECTRONICA}`}>
                      <td className="num">{a.RANK}</td>
                      <td className="celda-orden">
                        <span className="mono">{a.ORDEN_ELECTRONICA || '—'}</span>
                        {a.FECHA_FORMALIZACION && (
                          <span className="texto-tenue">{a.FECHA_FORMALIZACION.slice(0, 10)}</span>
                        )}
                      </td>
                      <td title={a.ENTIDAD || ''}>{a.ENTIDAD || '—'}</td>
                      <td title={a.PROVEEDOR || ''}>{a.PROVEEDOR || '—'}</td>
                      <td>{a.TIPO_ANOMALIA || '—'}</td>
                      <td className="num">{formatoSoles(a.TOTAL)}</td>
                      <td><BarraRiesgo puntaje={a.PUNTAJE_RIESGO} /></td>
                      <td><BadgeNivel nivel={a.NIVEL} /></td>
                      <td className="celda-motivo" title={a.MOTIVO || ''}>{a.MOTIVO || '—'}</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>

          {/* Controles de paginación */}
          <div className="paginacion">
            <button
              type="button"
              className="boton"
              onClick={() => setPagina((p) => Math.max(1, p - 1))}
              disabled={cargando || pagina <= 1}
            >
              ← Anterior
            </button>
            <span className="paginacion__info">
              Página <strong>{pagina}</strong> de <strong>{totalPaginas || 1}</strong>
            </span>
            <button
              type="button"
              className="boton"
              onClick={() => setPagina((p) => p + 1)}
              disabled={cargando || pagina >= totalPaginas}
            >
              Siguiente →
            </button>
          </div>
        </>
      )}
    </Tarjeta>
  )
}
