// =============================================================================
// format.js — Utilidades de formato (moneda, números y fechas) en español.
//
// Se usa el locale 'es-PE' y la moneda 'PEN' (soles) porque los datos provienen
// de órdenes de compra del Estado peruano.
// =============================================================================

// Formateador de moneda completo: S/ 8,078,350,718
const fmtSoles = new Intl.NumberFormat('es-PE', {
  style: 'currency',
  currency: 'PEN',
  maximumFractionDigits: 0,
})

// Formateador de moneda compacto: S/ 8.1 mil M (ideal para tarjetas).
const fmtSolesCompacto = new Intl.NumberFormat('es-PE', {
  style: 'currency',
  currency: 'PEN',
  notation: 'compact',
  maximumFractionDigits: 1,
})

// Formateador de números enteros con separador de miles: 483,692
const fmtEntero = new Intl.NumberFormat('es-PE', { maximumFractionDigits: 0 })

// Meses en español (índice 0 = enero) para etiquetas deterministas.
const MESES = [
  'ene', 'feb', 'mar', 'abr', 'may', 'jun',
  'jul', 'ago', 'sep', 'oct', 'nov', 'dic',
]
const MESES_LARGOS = [
  'enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio',
  'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre',
]

/** Soles con todos los dígitos: 106469175 -> "S/ 106,469,175". */
export const formatoSoles = (n) => (n == null ? '—' : fmtSoles.format(n))

/** Soles en notación compacta: 8078350718 -> "S/ 8.1 mil M". */
export const formatoSolesCompacto = (n) => (n == null ? '—' : fmtSolesCompacto.format(n))

/** Número entero con separador de miles: 483692 -> "483,692". */
export const formatoEntero = (n) => (n == null ? '—' : fmtEntero.format(n))

/** Proporción [0,1] a porcentaje: 0.292 -> "29.2%". */
export const formatoPorcentaje = (x, decimales = 1) =>
  x == null ? '—' : `${(x * 100).toFixed(decimales)}%`

/** Valor que YA está en unidades de porcentaje: 16.46 -> "16.5%" (métricas WAPE/MAPE). */
export const formatoPct = (x, decimales = 1) =>
  x == null ? '—' : `${Number(x).toFixed(decimales)}%`

/** Número con decimales fijos: 0.7529 -> "0.75" (métricas tipo MASE/Jaccard). */
export const formatoNumero = (x, decimales = 2) =>
  x == null ? '—' : Number(x).toFixed(decimales)

/**
 * Etiqueta corta de mes a partir de una fecha "YYYY-MM-DD": "2026-05-31" -> "may 26".
 * Se parsea manualmente para no depender de zonas horarias.
 */
export function etiquetaMes(fechaIso) {
  if (!fechaIso) return ''
  const [anio, mes] = fechaIso.split('-')
  return `${MESES[Number(mes) - 1]} ${anio.slice(2)}`
}

/** Mes y año en texto largo: "2026-06-30" -> "junio 2026". */
export function mesLargo(fechaIso) {
  if (!fechaIso) return ''
  const [anio, mes] = fechaIso.split('-')
  return `${MESES_LARGOS[Number(mes) - 1]} ${anio}`
}
