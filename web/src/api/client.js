// =============================================================================
// client.js — Capa dedicada a las llamadas al API (FastAPI).
//
// Toda comunicación con el backend pasa por aquí. Así los componentes no saben
// nada de URLs ni de fetch: solo piden datos. La URL base se lee de la variable
// de entorno VITE_API_URL (definida en el archivo .env), de modo que cambiar de
// servidor es tan simple como editar esa variable.
// =============================================================================

// URL base del API. Si no estuviera definida, se usa el valor por defecto local.
const URL_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

/**
 * Realiza una petición GET al API y devuelve el JSON ya parseado.
 *
 * @param {string} ruta   Ruta del endpoint (p.ej. "/summary").
 * @param {object} params Parámetros de query opcionales (p.ej. { page: 1 }).
 * @returns {Promise<object>} Cuerpo de la respuesta como objeto JS.
 * @throws {Error} Con un mensaje en español si hay fallo de red o HTTP.
 */
async function obtener(ruta, params = {}) {
  // Se arma la URL con sus parámetros de query, omitiendo los vacíos/nulos.
  const url = new URL(ruta, URL_BASE)
  Object.entries(params).forEach(([clave, valor]) => {
    if (valor !== undefined && valor !== null && valor !== '') {
      url.searchParams.set(clave, valor)
    }
  })

  let respuesta
  try {
    respuesta = await fetch(url, { headers: { Accept: 'application/json' } })
  } catch {
    // Falla de red: el backend está apagado, la URL es incorrecta o hay CORS.
    throw new Error(
      `No se pudo conectar con el API en ${URL_BASE}. ` +
        'Verifica que el backend esté corriendo (uvicorn) y que la URL sea correcta.',
    )
  }

  if (!respuesta.ok) {
    // El servidor respondió con un código de error (4xx / 5xx).
    throw new Error(`El API respondió con un error ${respuesta.status} (${respuesta.statusText}).`)
  }

  return respuesta.json()
}

// --- Endpoints expuestos como funciones con nombre claro ---------------------

/** /summary — indicadores clave del tablero. */
export const obtenerResumen = () => obtener('/summary')

/** /forecast — histórico mensual + pronóstico con intervalo de confianza. */
export const obtenerPronostico = () => obtener('/forecast')

/** /categories — gasto acumulado por categoría (ACUERDO_MARCO). */
export const obtenerCategorias = () => obtener('/categories')

/**
 * /alerts — alertas rankeadas por riesgo, con filtros y paginación.
 * @param {{ type?: string, level?: string, page?: number, page_size?: number }} filtros
 */
export const obtenerAlertas = (filtros = {}) => obtener('/alerts', filtros)

/** /metrics/forecast — métricas del Módulo A (comparación de modelos, mejor modelo). */
export const obtenerMetricasPronostico = () => obtener('/metrics/forecast')

/** /metrics/anomalies — métricas del Módulo B (precision@k, recall, concordancia, etc.). */
export const obtenerMetricasAnomalias = () => obtener('/metrics/anomalies')

/** /models — registro compacto de modelos comparados. */
export const obtenerModelos = () => obtener('/models')

/** Construye la URL de una figura del ML servida por el API (/figures/<archivo>). */
export const urlFigura = (archivo) => `${URL_BASE}/figures/${archivo}`

export { URL_BASE }
