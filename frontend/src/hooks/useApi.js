// =============================================================================
// useApi.js — Hook reutilizable para llamadas al API.
//
// Centraliza los tres estados típicos de cualquier petición: cargando, error y
// datos. Cualquier componente que necesite consumir un endpoint usa este hook y
// se olvida de manejar el ciclo de vida a mano.
// =============================================================================

import { useCallback, useEffect, useState } from 'react'

/**
 * Ejecuta una función asíncrona del cliente de API y expone su estado.
 *
 * @param {() => Promise<any>} funcionApi  Función que devuelve una promesa (p.ej. obtenerResumen).
 * @param {Array} dependencias  Dependencias que, al cambiar, vuelven a disparar la llamada.
 * @returns {{ datos: any, cargando: boolean, error: string|null, recargar: () => void }}
 */
export function useApi(funcionApi, dependencias = []) {
  const [datos, setDatos] = useState(null)
  const [cargando, setCargando] = useState(true)
  const [error, setError] = useState(null)

  // useCallback evita recrear la función en cada render salvo que cambien las deps.
  const ejecutar = useCallback(() => {
    let cancelado = false
    setCargando(true)
    setError(null)

    funcionApi()
      .then((resultado) => {
        if (!cancelado) setDatos(resultado)
      })
      .catch((err) => {
        if (!cancelado) setError(err.message || 'Ocurrió un error inesperado.')
      })
      .finally(() => {
        if (!cancelado) setCargando(false)
      })

    // Función de limpieza: evita actualizar el estado si el componente se desmonta.
    return () => {
      cancelado = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, dependencias)

  useEffect(() => {
    const limpiar = ejecutar()
    return limpiar
  }, [ejecutar])

  return { datos, cargando, error, recargar: ejecutar }
}
