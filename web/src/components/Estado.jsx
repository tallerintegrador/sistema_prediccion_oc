// =============================================================================
// Estado.jsx — Componentes pequeños y reutilizables para los estados de la UI.
//
// Cargando      : indicador de "cargando..." mientras llega la respuesta del API.
// MensajeError  : aviso de error de conexión con botón para reintentar.
// Tarjeta       : contenedor visual con título, usado por todas las secciones.
// =============================================================================

/** Indicador de carga centrado dentro de su contenedor. */
export function Cargando({ texto = 'Cargando…' }) {
  return (
    <div className="estado estado--cargando" role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" />
      <span>{texto}</span>
    </div>
  )
}

/** Mensaje de error con opción de reintentar la llamada al API. */
export function MensajeError({ mensaje, onReintentar }) {
  return (
    <div className="estado estado--error" role="alert">
      <strong>No se pudieron cargar los datos</strong>
      <p>{mensaje}</p>
      {onReintentar && (
        <button type="button" className="boton" onClick={onReintentar}>
          Reintentar
        </button>
      )}
    </div>
  )
}

/**
 * Contenedor con estilo de "tarjeta/panel". Recibe un título y, opcionalmente,
 * un subtítulo y contenido al costado del encabezado (acciones, leyenda, etc.).
 */
export function Tarjeta({ titulo, subtitulo, acciones, children, className = '' }) {
  return (
    <section className={`tarjeta ${className}`}>
      {(titulo || acciones) && (
        <header className="tarjeta__header">
          <div>
            {titulo && <h2 className="tarjeta__titulo">{titulo}</h2>}
            {subtitulo && <p className="tarjeta__subtitulo">{subtitulo}</p>}
          </div>
          {acciones && <div className="tarjeta__acciones">{acciones}</div>}
        </header>
      )}
      <div className="tarjeta__cuerpo">{children}</div>
    </section>
  )
}
