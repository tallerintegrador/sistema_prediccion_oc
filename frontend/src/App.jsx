// =============================================================================
// App.jsx — Componente raíz del tablero.
//
// Arma la estructura de la página: encabezado + las cuatro secciones del
// dashboard. Cada sección es un componente independiente que se encarga de su
// propia llamada al API y de sus estados de carga/error.
// =============================================================================

import TarjetasIndicadores from './components/TarjetasIndicadores'
import GraficoPronostico from './components/GraficoPronostico'
import GraficoCategorias from './components/GraficoCategorias'
import TablaAlertas from './components/TablaAlertas'
import { URL_BASE } from './api/client'

export default function App() {
  return (
    <div className="app">
      <header className="encabezado">
        <div className="encabezado__contenido">
          <h1 className="encabezado__titulo">Sistema de Predicción de Órdenes de Compra</h1>
          <p className="encabezado__subtitulo">
            Pronóstico del gasto público y detección de órdenes anómalas
          </p>
        </div>
        <span className="encabezado__api" title="Origen de los datos">
          API: {URL_BASE}
        </span>
      </header>

      <main className="contenido">
        {/* 1) Indicadores clave (/summary) */}
        <TarjetasIndicadores />

        {/* 2) Pronóstico del gasto (/forecast) */}
        <GraficoPronostico />

        {/* 3) Gasto por categoría (/categories) */}
        <GraficoCategorias />

        {/* 4) Alertas de anomalías (/alerts) */}
        <TablaAlertas />
      </main>

      <footer className="pie">
        SistemaPrediccionOC · Tablero de sustentación — datos servidos por el API (FastAPI).
      </footer>
    </div>
  )
}
