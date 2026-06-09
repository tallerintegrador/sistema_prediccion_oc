// =============================================================================
// App.jsx — Layout raíz + enrutado.
//
// Provee el encabezado con la navegación (Dashboard / Métricas ML), el contenedor
// principal donde se renderiza la página activa, y el pie. Cada página vive en
// `src/pages/` y arma sus propias secciones y llamadas al API.
// =============================================================================

import { NavLink, Route, Routes } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import MetricasML from './pages/MetricasML'
import { URL_BASE } from './api/client'

const claseTab = ({ isActive }) => `nav__tab ${isActive ? 'nav__tab--activa' : ''}`

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
        <nav className="nav" aria-label="Secciones">
          <NavLink to="/" end className={claseTab}>Dashboard</NavLink>
          <NavLink to="/metricas" className={claseTab}>Métricas ML</NavLink>
        </nav>
        <span className="encabezado__api" title="Origen de los datos">
          API: {URL_BASE}
        </span>
      </header>

      <main className="contenido">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/metricas" element={<MetricasML />} />
        </Routes>
      </main>

      <footer className="pie">
        SistemaPrediccionOC · Tablero de sustentación — datos servidos por el API (FastAPI).
      </footer>
    </div>
  )
}
