// =============================================================================
// Dashboard.jsx — Vista principal del tablero (KPIs + pronóstico + categorías +
// alertas). Es el contenido que antes vivía en App.jsx; ahora App provee el
// layout y el enrutado, y esta página agrupa las cuatro secciones del negocio.
// =============================================================================

import TarjetasIndicadores from '../components/TarjetasIndicadores'
import GraficoPronostico from '../components/GraficoPronostico'
import GraficoCategorias from '../components/GraficoCategorias'
import TablaAlertas from '../components/TablaAlertas'

export default function Dashboard() {
  return (
    <>
      {/* 1) Indicadores clave (/summary) */}
      <TarjetasIndicadores />

      {/* 2) Pronóstico del gasto (/forecast) */}
      <GraficoPronostico />

      {/* 3) Gasto por categoría (/categories) */}
      <GraficoCategorias />

      {/* 4) Alertas de anomalías (/alerts) */}
      <TablaAlertas />
    </>
  )
}
