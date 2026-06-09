// Punto de entrada de la aplicación React.
// Monta el componente raíz <App /> dentro del <div id="root"> de index.html,
// envuelto en el enrutador del navegador (react-router-dom).
import React from 'react'
import ReactDOM from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import App from './App.jsx'
import './styles/index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>,
)
