// Punto de entrada de la aplicación React.
// Monta el componente raíz <App /> dentro del <div id="root"> de index.html.
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App.jsx'
import './styles/index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
)
