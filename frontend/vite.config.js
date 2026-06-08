// Configuración de Vite para el frontend de React.
// El servidor de desarrollo corre por defecto en http://localhost:5173.
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173, // Puerto del servidor de desarrollo (coincide con el CORS del backend).
    open: true, // Abre el navegador automáticamente al levantar.
  },
})
