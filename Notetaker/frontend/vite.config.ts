import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',  // Escuchar en todas las interfaces (accesible desde red/VPN)
    port: 5173,       // Puerto por defecto de Vite
    strictPort: false,
    // Permitir el dominio de Cloudflare
    allowedHosts: [
      'notetaker.cosgs.com',
      'localhost',
      '.cosgs.com',  // Permite cualquier subdominio de cosgs.com
    ],
    // Forzar que escuche en todas las interfaces incluyendo VPN
    hmr: {
      host: '0.0.0.0',
    },
  },
  preview: {
    host: '0.0.0.0',  // También para preview (producción)
    port: 4173,       // Puerto por defecto de preview
  },
})