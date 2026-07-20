import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  // Next K embed: CLAWBY_UI_BASE=/clawby-ui/ npm run build
  // Standalone: npm run build  (base /)
  base: process.env.CLAWBY_UI_BASE || '/',
  plugins: [react()],
  server: {
    proxy: { '/api': 'http://127.0.0.1:8899' },
  },
  build: {
    outDir: 'dist',
    chunkSizeWarningLimit: 1500,
  },
})
