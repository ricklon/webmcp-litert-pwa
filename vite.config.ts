import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { VitePWA } from 'vite-plugin-pwa';

export default defineConfig({
  // Relative URLs allow the same build to run at localhost and from a
  // GitHub Pages project path such as /webmcp-litert-pwa/.
  base: './',
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['mark.svg', 'icon-192.png', 'icon-512.png'],
      manifest: {
        name: 'Local Tools Lab',
        short_name: 'Tools Lab',
        description: 'A local-first WebMCP and LiteRT-LM interoperability lab.',
        theme_color: '#161814',
        background_color: '#f3f0e8',
        display: 'standalone',
        start_url: './',
        scope: './',
        icons: [
          { src: 'icon-192.png', sizes: '192x192', type: 'image/png' },
          { src: 'icon-512.png', sizes: '512x512', type: 'image/png' },
          { src: 'icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' }
        ]
      },
      workbox: {
        globPatterns: ['**/*.{js,css,html,svg}'],
        navigateFallback: 'index.html'
      }
    })
  ],
  server: {
    headers: { 'Origin-Agent-Cluster': '?1' }
  },
  preview: {
    headers: { 'Origin-Agent-Cluster': '?1' }
  }
});
