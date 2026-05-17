/// <reference types="vitest" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'node:path';

// Browser-only config: no electron plugin, so the app runs in Chrome/Firefox
// for UI dev and bug-fixing without needing the Electron runtime.
export default defineConfig({
  resolve: {
    alias: { '@': path.resolve(__dirname, 'src') }
  },
  plugins: [react()],
  build: { outDir: 'dist-browser' },
  server: {
    port: 5174,
    open: true,
  },
});
