import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  darkMode: 'class',
  theme: {
    extend: {
      colors: {
        bg: {
          primary: '#0a0e1a',
          secondary: '#0f1420',
          tertiary: '#161b2c'
        },
        accent: {
          cyan: '#00d4ff',
          green: '#00ff88',
          red: '#ff3b6b',
          gold: '#ffb800',
          purple: '#7c5cff'
        }
      },
      fontFamily: {
        mono: ['JetBrains Mono', 'IBM Plex Mono', 'monospace'],
        ui: ['Space Grotesk', 'DM Sans', 'sans-serif'],
        hero: ['Orbitron', 'Rajdhani', 'sans-serif']
      }
    }
  },
  plugins: []
} satisfies Config;
