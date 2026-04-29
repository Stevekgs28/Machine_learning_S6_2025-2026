/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx,ts,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        mono: ['"JetBrains Mono"', '"Fira Code"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
        sans: ['"Inter"', 'system-ui', 'sans-serif'],
        display: ['"Space Grotesk"', '"Inter"', 'system-ui', 'sans-serif'],
      },
      colors: {
        // Terminal-ML palette
        ink: {
          950: '#070a0d',
          900: '#0b1117',
          800: '#0f1620',
          700: '#161e2a',
          600: '#1f2937',
          500: '#2b3647',
        },
        mint: {
          DEFAULT: '#5eead4',
          400: '#7df0d9',
          500: '#5eead4',
          600: '#2dd4bf',
        },
        amber: {
          DEFAULT: '#fbbf24',
          400: '#fcd34d',
          500: '#fbbf24',
          600: '#f59e0b',
        },
        crimson: {
          DEFAULT: '#f87171',
          500: '#f87171',
          600: '#ef4444',
        },
        muted: '#7d8a9c',
      },
      boxShadow: {
        'glow-mint': '0 0 0 1px rgba(94,234,212,0.25), 0 8px 30px -12px rgba(94,234,212,0.35)',
        'glow-amber': '0 0 0 1px rgba(251,191,36,0.25), 0 8px 30px -12px rgba(251,191,36,0.35)',
      },
      backgroundImage: {
        'grid-faint': 'linear-gradient(rgba(94,234,212,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(94,234,212,0.05) 1px, transparent 1px)',
      },
    },
  },
  plugins: [],
}
