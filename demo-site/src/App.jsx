import { useEffect, useState } from 'react'
import Hero from './sections/Hero.jsx'
import Playground from './sections/Playground.jsx'
import ModelComparison from './sections/ModelComparison.jsx'
import Diagnostic from './sections/Diagnostic.jsx'
import PivotB from './sections/PivotB.jsx'
import About from './sections/About.jsx'
import { checkHealth, getApiBase } from './lib/api.js'

const NAV = [
  { id: 'hero', label: '00 / Pitch' },
  { id: 'playground', label: '01 / Predict' },
  { id: 'models', label: '02 / Models' },
  { id: 'diagnostic', label: '03 / Diagnostic' },
  { id: 'pivot-b', label: '04 / Pivot B' },
  { id: 'about', label: '05 / About' },
]

export default function App() {
  const [active, setActive] = useState('hero')
  const [api, setApi] = useState({ status: 'checking', model_loaded: false, model_name: null })

  // Surveille la section visible
  useEffect(() => {
    const sections = NAV.map((n) => document.getElementById(n.id)).filter(Boolean)
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) setActive(e.target.id)
        })
      },
      { rootMargin: '-40% 0px -55% 0px' }
    )
    sections.forEach((s) => observer.observe(s))
    return () => observer.disconnect()
  }, [])

  // Health-check API
  useEffect(() => {
    let mounted = true
    const run = async () => {
      const h = await checkHealth()
      if (mounted) setApi(h)
    }
    run()
    const i = setInterval(run, 15000)
    return () => {
      mounted = false
      clearInterval(i)
    }
  }, [])

  return (
    <div className="min-h-screen">
      <TopBar active={active} api={api} />
      <main className="mx-auto max-w-6xl px-5 md:px-8 pt-24 pb-32">
        <Hero />
        <Playground apiStatus={api} />
        <ModelComparison />
        <Diagnostic />
        <PivotB />
        <About />
      </main>
      <Footer />
    </div>
  )
}

function TopBar({ active, api }) {
  const ok = api.status === 'ok' && api.model_loaded
  return (
    <header className="fixed top-0 left-0 right-0 z-40 border-b border-ink-700/70 bg-ink-950/80 backdrop-blur-md">
      <div className="mx-auto max-w-6xl px-5 md:px-8 h-14 flex items-center justify-between">
        <a href="#hero" className="flex items-center gap-2.5 group">
          <span className="font-mono text-mint text-lg">›</span>
          <span className="font-mono text-sm text-slate-100 group-hover:text-mint transition-colors">
            steve-ml-cybersec
          </span>
          <span className="hidden md:inline pill">MOD10 · W26</span>
        </a>
        <nav className="hidden md:flex items-center gap-1">
          {NAV.map((n) => (
            <a
              key={n.id}
              href={`#${n.id}`}
              className={`px-2.5 py-1 rounded-md font-mono text-[11px] tracking-wider transition-colors ${
                active === n.id
                  ? 'text-mint bg-mint/10'
                  : 'text-muted hover:text-slate-200'
              }`}
            >
              {n.label}
            </a>
          ))}
        </nav>
        <ApiBadge api={api} ok={ok} />
      </div>
    </header>
  )
}

function ApiBadge({ api, ok }) {
  let color, label
  if (api.status === 'checking') {
    color = 'bg-amber'
    label = 'API · checking'
  } else if (ok) {
    color = 'bg-mint'
    label = `API · ${api.model_name || 'live'}`
  } else if (api.status === 'ok' && !api.model_loaded) {
    color = 'bg-amber'
    label = 'API · no model'
  } else {
    color = 'bg-crimson'
    label = 'API · offline'
  }
  return (
    <div
      className="flex items-center gap-2 rounded-full border border-ink-600 bg-ink-900/80 pl-2 pr-3 py-1 font-mono text-[11px]"
      title={`Base: ${getApiBase()}`}
    >
      <span className={`relative flex h-2 w-2`}>
        <span className={`absolute inline-flex h-full w-full rounded-full ${color} opacity-70 ${ok ? 'animate-ping' : ''}`} />
        <span className={`relative inline-flex rounded-full h-2 w-2 ${color}`} />
      </span>
      <span className="text-slate-300">{label}</span>
    </div>
  )
}

function Footer() {
  return (
    <footer className="border-t border-ink-700/70 py-8">
      <div className="mx-auto max-w-6xl px-5 md:px-8 flex flex-col md:flex-row items-start md:items-center justify-between gap-3 text-xs font-mono text-muted">
        <p>
          MOD10 — Machine Learning Winter 2026 · Instructor : Mohammed A. Shehab · 30 % du cours
        </p>
        <p>
          built with <span className="text-mint">vite + react + tailwind</span> · backend <span className="text-mint">fastapi + sklearn</span>
        </p>
      </div>
    </footer>
  )
}
