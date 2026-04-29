import { useEffect, useState } from 'react'

const KILLER_STATS = [
  {
    label: 'KS-test p-value',
    value: '0.804',
    sub: 'Financial Loss vs Uniform[0.5, 99.99]',
    accent: 'crimson',
    explain: 'Hypothèse "uniforme aléatoire" non rejetable.',
  },
  {
    label: 'AUC dataset officiel',
    value: '0.50',
    sub: '6 modèles convergent au hasard',
    accent: 'crimson',
    explain: 'Indistinguable d\'un tirage à pile ou face.',
  },
  {
    label: 'F1 Pivot B',
    value: '0.97',
    sub: 'Même pipeline · PhishingWebsites',
    accent: 'mint',
    explain: '+0.48 sans changer une ligne de code.',
  },
]

const TYPED_LINES = [
  '$ python main.py --dataset Global_Cybersecurity_Threats_2015-2024.csv',
  '> 6 models trained, 5-fold CV, GridSearchCV',
  '> test F1 ≈ 0.49 across all models, AUC ≈ 0.50',
  '> running diagnostic: KS, MI, Cramér\'s V, χ², SHAP, Pivot B…',
  '> verdict: dataset carries no exploitable signal',
  '> pipeline validated on a real dataset → F1 0.97',
]

export default function Hero() {
  return (
    <section id="hero" className="pt-8 pb-24 scroll-mt-24">
      <div className="grid md:grid-cols-12 gap-8 items-start">
        <div className="md:col-span-7 fade-up">
          <div className="flex items-center gap-2 mb-5">
            <span className="pill">Cybersecurity ML · Winter 2026</span>
            <span className="pill-amber">Pivot C narrative</span>
          </div>
          <h1 className="font-display text-4xl md:text-6xl font-semibold text-white leading-[1.05] tracking-tight">
            We did not invent <br className="hidden md:block" />
            <span className="text-mint">a problem.</span>
          </h1>
          <p className="mt-6 text-slate-300 text-lg leading-relaxed max-w-2xl">
            We applied a rigorous ML pipeline — preprocessing, six models including ensembles,
            interpretability, deployment, MLflow tracking — and observed flat results. Rather than
            tune our way around them, we instrumented the data with{' '}
            <span className="text-mint">four convergent statistical tests</span> and a{' '}
            <span className="text-mint">Pivot B validation</span> on a real dataset. Identifying when
            ML is the wrong tool, while still delivering every component the rubric requires, is a
            methodological success.
          </p>
          <div className="mt-7 flex flex-wrap gap-3">
            <a href="#playground" className="btn-primary">
              <span>Run a prediction</span>
              <span aria-hidden>→</span>
            </a>
            <a href="#diagnostic" className="btn-ghost">See the diagnostic</a>
            <a href="#pivot-b" className="btn-ghost">Pivot B proof</a>
          </div>
        </div>

        <div className="md:col-span-5">
          <Terminal lines={TYPED_LINES} />
        </div>
      </div>

      {/* Killer stats */}
      <div className="mt-14 grid md:grid-cols-3 gap-4">
        {KILLER_STATS.map((s, i) => (
          <div
            key={s.label}
            className="card card-hover p-6 fade-up"
            style={{ animationDelay: `${0.1 + i * 0.1}s` }}
          >
            <div className="flex items-center justify-between">
              <span className="font-mono text-[11px] uppercase tracking-wider text-muted">
                {s.label}
              </span>
              <span className={s.accent === 'mint' ? 'pill' : 'pill-crimson'}>
                {s.accent === 'mint' ? 'GOOD' : 'NULL SIGNAL'}
              </span>
            </div>
            <div className="mt-4 stat-num tabular-nums">{s.value}</div>
            <div className="mt-2 text-sm text-slate-300">{s.sub}</div>
            <div className="mt-3 text-xs text-muted leading-relaxed">{s.explain}</div>
          </div>
        ))}
      </div>
    </section>
  )
}

function Terminal({ lines }) {
  const [shown, setShown] = useState([])
  const [current, setCurrent] = useState('')
  const [done, setDone] = useState(false)

  useEffect(() => {
    let cancelled = false
    let i = 0
    let j = 0
    const tick = () => {
      if (cancelled) return
      if (i >= lines.length) {
        setDone(true)
        return
      }
      const line = lines[i]
      j += Math.max(1, Math.floor(line.length / 40))
      if (j >= line.length) {
        setShown((s) => [...s, line])
        setCurrent('')
        i += 1
        j = 0
        setTimeout(tick, 350)
      } else {
        setCurrent(line.slice(0, j))
        setTimeout(tick, 22)
      }
    }
    tick()
    return () => {
      cancelled = true
    }
  }, [lines])

  return (
    <div className="card overflow-hidden shadow-glow-mint/30">
      <div className="flex items-center gap-1.5 px-3.5 py-2 border-b border-ink-700 bg-ink-800/60">
        <span className="h-2.5 w-2.5 rounded-full bg-crimson/70" />
        <span className="h-2.5 w-2.5 rounded-full bg-amber/70" />
        <span className="h-2.5 w-2.5 rounded-full bg-mint/70" />
        <span className="ml-3 font-mono text-[11px] text-muted">
          ~/Machine_learning_S6_2025-2026 — bash
        </span>
      </div>
      <div className="p-4 font-mono text-[12.5px] leading-6 min-h-[260px] text-slate-300">
        {shown.map((l, idx) => (
          <div key={idx} className={l.startsWith('$') ? 'text-mint' : 'text-slate-300'}>
            {l}
          </div>
        ))}
        {!done && (
          <div className={current.startsWith('$') ? 'text-mint' : 'text-slate-300'}>
            {current}
            <span className="blink" />
          </div>
        )}
        {done && (
          <div className="mt-2 text-amber">
            ▸ exit 0 · all rubric components delivered
          </div>
        )}
      </div>
    </div>
  )
}
