import { TEAM, RUBRIC } from '../data/models.js'
import { SectionHeader } from './Playground.jsx'

const STACK = [
  { name: 'scikit-learn', role: 'pipeline + 4 base models + 2 ensembles', tag: 'core' },
  { name: 'SHAP', role: 'TreeExplainer for interpretability', tag: 'core' },
  { name: 'MLflow', role: '2 experiments, 10 runs, params + metrics + artifacts', tag: 'core' },
  { name: 'FastAPI + Pydantic', role: '/predict, /health, auto Swagger /docs', tag: 'deploy' },
  { name: 'Docker', role: 'python:3.11-slim, healthcheck on /health', tag: 'deploy' },
  { name: 'matplotlib + scipy', role: 'EDA figures, KS / χ² / Cramér\'s V', tag: 'analysis' },
  { name: 'React + Vite + Tailwind', role: 'this demo site', tag: 'demo' },
  { name: 'Recharts', role: 'interactive model comparison + Pivot B', tag: 'demo' },
]

export default function About() {
  return (
    <section id="about" className="py-20 scroll-mt-20">
      <SectionHeader
        eyebrow="05 / about"
        title="Team, stack & rubric coverage"
        subtitle="MOD10 Machine Learning, Winter 2026 · Instructor: Mohammed A. Shehab · 30% of final grade."
      />

      <div className="mt-8 grid md:grid-cols-12 gap-5">
        {/* Team */}
        <div className="md:col-span-5 card p-6">
          <h3 className="font-mono text-[11px] uppercase tracking-wider text-mint mb-4">Team</h3>
          <ul className="space-y-2.5">
            {TEAM.map((m) => (
              <li key={m.name} className="flex items-center gap-3">
                <span className="font-mono text-mint">›</span>
                <span className="font-display text-base text-slate-100">{m.name}</span>
              </li>
            ))}
          </ul>
          <div className="mt-6 pt-6 border-t border-ink-700">
            <h4 className="font-mono text-[11px] uppercase tracking-wider text-muted mb-3">
              Repo
            </h4>
            <div className="font-mono text-xs text-slate-300 leading-relaxed">
              <span className="text-mint">~/</span>Machine_learning_S6_2025-2026/
              <ul className="mt-2 space-y-1 text-muted pl-4 border-l border-ink-700">
                <li>├── code/ <span className="text-mint">→ pipeline + api</span></li>
                <li>├── graphs/ <span className="text-mint">→ EDA + SHAP figures</span></li>
                <li>├── mlruns/ <span className="text-mint">→ 10 MLflow runs</span></li>
                <li>├── saved_models/best_model.joblib</li>
                <li>├── Dockerfile</li>
                <li>└── rapport_draft.pdf</li>
              </ul>
            </div>
          </div>
        </div>

        {/* Rubric */}
        <div className="md:col-span-7 card p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-mono text-[11px] uppercase tracking-wider text-mint">
              Rubric coverage
            </h3>
            <span className="pill">8 / 8 components</span>
          </div>
          <ul className="space-y-2.5">
            {RUBRIC.map((r) => (
              <li
                key={r.component}
                className="grid grid-cols-12 gap-3 items-center rounded-md border border-ink-700/60 bg-ink-900/40 px-3 py-2.5 hover:border-mint/30 transition-colors"
              >
                <div className="col-span-1 font-mono text-mint">✓</div>
                <div className="col-span-5">
                  <div className="font-display text-sm text-slate-100">{r.component}</div>
                  <div className="font-mono text-[11px] text-muted mt-0.5">
                    weight: {r.weight}%
                  </div>
                </div>
                <div className="col-span-6 font-mono text-xs text-slate-300">{r.artifact}</div>
              </li>
            ))}
          </ul>
        </div>

        {/* Stack */}
        <div className="md:col-span-12 card p-6">
          <h3 className="font-mono text-[11px] uppercase tracking-wider text-mint mb-4">
            Stack
          </h3>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {STACK.map((s) => (
              <div
                key={s.name}
                className="rounded-lg border border-ink-700 bg-ink-900/50 p-4 hover:border-mint/30 transition-colors"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="font-display text-sm text-slate-100">{s.name}</span>
                  <span
                    className={
                      s.tag === 'core'
                        ? 'pill'
                        : s.tag === 'deploy'
                        ? 'pill-amber'
                        : 'pill-crimson'
                    }
                  >
                    {s.tag}
                  </span>
                </div>
                <p className="text-[12px] text-slate-300 leading-relaxed">{s.role}</p>
              </div>
            ))}
          </div>
        </div>

        {/* Quotables */}
        <div className="md:col-span-12 card border-mint/30 p-6 bg-gradient-to-br from-mint/5 to-transparent">
          <div className="font-mono text-[11px] uppercase tracking-wider text-mint mb-3">
            The thesis, one sentence
          </div>
          <blockquote className="font-display text-xl md:text-2xl text-white leading-relaxed">
            "Identifying when ML is the wrong tool, while still delivering every component the
            rubric requires, is a methodological success."
          </blockquote>
        </div>
      </div>
    </section>
  )
}
