import { useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ReferenceLine } from 'recharts'
import { MODELS } from '../data/models.js'
import { SectionHeader } from './Playground.jsx'

export default function ModelComparison() {
  const [focusModel, setFocusModel] = useState('random_forest')
  const focused = MODELS.find((m) => m.id === focusModel)

  const data = MODELS.map((m) => ({
    name: m.short,
    'Train F1': +m.train_f1.toFixed(3),
    'Test F1': +m.test_f1.toFixed(3),
    'ROC-AUC': +m.roc_auc.toFixed(3),
  }))

  return (
    <section id="models" className="py-20 scroll-mt-20">
      <SectionHeader
        eyebrow="02 / models"
        title="Six models, one verdict"
        subtitle={
          <>
            Baseline + linear + tree + random forest + voting + stacking. Three families, two
            ensemble strategies. <span className="text-mint">Test F1 converges around 0.49</span>;
            ROC-AUC hovers at 0.50. The Random Forest reaches train F1 = 0.91 — capacity is not the
            bottleneck.
          </>
        }
      />

      {/* Chart */}
      <div className="mt-8 card p-5">
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-mono text-xs uppercase tracking-wider text-muted">
            train_f1 / test_f1 / roc_auc — 70/30 stratified split, random_state=42
          </h3>
          <div className="flex items-center gap-2 text-[11px] font-mono text-muted">
            <Dot color="#5eead4" /> Test F1
            <Dot color="#fbbf24" /> Train F1
            <Dot color="#a78bfa" /> ROC-AUC
          </div>
        </div>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} margin={{ top: 8, right: 12, bottom: 0, left: -10 }}>
              <CartesianGrid strokeDasharray="2 4" stroke="#1f2937" />
              <XAxis
                dataKey="name"
                tick={{ fill: '#94a3b8', fontFamily: 'JetBrains Mono', fontSize: 11 }}
                axisLine={{ stroke: '#1f2937' }}
                tickLine={false}
              />
              <YAxis
                domain={[0, 1]}
                tick={{ fill: '#94a3b8', fontFamily: 'JetBrains Mono', fontSize: 11 }}
                axisLine={{ stroke: '#1f2937' }}
                tickLine={false}
              />
              <Tooltip
                contentStyle={{
                  background: '#0b1117',
                  border: '1px solid #1f2937',
                  borderRadius: 8,
                  fontFamily: 'JetBrains Mono',
                  fontSize: 12,
                }}
                cursor={{ fill: 'rgba(94,234,212,0.05)' }}
              />
              <ReferenceLine y={0.5} stroke="#f87171" strokeDasharray="3 3" label={{ value: 'AUC random = 0.50', fill: '#f87171', fontSize: 10, fontFamily: 'JetBrains Mono', position: 'right' }} />
              <Bar dataKey="Train F1" fill="#fbbf24" radius={[3, 3, 0, 0]} />
              <Bar dataKey="Test F1" fill="#5eead4" radius={[3, 3, 0, 0]} />
              <Bar dataKey="ROC-AUC" fill="#a78bfa" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="mt-4 text-xs text-muted leading-relaxed max-w-3xl">
          <span className="text-mint">Read it like this:</span> the gap between Train F1 (amber) and
          Test F1 (mint) tells you about overfitting — RF memorizes (gap = +0.42), Stacking generalizes
          most honestly (gap = +0.009). But every test ROC-AUC sits right on the dashed crimson line
          at 0.50 — the model can't separate the classes in any meaningful way.
        </p>
      </div>

      {/* Table */}
      <div className="mt-6 card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-ink-700 bg-ink-800/50">
                {['Model', 'Family', 'Train F1', 'Test F1', 'Acc', 'ROC-AUC', 'Train−Test', ''].map(
                  (h) => (
                    <th
                      key={h}
                      className="text-left px-4 py-3 font-mono text-[10px] uppercase tracking-wider text-muted"
                    >
                      {h}
                    </th>
                  )
                )}
              </tr>
            </thead>
            <tbody>
              {MODELS.map((m) => {
                const isActive = m.id === focusModel
                return (
                  <tr
                    key={m.id}
                    className={`border-b border-ink-700/50 transition-colors cursor-pointer ${
                      isActive ? 'bg-mint/5' : 'hover:bg-ink-800/40'
                    }`}
                    onClick={() => setFocusModel(m.id)}
                  >
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        {m.isChampion && (
                          <span className="h-1.5 w-1.5 rounded-full bg-mint pulse-dot" />
                        )}
                        <span className="font-mono text-slate-100">{m.short}</span>
                      </div>
                    </td>
                    <td className="px-4 py-3 font-mono text-xs text-muted">{m.family}</td>
                    <td className="px-4 py-3 font-mono tabular-nums text-amber">
                      {m.train_f1.toFixed(3)}
                    </td>
                    <td className="px-4 py-3 font-mono tabular-nums text-mint">
                      {m.test_f1.toFixed(3)}
                    </td>
                    <td className="px-4 py-3 font-mono tabular-nums text-slate-200">
                      {m.accuracy.toFixed(3)}
                    </td>
                    <td className="px-4 py-3 font-mono tabular-nums text-slate-200">
                      {m.roc_auc.toFixed(3)}
                    </td>
                    <td className="px-4 py-3 font-mono tabular-nums">
                      <span className={m.gap > 0.1 ? 'text-crimson' : m.gap > 0.05 ? 'text-amber' : 'text-mint'}>
                        {m.gap >= 0 ? '+' : ''}
                        {m.gap.toFixed(3)}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-right">
                      <span className="font-mono text-[10px] text-muted">
                        {isActive ? '◉' : '○'}
                      </span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Selected model figures */}
      <div className="mt-6 grid md:grid-cols-2 gap-5">
        {focused.cmFigure && (
          <FigureCard
            title="Confusion matrix"
            sub={focused.name}
            src={focused.cmFigure}
            alt={`Confusion matrix for ${focused.name}`}
          />
        )}
        {focused.rocFigure && (
          <FigureCard
            title="ROC curve"
            sub={`AUC = ${focused.roc_auc.toFixed(3)}`}
            src={focused.rocFigure}
            alt={`ROC curve for ${focused.name}`}
          />
        )}
        {!focused.rocFigure && focused.cmFigure && (
          <div className="card p-6 flex items-center justify-center text-muted text-sm">
            Baseline DummyClassifier — ROC undefined (single-class output).
          </div>
        )}
      </div>

      {/* Summary cards */}
      <div className="mt-6 grid md:grid-cols-3 gap-4">
        <Insight
          tag="convergence"
          color="crimson"
          text="6 models converge around F1 ≈ 0.49, AUC ≈ 0.50."
        />
        <Insight
          tag="capacity"
          color="amber"
          text="RF train F1 = 0.91 → the model can memorize. Capacity isn't the issue."
        />
        <Insight
          tag="conclusion"
          color="mint"
          text="Convergence + unsaturated capacity ⇒ investigate the data, not the models."
        />
      </div>
    </section>
  )
}

function FigureCard({ title, sub, src, alt }) {
  return (
    <div className="card overflow-hidden">
      <div className="flex items-center justify-between px-4 py-2.5 border-b border-ink-700 bg-ink-800/40">
        <div>
          <div className="font-mono text-[11px] uppercase tracking-wider text-mint">{title}</div>
          <div className="font-mono text-[11px] text-muted">{sub}</div>
        </div>
      </div>
      <div className="bg-white/5 p-3">
        <img src={src} alt={alt} className="w-full h-auto rounded" loading="lazy" />
      </div>
    </div>
  )
}

function Insight({ tag, color, text }) {
  const cls = {
    mint: 'pill',
    amber: 'pill-amber',
    crimson: 'pill-crimson',
  }[color]
  return (
    <div className="card p-5">
      <span className={cls}>{tag}</span>
      <p className="mt-3 text-sm text-slate-200 leading-relaxed">{text}</p>
    </div>
  )
}

function Dot({ color }) {
  return (
    <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: color }} />
  )
}
