import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { PIVOT_B } from '../data/models.js'
import { SectionHeader } from './Playground.jsx'

export default function PivotB() {
  return (
    <section id="pivot-b" className="py-20 scroll-mt-20">
      <SectionHeader
        eyebrow="04 / pivot b"
        title="Same pipeline, real dataset"
        subtitle={
          <>
            We re-ran the <strong>exact same code</strong> on{' '}
            <span className="text-mint">PhishingWebsites</span> (OpenML 4534, 11 055 URLs, real
            cybersec signal). The Random Forest jumps from{' '}
            <span className="text-mint">F1 0.49 → F1 0.97</span> and AUC 0.50 → 0.997 — proof that
            the pipeline is correct and the bottleneck is the official dataset, not our code.
          </>
        }
      />

      <div className="mt-8 grid md:grid-cols-12 gap-5">
        {/* Headline figure */}
        <figure className="md:col-span-7 card overflow-hidden">
          <div className="flex items-center justify-between px-4 py-2.5 border-b border-ink-700 bg-ink-800/40">
            <span className="font-mono text-[11px] uppercase tracking-wider text-mint">
              the killer figure
            </span>
            <span className="pill">graphs/08</span>
          </div>
          <div className="bg-white/5 p-3">
            <img
              src="/figures/08_pivot_b_synthetic_vs_real.png"
              alt="Pivot B comparison: synthetic vs real dataset"
              className="w-full h-auto rounded"
              loading="lazy"
            />
          </div>
          <figcaption className="px-4 py-3 text-xs text-slate-300 leading-relaxed">
            Side-by-side F1 macro per model. Same preprocessing, same GridSearchCV grids, same
            random_state. The only thing that changed is the dataset.
          </figcaption>
        </figure>

        {/* Comparative chart */}
        <div className="md:col-span-5 card p-4">
          <h3 className="font-mono text-[11px] uppercase tracking-wider text-muted mb-2">
            F1 by model — official vs PhishingWebsites
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={PIVOT_B} margin={{ top: 8, right: 8, bottom: 0, left: -10 }}>
                <CartesianGrid strokeDasharray="2 4" stroke="#1f2937" />
                <XAxis
                  dataKey="model"
                  tick={{ fill: '#94a3b8', fontFamily: 'JetBrains Mono', fontSize: 10 }}
                  axisLine={{ stroke: '#1f2937' }}
                  tickLine={false}
                />
                <YAxis
                  domain={[0, 1]}
                  tick={{ fill: '#94a3b8', fontFamily: 'JetBrains Mono', fontSize: 10 }}
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
                <Legend wrapperStyle={{ fontFamily: 'JetBrains Mono', fontSize: 11 }} />
                <Bar dataKey="steve_f1" name="Official dataset" fill="#f87171" radius={[3, 3, 0, 0]} />
                <Bar dataKey="phishing_f1" name="PhishingWebsites" fill="#5eead4" radius={[3, 3, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-3 text-xs text-muted leading-relaxed">
            Crimson = official dataset (no signal). Mint = real dataset (signal recovered by the
            same pipeline).
          </p>
        </div>

        {/* Table */}
        <div className="md:col-span-12 card overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-ink-700 bg-ink-800/40">
                {['Model', 'Steve dataset · F1', 'PhishingWebsites · F1', 'PhishingWebsites · AUC', 'Δ'].map((h) => (
                  <th key={h} className="text-left px-4 py-3 font-mono text-[10px] uppercase tracking-wider text-muted">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {PIVOT_B.map((row) => {
                const delta = row.phishing_f1 - row.steve_f1
                return (
                  <tr key={row.model} className="border-b border-ink-700/40">
                    <td className="px-4 py-3 font-mono text-slate-100">{row.model}</td>
                    <td className="px-4 py-3 font-mono tabular-nums text-crimson">{row.steve_f1.toFixed(4)}</td>
                    <td className="px-4 py-3 font-mono tabular-nums text-mint">{row.phishing_f1.toFixed(4)}</td>
                    <td className="px-4 py-3 font-mono tabular-nums text-mint">{row.phishing_auc.toFixed(4)}</td>
                    <td className="px-4 py-3 font-mono tabular-nums">
                      <span className={delta > 0.3 ? 'text-mint' : 'text-muted'}>
                        +{delta.toFixed(3)}
                      </span>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Conclusion strip */}
      <div className="mt-8 card border-mint/30 p-6 bg-mint/5">
        <div className="flex items-start gap-4">
          <span className="font-mono text-mint text-2xl leading-none mt-0.5">▸</span>
          <div>
            <p className="text-slate-100 leading-relaxed">
              <strong className="text-mint">Conclusion:</strong> the same code, with no modification,
              recovers near-perfect performance on a real dataset. The 0.5 AUC observed on the official
              dataset is a property of the data, not of our implementation.
            </p>
            <p className="mt-2 text-sm text-slate-300">
              This is what we mean by <em>methodological success</em>: not pretending a flat result is
              good, but proving rigorously that the limitation is upstream of any model choice.
            </p>
          </div>
        </div>
      </div>
    </section>
  )
}
