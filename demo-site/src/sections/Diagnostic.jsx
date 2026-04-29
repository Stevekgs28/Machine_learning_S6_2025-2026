import { DIAGNOSTIC_TESTS } from '../data/models.js'
import { SectionHeader } from './Playground.jsx'

const FIGURES = [
  {
    src: '/figures/01_loss_distribution.png',
    title: 'Financial Loss distribution',
    caption:
      'Histogram + uniform overlay. KS-test p = 0.804 → cannot reject the hypothesis that losses are drawn from a uniform[0.5, 99.99] distribution. The target is built by thresholding this column at Q3, so it inherits the same lack of structure.',
    test: 'KS-test',
  },
  {
    src: '/figures/02_mutual_info_numeric.png',
    title: 'Mutual information — numeric features',
    caption:
      'MI of all numeric columns vs target. Threshold for "weak" is 0.02 — every feature sits below. Year and # Affected Users have MI = 0.000.',
    test: 'MI',
  },
  {
    src: '/figures/03_cramers_v_categorical.png',
    title: "Cramér's V — categorical features",
    caption:
      'Max V = 0.058 (Country); χ² p-value ≥ 0.34 everywhere. Independence between every categorical and the target cannot be rejected at α=0.05.',
    test: "χ²",
  },
  {
    src: '/figures/04_pos_rate_top_categories.png',
    title: 'Positive rate by modality',
    caption:
      'Per-modality positive rate with 95% CI for the top categories. Every CI overlaps the 0.25 baseline — no modality predicts the target above chance.',
    test: '95% CI',
  },
  {
    src: '/figures/05_train_vs_test_overfit.png',
    title: 'Train vs Test F1',
    caption:
      'Visual diagnostic of overfitting across the 4 base models. RF\'s gap (+0.42) is the loudest signal that the model can memorize but not generalize on this data.',
    test: 'overfit',
  },
  {
    src: '/figures/06_shap_mean_abs.png',
    title: 'SHAP mean |value| — Random Forest',
    caption:
      'Top 20 features by mean |SHAP|. Max value = 0.0171, importance is distributed across many features instead of concentrated on 1–3 dominant ones — the signature of a no-signal dataset.',
    test: 'SHAP',
  },
  {
    src: '/figures/07_shap_summary_beeswarm.png',
    title: 'SHAP summary beeswarm',
    caption:
      'Per-sample SHAP values. The cloud is uniform across features — no clear "high value pushes prediction up" patterns emerge.',
    test: 'SHAP',
  },
]

export default function Diagnostic() {
  return (
    <section id="diagnostic" className="py-20 scroll-mt-20">
      <SectionHeader
        eyebrow="03 / diagnostic"
        title="Why all six models plateau"
        subtitle={
          <>
            Four convergent statistical tests, plus SHAP, applied to instrument the dataset.
            The verdict is unambiguous: <span className="text-mint">no exploitable signal</span> —
            and the proof is in the numbers below.
          </>
        }
      />

      {/* Hypothesis grid */}
      <div className="mt-8 grid md:grid-cols-2 gap-4">
        <HypoCard
          n="H1"
          title="Insufficient capacity"
          status="pre-refuted"
          color="muted"
          body={
            <>
              Random Forest reaches train F1 = <span className="text-amber">0.910</span>. Capacity
              is not saturated — the model can fit, it just can't generalize.
            </>
          }
        />
        <HypoCard
          n="H2"
          title="Inadequate preprocessing"
          status="pre-refuted"
          color="muted"
          body={
            <>
              Standard sklearn pipeline: median imputer + StandardScaler for numerics,
              most_frequent + OneHot for categoricals, class_weight='balanced'. Validated by
              Pivot B reaching F1 0.97 on a real dataset with the same code.
            </>
          }
        />
        <HypoCard
          n="H3"
          title="Non-informative features"
          status="confirmed"
          color="crimson"
          body={
            <>
              MI ≤ 0.006 on all numerics, Cramér's V max = 0.058 on categoricals, χ² p ≥ 0.34
              everywhere. <span className="text-mint">SHAP confirms independently</span>: importance
              spread across 44 OneHot features with mean |SHAP| max = 0.0171.
            </>
          }
        />
        <HypoCard
          n="H4"
          title="Structurally random dataset"
          status="confirmed"
          color="crimson"
          body={
            <>
              KS-test of Financial Loss vs Uniform[0.5, 99.99]:{' '}
              <span className="text-mint">D = 0.012, p = 0.804</span>. Observed std 28.79 ≈ uniform
              theoretical 28.72. Direct regression on Financial Loss → R² test{' '}
              <span className="text-crimson">negative</span>.
            </>
          }
        />
      </div>

      {/* Tests table */}
      <div className="mt-8 card overflow-hidden">
        <div className="px-5 py-3 border-b border-ink-700 bg-ink-800/40 flex items-center justify-between">
          <h3 className="font-mono text-[11px] uppercase tracking-wider text-mint">
            Statistical test summary
          </h3>
          <span className="font-mono text-[10px] text-muted">α = 0.05</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-ink-700">
                {['Test', 'Target', 'Statistic', 'p-value', 'Verdict'].map((h) => (
                  <th
                    key={h}
                    className="text-left px-4 py-3 font-mono text-[10px] uppercase tracking-wider text-muted"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {DIAGNOSTIC_TESTS.map((t) => (
                <tr key={t.test} className="border-b border-ink-700/40 hover:bg-ink-800/30 transition-colors">
                  <td className="px-4 py-3 font-mono text-slate-100">{t.test}</td>
                  <td className="px-4 py-3 text-slate-300">{t.target}</td>
                  <td className="px-4 py-3 font-mono tabular-nums text-amber">{t.statistic}</td>
                  <td className="px-4 py-3 font-mono tabular-nums text-slate-200">{t.pvalue}</td>
                  <td className="px-4 py-3 text-sm">
                    <span className={t.severity === 'critical' ? 'text-crimson' : 'text-amber'}>
                      ▸{' '}
                    </span>
                    <span className="text-slate-200">{t.verdict}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Figures */}
      <div className="mt-8 grid md:grid-cols-2 gap-5">
        {FIGURES.map((f) => (
          <figure key={f.src} className="card overflow-hidden">
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-ink-700 bg-ink-800/40">
              <span className="font-mono text-[11px] uppercase tracking-wider text-mint">
                {f.title}
              </span>
              <span className="pill-amber">{f.test}</span>
            </div>
            <div className="bg-white/5 p-3">
              <img src={f.src} alt={f.title} className="w-full h-auto rounded" loading="lazy" />
            </div>
            <figcaption className="px-4 py-3 text-xs text-slate-300 leading-relaxed">
              {f.caption}
            </figcaption>
          </figure>
        ))}
      </div>
    </section>
  )
}

function HypoCard({ n, title, status, color, body }) {
  const pill = {
    crimson: 'pill-crimson',
    amber: 'pill-amber',
    mint: 'pill',
    muted: 'inline-flex items-center gap-1.5 rounded-full border border-ink-600 bg-ink-800/60 px-2.5 py-0.5 text-[11px] font-mono uppercase tracking-wider text-muted',
  }[color]
  return (
    <div className="card p-5">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="font-mono text-[11px] text-muted">{n}</div>
          <div className="mt-1 font-display text-lg text-white leading-tight">{title}</div>
        </div>
        <span className={pill}>{status}</span>
      </div>
      <p className="mt-3 text-sm text-slate-300 leading-relaxed">{body}</p>
    </div>
  )
}
