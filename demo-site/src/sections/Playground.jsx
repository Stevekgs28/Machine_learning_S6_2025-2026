import { useState } from 'react'
import { FORM_OPTIONS, FORM_DEFAULTS, MODELS } from '../data/models.js'
import { predict, getApiBase } from '../lib/api.js'

const NUMERIC_FIELDS = [
  { name: 'Year', min: 2015, max: 2030, step: 1 },
  { name: 'Number of Affected Users', min: 0, max: 1_000_000, step: 1 },
  { name: 'Incident Resolution Time (in Hours)', min: 0, max: 200, step: 1 },
]

const CATEGORICAL_FIELDS = [
  'Country',
  'Attack Type',
  'Target Industry',
  'Attack Source',
  'Security Vulnerability Type',
  'Defense Mechanism Used',
]

export default function Playground({ apiStatus }) {
  const [form, setForm] = useState(FORM_DEFAULTS)
  const [selectedModelId, setSelectedModelId] = useState('random_forest')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const apiOk = apiStatus.status === 'ok' && apiStatus.model_loaded
  const selectedModel = MODELS.find((m) => m.id === selectedModelId)

  const handleChange = (k, v) => {
    setForm((f) => ({ ...f, [k]: v }))
    setResult(null)
    setError(null)
  }

  const onSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      // Construire payload avec les noms de colonnes exacts attendus par /predict
      const payload = {
        Country: form.Country,
        Year: Number(form.Year),
        'Attack Type': form['Attack Type'],
        'Target Industry': form['Target Industry'],
        'Number of Affected Users': Number(form['Number of Affected Users']),
        'Attack Source': form['Attack Source'],
        'Security Vulnerability Type': form['Security Vulnerability Type'],
        'Defense Mechanism Used': form['Defense Mechanism Used'],
        'Incident Resolution Time (in Hours)': Number(form['Incident Resolution Time (in Hours)']),
      }
      const data = await predict(payload)
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const onRandom = () => {
    const rand = (arr) => arr[Math.floor(Math.random() * arr.length)]
    setForm({
      Country: rand(FORM_OPTIONS.Country),
      Year: 2015 + Math.floor(Math.random() * 10),
      'Attack Type': rand(FORM_OPTIONS['Attack Type']),
      'Target Industry': rand(FORM_OPTIONS['Target Industry']),
      'Number of Affected Users': Math.floor(Math.random() * 999_000) + 1000,
      'Attack Source': rand(FORM_OPTIONS['Attack Source']),
      'Security Vulnerability Type': rand(FORM_OPTIONS['Security Vulnerability Type']),
      'Defense Mechanism Used': rand(FORM_OPTIONS['Defense Mechanism Used']),
      'Incident Resolution Time (in Hours)': Math.floor(Math.random() * 72) + 1,
    })
    setResult(null)
    setError(null)
  }

  return (
    <section id="playground" className="py-20 scroll-mt-20">
      <SectionHeader
        eyebrow="01 / playground"
        title="Live prediction"
        subtitle={
          <>
            Hit <span className="text-mint">/predict</span> on the FastAPI backend exposing the
            serialized Random Forest. The form mirrors the 9-field <code className="text-mint">IncidentInput</code>{' '}
            schema in <code className="text-mint">code/api.py</code>.
          </>
        }
      />

      {/* Model selector */}
      <div className="mt-8 card p-5">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <div className="font-mono text-[11px] uppercase tracking-wider text-muted">
              Champion model
            </div>
            <div className="mt-1 text-sm text-slate-300">
              Switch to inspect each model's metrics. The live <code className="text-mint">/predict</code> call always
              hits the serialized <span className="text-mint">Random Forest</span> — but on this dataset all six models
              converge at AUC ≈ 0.5, so the prediction wouldn't change meaningfully.
            </div>
          </div>
        </div>
        <div className="mt-4 grid grid-cols-2 md:grid-cols-6 gap-2">
          {MODELS.map((m) => (
            <button
              key={m.id}
              onClick={() => setSelectedModelId(m.id)}
              className={`relative rounded-lg border px-3 py-2.5 text-left transition-all ${
                selectedModelId === m.id
                  ? 'border-mint/60 bg-mint/10 shadow-glow-mint'
                  : 'border-ink-600 bg-ink-900/60 hover:border-mint/30'
              }`}
            >
              {m.isChampion && (
                <span className="absolute -top-1.5 -right-1.5 h-3 w-3 rounded-full bg-mint shadow-glow-mint" />
              )}
              <div className="font-mono text-[10px] uppercase tracking-wider text-muted">
                {m.family}
              </div>
              <div className="mt-0.5 font-mono text-xs text-slate-100 truncate">{m.short}</div>
              <div className="mt-1 font-mono text-[11px] text-mint tabular-nums">
                F1 {m.test_f1.toFixed(3)}
              </div>
            </button>
          ))}
        </div>
        <ModelExplain model={selectedModel} />
      </div>

      {/* Form + result */}
      <form onSubmit={onSubmit} className="mt-6 grid md:grid-cols-12 gap-6">
        {/* Form */}
        <div className="md:col-span-7 card p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-display text-lg text-white">Incident profile</h3>
            <button type="button" onClick={onRandom} className="btn-ghost !py-1.5 !px-3 text-xs">
              ⚄ Random
            </button>
          </div>

          <div className="grid sm:grid-cols-2 gap-4">
            {CATEGORICAL_FIELDS.map((f) => (
              <div key={f}>
                <label className="label-ml">{f}</label>
                <select
                  className="input-ml"
                  value={form[f]}
                  onChange={(e) => handleChange(f, e.target.value)}
                >
                  {FORM_OPTIONS[f].map((opt) => (
                    <option key={opt} value={opt}>
                      {opt}
                    </option>
                  ))}
                </select>
              </div>
            ))}
            {NUMERIC_FIELDS.map((f) => (
              <div key={f.name}>
                <label className="label-ml">{f.name}</label>
                <input
                  type="number"
                  className="input-ml"
                  min={f.min}
                  max={f.max}
                  step={f.step}
                  value={form[f.name]}
                  onChange={(e) => handleChange(f.name, e.target.value)}
                />
              </div>
            ))}
          </div>

          <div className="mt-6 flex items-center gap-3">
            <button type="submit" disabled={loading || !apiOk} className="btn-primary">
              {loading ? '⏳ Predicting…' : '▶ Predict'}
            </button>
            {!apiOk && (
              <span className="font-mono text-[11px] text-crimson">
                API offline — start backend on {getApiBase()}
              </span>
            )}
          </div>
        </div>

        {/* Result */}
        <div className="md:col-span-5">
          <ResultPanel
            loading={loading}
            error={error}
            result={result}
            apiOk={apiOk}
          />
        </div>
      </form>

      {/* API help */}
      <div className="mt-5 card p-4 text-xs font-mono text-muted">
        <div className="terminal-prompt text-mint">cd code &amp;&amp; uvicorn api:app --reload --port 8000</div>
        <div className="terminal-prompt mt-1">docker run -p 8000:8000 steve-ml-api</div>
        <div className="mt-2 text-[11px]">
          API base : <span className="text-mint">{getApiBase()}</span> · override via{' '}
          <code className="text-mint">?api=https://...</code> in URL or{' '}
          <code className="text-mint">localStorage.api_base</code>.
        </div>
      </div>
    </section>
  )
}

function ModelExplain({ model }) {
  return (
    <div className="mt-4 rounded-lg border border-ink-700 bg-ink-950/50 p-4 text-sm">
      <div className="flex flex-wrap items-baseline gap-x-5 gap-y-1">
        <span className="font-display text-white text-base">{model.name}</span>
        <span className="font-mono text-[11px] text-muted">family: {model.family}</span>
        {model.isChampion && <span className="pill">SERIALIZED · best_model.joblib</span>}
      </div>
      <div className="mt-3 grid grid-cols-2 md:grid-cols-5 gap-2 text-[11px] font-mono">
        <Stat label="train F1" value={model.train_f1.toFixed(3)} />
        <Stat label="test F1" value={model.test_f1.toFixed(3)} highlight />
        <Stat label="accuracy" value={model.accuracy.toFixed(3)} />
        <Stat label="ROC-AUC" value={model.roc_auc.toFixed(3)} />
        <Stat label="train−test" value={(model.gap >= 0 ? '+' : '') + model.gap.toFixed(3)} />
      </div>
      <p className="mt-3 text-slate-300 leading-relaxed">{model.notes}</p>
    </div>
  )
}

function Stat({ label, value, highlight }) {
  return (
    <div className="rounded-md border border-ink-700 bg-ink-900/70 px-2.5 py-2">
      <div className="text-[10px] uppercase tracking-wider text-muted">{label}</div>
      <div className={`mt-0.5 tabular-nums ${highlight ? 'text-mint' : 'text-slate-100'}`}>
        {value}
      </div>
    </div>
  )
}

function ResultPanel({ loading, error, result, apiOk }) {
  if (loading) {
    return (
      <div className="card p-6 h-full flex items-center justify-center">
        <div className="font-mono text-mint text-sm">
          ⏳ POST /predict
          <span className="blink" />
        </div>
      </div>
    )
  }
  if (error) {
    return (
      <div className="card border-crimson/40 p-6">
        <div className="font-mono text-[11px] uppercase tracking-wider text-crimson mb-2">
          ✗ error
        </div>
        <div className="font-mono text-sm text-slate-200 break-words">{error}</div>
        <div className="mt-3 text-xs text-muted">
          Check that <code className="text-mint">uvicorn api:app</code> is running and CORS allows
          the origin.
        </div>
      </div>
    )
  }
  if (!result) {
    return (
      <div className="card p-6 h-full flex flex-col justify-center">
        <div className="font-mono text-[11px] uppercase tracking-wider text-muted mb-2">
          awaiting input
        </div>
        <div className="text-sm text-slate-300">
          Fill the incident profile and hit <span className="text-mint font-mono">Predict</span> to
          get a binary classification + probability of high financial impact (≥ Q3 of losses).
        </div>
        {!apiOk && (
          <div className="mt-4 text-xs text-amber">
            ⚠ The API isn't reachable right now. Start the FastAPI backend to enable predictions.
          </div>
        )}
      </div>
    )
  }

  const isHigh = result.prediction === 1
  const proba = result.probability_high_impact
  const pct = proba !== null && proba !== undefined ? (proba * 100).toFixed(1) : null

  return (
    <div className={`card p-6 ${isHigh ? 'border-amber/50' : 'border-mint/40'}`}>
      <div className="flex items-center justify-between">
        <span className={isHigh ? 'pill-amber' : 'pill'}>
          {isHigh ? 'CLASS 1' : 'CLASS 0'}
        </span>
        <span className="font-mono text-[10px] text-muted">
          model: {result.model_name}
        </span>
      </div>
      <div className="mt-4 font-display text-2xl text-white leading-tight">
        {result.label}
      </div>
      {pct !== null && (
        <div className="mt-4">
          <div className="flex justify-between font-mono text-[11px] text-muted mb-1.5">
            <span>P(High Financial Impact)</span>
            <span className="text-slate-100">{pct}%</span>
          </div>
          <div className="h-2 rounded-full bg-ink-700 overflow-hidden">
            <div
              className={`h-full rounded-full ${isHigh ? 'bg-amber' : 'bg-mint'}`}
              style={{ width: `${pct}%` }}
            />
          </div>
        </div>
      )}
      <div className="mt-5 rounded-md border border-amber/30 bg-amber/5 p-3 text-[11px] text-amber leading-relaxed">
        ⚠ Model trained on a dataset shown to be statistically random (KS p=0.804, MI ≈ 0).
        Predictions are <strong>not operational</strong> — the API exists to validate the deployment chain.
        See <a href="#diagnostic" className="underline">Diagnostic</a>.
      </div>
    </div>
  )
}

export function SectionHeader({ eyebrow, title, subtitle }) {
  return (
    <div className="max-w-3xl">
      <div className="section-eyebrow">{eyebrow}</div>
      <h2 className="mt-2 section-title">{title}</h2>
      {subtitle && <p className="mt-3 text-slate-300 leading-relaxed">{subtitle}</p>}
    </div>
  )
}
