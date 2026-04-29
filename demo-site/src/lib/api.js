// Wrapper minimal pour l'API FastAPI exposée par code/api.py.
// Lancer en local : `uvicorn api:app --reload --host 0.0.0.0 --port 8000` depuis code/
// ou via Docker : `docker run -p 8000:8000 steve-ml-api`.

const DEFAULT_API_BASE = 'http://localhost:8000'

export function getApiBase() {
  // Permet de surcharger via ?api=... dans l'URL pour démo (ngrok, IP locale, etc.)
  const params = new URLSearchParams(window.location.search)
  const fromQuery = params.get('api')
  if (fromQuery) return fromQuery.replace(/\/$/, '')
  // Permet aussi de forcer via localStorage
  try {
    const fromStorage = localStorage.getItem('api_base')
    if (fromStorage) return fromStorage.replace(/\/$/, '')
  } catch {}
  return DEFAULT_API_BASE
}

export async function checkHealth() {
  const base = getApiBase()
  const ctrl = new AbortController()
  const t = setTimeout(() => ctrl.abort(), 2500)
  try {
    const res = await fetch(`${base}/health`, { signal: ctrl.signal })
    clearTimeout(t)
    if (!res.ok) throw new Error(`HTTP ${res.status}`)
    return await res.json()
  } catch (err) {
    clearTimeout(t)
    return { status: 'unreachable', error: err.message }
  }
}

export async function predict(payload) {
  const base = getApiBase()
  const res = await fetch(`${base}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  })
  if (!res.ok) {
    const text = await res.text().catch(() => '')
    throw new Error(`API ${res.status}: ${text || res.statusText}`)
  }
  return res.json()
}
