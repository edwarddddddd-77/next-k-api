/** API helper — prefix allows Next K reverse-proxy under /api/clawby-quant */
const PREFIX = (typeof window !== 'undefined' && window.__CLAWBY_API_PREFIX__) || ''

export const apiPath = (path) => `${PREFIX}${path}`

const get = async (path) => {
  const r = await fetch(apiPath(path))
  if (!r.ok) throw new Error(`${path}: ${r.status}`)
  return r.json()
}

const post = async (path, body) => {
  const r = await fetch(apiPath(path), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  })
  const d = await r.json().catch(() => ({}))
  if (!r.ok) throw new Error(d.detail || `${path}: ${r.status}`)
  return d
}

export const api = {
  status: () => get('/api/status'),
  positions: () => get('/api/positions'),
  closePosition: (pid) => post(`/api/positions/${pid}/close`),
  closeAll: () => post('/api/positions/close-all'),
  trades: (limit = 100) => get(`/api/trades?limit=${limit}`),
  signals: (limit = 50) => get(`/api/signals?limit=${limit}`),
  equity: () => get('/api/equity'),
  factors: () => get('/api/factors'),
  factorConfig: () => get('/api/factor-config'),
  setFactorConfig: (name, payload) => post(`/api/factor-config/${name}`, payload),
  strategies: () => get('/api/strategies'),
  toggleStrategy: (id, enabled) => post(`/api/strategies/${id}/toggle`, { enabled }),
  strategyConfig: (id, payload) => post(`/api/strategies/${id}/config`, payload),
  createStrategy: (base, name) => post('/api/strategies/create', { base, name }),
  deleteStrategy: async (id) => {
    const r = await fetch(apiPath(`/api/strategies/${id}`), { method: 'DELETE' })
    const d = await r.json().catch(() => ({}))
    if (!r.ok) throw new Error(d.detail || `${r.status}`)
    return d
  },
  journal: (q = {}) => get(`/api/journal?${new URLSearchParams(q)}`),
  risk: () => get('/api/risk'),
  setGlobalRisk: (payload) => post('/api/risk/global', payload),
  universe: () => get('/api/universe'),
  setUniverse: (symbols) => post('/api/universe', { symbols }),
  credentials: () => get('/api/credentials'),
  setCredentials: (payload) => post('/api/credentials', payload),
  testCredentials: (which) => post('/api/credentials/test', { which }),
  setMode: (mode, confirm) => post('/api/mode', { mode, confirm }),
  klines: (symbol, interval = '15m') => get(`/api/klines?symbol=${symbol}&interval=${interval}`),
  logs: (limit = 100) => get(`/api/logs?limit=${limit}`),
}

export const fmtUsd = (v, digits = 2) =>
  v == null ? '-' : `$${Number(v).toLocaleString('en-US', { maximumFractionDigits: digits })}`

export const fmtPnl = (v) =>
  v == null ? '-' : `${v >= 0 ? '+' : ''}${Number(v).toFixed(2)}`

export const fmtDur = (sec) => {
  if (sec < 3600) return `${Math.floor(sec / 60)}m`
  return `${Math.floor(sec / 3600)}h${Math.floor((sec % 3600) / 60)}m`
}
