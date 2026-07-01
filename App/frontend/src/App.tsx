import { useState, useEffect } from 'react'
import './App.css'

const API = 'http://localhost:8001'

interface Options {
  regiones: string[]
  sectores: string[]
  tipos_adquisicion: string[]
}

interface Prediction {
  key: string
  name: string
  avatar: string
  description: string
  style: string
  threshold: number
  probability: number
  verdict: string
  selected: boolean
  model_precision: number
  model_recall: number
  model_f1: number
}

interface Comparison {
  field: string
  value: number
  median: number
  ratio: number
  status: string
}

interface Recommendations {
  tips: string[]
  comparisons: Comparison[]
}

interface PredictResponse {
  probability: number
  predictions: Prediction[]
  recommendations: Recommendations
}

interface FormData {
  monto_estimado: string
  valor_total_ofertado: string
  numero_oferentes: string
  cantidad_reclamos: string
  region: string
  sector: string
  tipo_adquisicion: string
}

interface CsvRow {
  nombre: string
  monto_estimado: number
  valor_total_ofertado: number
  numero_oferentes: number
  cantidad_reclamos: number
  region: string
  sector: string
  tipo_adquisicion: string
  proveedor: string
  resultado_real: string
}

interface Desierta {
  codigo: string
  nombre: string
  monto_estimado: number
  numero_oferentes: number
  cantidad_reclamos: number
  region: string
  sector: string
  tipo_adquisicion: string
  n_ofertas: number
  valor_sugerido: number
  prob_sugerida: number
  aprueba_riesgoso: boolean
  aprueba_moderado: boolean
  aprueba_conservador: boolean
}

const profileClass: Record<string, string> = {
  riesgoso: 'risk',
  moderado: 'mod',
  conservador: 'cons',
}

function fmt(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(1)}B`
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(0)}K`
  return n.toFixed(0)
}

function fmtPesos(n: number): string {
  return '$' + n.toLocaleString('es-CL', { maximumFractionDigits: 0 })
}

type Tab = 'predictor' | 'oportunidades'

function App() {
  const [activeTab, setActiveTab] = useState<Tab>('predictor')
  const [options, setOptions] = useState<Options | null>(null)
  const [form, setForm] = useState<FormData>({
    monto_estimado: '',
    valor_total_ofertado: '',
    numero_oferentes: '',
    cantidad_reclamos: '0',
    region: '',
    sector: '',
    tipo_adquisicion: '',
  })
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const [csvRows, setCsvRows] = useState<CsvRow[]>([])
  const [csvTotal, setCsvTotal] = useState(0)
  const [csvPage, setCsvPage] = useState(1)
  const [csvLoading, setCsvLoading] = useState(false)
  const [selectedCsv, setSelectedCsv] = useState<CsvRow | null>(null)
  const [csvResult, setCsvResult] = useState<PredictResponse | null>(null)

  const [desiertas, setDesiertas] = useState<Desierta[]>([])
  const [desiertasTotal, setDesiertasTotal] = useState(0)
  const [desiertasPage, setDesiertasPage] = useState(0)
  const [desiertasLoading, setDesiertasLoading] = useState(false)
  const [selectedDesierta, setSelectedDesierta] = useState<Desierta | null>(null)

  useEffect(() => {
    fetch(`${API}/options`)
      .then(r => r.json())
      .then(setOptions)
      .catch(() => setError('No se pudo conectar al servidor. Asegurate de ejecutar la API.'))
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    await runPrediction()
  }

  const runPrediction = async (overrideForm?: FormData) => {
    const f = overrideForm || form
    setLoading(true)
    setError('')
    setResult(null)

    try {
      const res = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          monto_estimado: parseFloat(f.monto_estimado) || 0,
          valor_total_ofertado: parseFloat(f.valor_total_ofertado) || 0,
          numero_oferentes: parseInt(f.numero_oferentes) || 0,
          cantidad_reclamos: parseInt(f.cantidad_reclamos) || 0,
          region: f.region,
          sector: f.sector,
          tipo_adquisicion: f.tipo_adquisicion,
        }),
      })
      const data: PredictResponse = await res.json()
      setResult(data)
    } catch {
      setError('Error al obtener predicciones')
    } finally {
      setLoading(false)
    }
  }

  const updateField = (field: keyof FormData) => (
    e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>
  ) => setForm(prev => ({ ...prev, [field]: e.target.value }))

  const canSubmit =
    form.monto_estimado && form.valor_total_ofertado && form.region && form.sector && form.tipo_adquisicion

  const loadCsvPage = async (page: number) => {
    setCsvLoading(true)
    try {
      const res = await fetch(`${API}/csv-data?page=${page}&size=15`)
      const data = await res.json()
      setCsvRows(data.data)
      setCsvTotal(data.total)
      setCsvPage(page)
    } catch {
      setError('Error al cargar datos CSV')
    } finally {
      setCsvLoading(false)
    }
  }

  const loadRandomCsv = async () => {
    setCsvLoading(true)
    try {
      const res = await fetch(`${API}/csv-random`)
      const row: CsvRow = await res.json()
      await selectCsvRow(row)
    } catch {
      setError('Error al cargar licitacion aleatoria')
    } finally {
      setCsvLoading(false)
    }
  }

  const selectCsvRow = async (row: CsvRow) => {
    setSelectedCsv(row)
    setCsvResult(null)
    try {
      const res = await fetch(`${API}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          monto_estimado: row.monto_estimado,
          valor_total_ofertado: row.valor_total_ofertado,
          numero_oferentes: row.numero_oferentes,
          cantidad_reclamos: row.cantidad_reclamos,
          region: row.region,
          sector: row.sector,
          tipo_adquisicion: row.tipo_adquisicion,
        }),
      })
      const data: PredictResponse = await res.json()
      setCsvResult(data)
    } catch {
      setError('Error al predecir licitacion CSV')
    }
  }

  const fillFormFromCsv = (row: CsvRow) => {
    setForm({
      monto_estimado: String(row.monto_estimado),
      valor_total_ofertado: String(row.valor_total_ofertado),
      numero_oferentes: String(row.numero_oferentes),
      cantidad_reclamos: String(row.cantidad_reclamos),
      region: row.region,
      sector: row.sector,
      tipo_adquisicion: row.tipo_adquisicion,
    })
    setActiveTab('predictor')
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const totalCsvPages = Math.ceil(csvTotal / 15)

  const loadDesiertasPage = async (page: number) => {
    setDesiertasLoading(true)
    try {
      const res = await fetch(`${API}/desiertas?page=${page}&size=15`)
      const data = await res.json()
      setDesiertas(data.data)
      setDesiertasTotal(data.total)
      setDesiertasPage(page)
    } catch {
      setError('Error al cargar oportunidades')
    } finally {
      setDesiertasLoading(false)
    }
  }

  useEffect(() => {
    if (activeTab === 'oportunidades' && desiertasPage === 0) {
      loadDesiertasPage(1)
    }
  }, [activeTab])

  const totalDesiertasPages = Math.ceil(desiertasTotal / 15)

  return (
    <div className="app">
      <header className="header">
        <h1>Predictor de Licitaciones</h1>
        <p>Un modelo Gradient Boosting, tres umbrales de riesgo, tres perfiles de asesor</p>
      </header>

      <nav className="tabs">
        <button
          className={`tab ${activeTab === 'predictor' ? 'active' : ''}`}
          onClick={() => setActiveTab('predictor')}
        >
          Predictor
        </button>
        <button
          className={`tab ${activeTab === 'oportunidades' ? 'active' : ''}`}
          onClick={() => setActiveTab('oportunidades')}
        >
          Oportunidades
        </button>
      </nav>

      {error && <div className="error-msg">{error}</div>}

      {activeTab === 'predictor' && (
        <>
          <section className="form-section">
            <h2>Datos de la Licitacion</h2>
            <form onSubmit={handleSubmit}>
              <div className="form-grid">
                <div className="form-group">
                  <label>Monto Estimado ($)</label>
                  <input
                    type="number"
                    placeholder="ej: 15000000"
                    value={form.monto_estimado}
                    onChange={updateField('monto_estimado')}
                    min="0"
                  />
                </div>
                <div className="form-group">
                  <label>Valor Total Ofertado ($)</label>
                  <input
                    type="number"
                    placeholder="ej: 14500000"
                    value={form.valor_total_ofertado}
                    onChange={updateField('valor_total_ofertado')}
                    min="0"
                  />
                </div>
                <div className="form-group">
                  <label>Numero de Oferentes</label>
                  <input
                    type="number"
                    placeholder="ej: 5"
                    value={form.numero_oferentes}
                    onChange={updateField('numero_oferentes')}
                    min="0"
                  />
                </div>
                <div className="form-group">
                  <label>Cantidad de Reclamos</label>
                  <input
                    type="number"
                    placeholder="0"
                    value={form.cantidad_reclamos}
                    onChange={updateField('cantidad_reclamos')}
                    min="0"
                  />
                </div>
                <div className="form-group">
                  <label>Region</label>
                  <select value={form.region} onChange={updateField('region')}>
                    <option value="">Seleccionar...</option>
                    {options?.regiones.map(r => (
                      <option key={r} value={r}>{r}</option>
                    ))}
                  </select>
                </div>
                <div className="form-group">
                  <label>Sector</label>
                  <select value={form.sector} onChange={updateField('sector')}>
                    <option value="">Seleccionar...</option>
                    {options?.sectores.map(s => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                </div>
                <div className="form-group full-width">
                  <label>Tipo de Adquisicion</label>
                  <select value={form.tipo_adquisicion} onChange={updateField('tipo_adquisicion')}>
                    <option value="">Seleccionar...</option>
                    {options?.tipos_adquisicion.map(t => (
                      <option key={t} value={t}>{t}</option>
                    ))}
                  </select>
                </div>
              </div>
              <div className="submit-row">
                <button type="submit" className="btn-predict" disabled={!canSubmit || loading}>
                  {loading ? 'Analizando...' : 'Consultar Asesores'}
                </button>
              </div>
            </form>
          </section>

          {loading && (
            <div className="loading-overlay">
              <div className="spinner" />
              <p>Los asesores estan analizando la licitacion...</p>
            </div>
          )}

          {result && (
            <>
              <section className="probability-banner">
                <span className="prob-label">Probabilidad del modelo</span>
                <span className={`prob-value ${result.probability >= 50 ? 'favorable' : 'unfavorable'}`}>
                  {result.probability}%
                </span>
                <div className="prob-bar-track">
                  <div
                    className={`prob-bar-fill ${result.probability >= 50 ? 'favorable' : 'unfavorable'}`}
                    style={{ width: `${Math.min(result.probability, 100)}%` }}
                  />
                  <div className="prob-bar-marks">
                    <span className="mark" style={{ left: '50%' }}>0.5</span>
                    <span className="mark" style={{ left: '70%' }}>0.7</span>
                    <span className="mark" style={{ left: '90%' }}>0.9</span>
                  </div>
                </div>
              </section>

              <section className="profiles-section">
                <h2>Perfiles de Riesgo</h2>
                <div className="profiles-grid">
                  {result.predictions.map(p => {
                    const cls = profileClass[p.key]
                    return (
                      <div key={p.key} className={`profile-card ${cls}`}>
                        <div className="profile-header">
                          <div className={`profile-avatar ${cls}`}>{p.avatar}</div>
                          <div className="profile-info">
                            <h3>{p.name}</h3>
                            <span className="profile-style">{p.style}</span>
                          </div>
                        </div>
                        <p className="profile-description">{p.description}</p>
                        <div className="profile-result">
                          <span className={`profile-badge ${p.selected ? 'selected' : 'not-selected'}`}>
                            {p.selected ? 'SELECCIONADA' : 'NO SELECCIONADA'}
                          </span>
                          <div className="profile-verdict">{p.verdict}</div>
                        </div>
                        <div className="profile-metrics">
                          <div className="metric">
                            <span className="metric-label">Precision</span>
                            <span className="metric-value">{p.model_precision}%</span>
                          </div>
                          <div className="metric">
                            <span className="metric-label">Recall</span>
                            <span className="metric-value">{p.model_recall}%</span>
                          </div>
                          <div className="metric">
                            <span className="metric-label">F1</span>
                            <span className="metric-value">{p.model_f1}%</span>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>
              </section>

              <section className="recommendations-section">
                <h2>Recomendaciones</h2>
                <div className="recs-grid">
                  <div className="recs-tips">
                    <h3>Analisis</h3>
                    <ul>
                      {result.recommendations.tips.map((tip, i) => (
                        <li key={i}>{tip}</li>
                      ))}
                    </ul>
                  </div>
                  <div className="recs-comparisons">
                    <h3>Comparacion con Medianas</h3>
                    {result.recommendations.comparisons.map(c => (
                      <div key={c.field} className="comparison-row">
                        <span className="comp-field">{c.field}</span>
                        <div className="comp-bar-track">
                          <div
                            className={`comp-bar-fill ${c.status}`}
                            style={{ width: `${Math.min(c.ratio * 50, 100)}%` }}
                          />
                        </div>
                        <span className={`comp-ratio ${c.status}`}>{c.ratio}x</span>
                        <span className="comp-detail">
                          {fmt(c.value)} vs {fmt(c.median)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </section>
            </>
          )}

          <section className="csv-section">
            <h2>Prueba con Datos Reales</h2>
            <p className="csv-subtitle">Licitaciones reales de mayo-junio 2026 del portal Mercado Publico</p>
            <div className="csv-actions">
              <button className="btn-random" onClick={loadRandomCsv} disabled={csvLoading}>
                Licitacion Aleatoria
              </button>
              {csvRows.length === 0 && (
                <button className="btn-browse" onClick={() => loadCsvPage(1)} disabled={csvLoading}>
                  Explorar Tabla
                </button>
              )}
            </div>

            {selectedCsv && (
              <div className="csv-selected">
                <div className="csv-selected-header">
                  <h3>Licitacion Seleccionada</h3>
                  <button className="btn-fill" onClick={() => fillFormFromCsv(selectedCsv)}>
                    Usar en Formulario
                  </button>
                </div>
                <div className="csv-detail-grid">
                  <div><strong>Nombre:</strong> {selectedCsv.nombre}</div>
                  <div><strong>Proveedor:</strong> {selectedCsv.proveedor}</div>
                  <div><strong>Monto Est.:</strong> ${fmt(selectedCsv.monto_estimado)}</div>
                  <div><strong>Valor Ofertado:</strong> ${fmt(selectedCsv.valor_total_ofertado)}</div>
                  <div><strong>Oferentes:</strong> {selectedCsv.numero_oferentes}</div>
                  <div><strong>Reclamos:</strong> {selectedCsv.cantidad_reclamos}</div>
                  <div><strong>Region:</strong> {selectedCsv.region}</div>
                  <div><strong>Sector:</strong> {selectedCsv.sector}</div>
                  <div><strong>Tipo:</strong> {selectedCsv.tipo_adquisicion}</div>
                  <div>
                    <strong>Resultado Real: </strong>
                    <span className={`real-badge ${selectedCsv.resultado_real === 'Seleccionada' ? 'real-sel' : 'real-no'}`}>
                      {selectedCsv.resultado_real || 'Sin dato'}
                    </span>
                  </div>
                </div>

                {csvResult && (
                  <div className="csv-predictions">
                    <h4>Predicciones vs Realidad</h4>
                    <div className="csv-pred-row">
                      <div className="csv-prob">
                        Probabilidad: <strong>{csvResult.probability}%</strong>
                      </div>
                      {csvResult.predictions.map(p => (
                        <div key={p.key} className={`csv-pred-chip ${profileClass[p.key]} ${p.selected ? 'sel' : 'nosel'}`}>
                          <span className="chip-name">{p.name}</span>
                          <span className="chip-verdict">{p.selected ? 'SI' : 'NO'}</span>
                        </div>
                      ))}
                      <div className={`csv-real-chip ${selectedCsv.resultado_real === 'Seleccionada' ? 'real-sel' : 'real-no'}`}>
                        <span className="chip-name">Real</span>
                        <span className="chip-verdict">{selectedCsv.resultado_real === 'Seleccionada' ? 'SI' : 'NO'}</span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}

            {csvRows.length > 0 && (
              <div className="csv-table-wrap">
                <table className="csv-table">
                  <thead>
                    <tr>
                      <th>Nombre</th>
                      <th>Monto Est.</th>
                      <th>Valor Ofert.</th>
                      <th>Oferentes</th>
                      <th>Sector</th>
                      <th>Resultado</th>
                      <th></th>
                    </tr>
                  </thead>
                  <tbody>
                    {csvRows.map((row, i) => (
                      <tr key={i} className={selectedCsv === row ? 'active-row' : ''}>
                        <td className="td-name" title={row.nombre}>{row.nombre.slice(0, 40)}</td>
                        <td>${fmt(row.monto_estimado)}</td>
                        <td>${fmt(row.valor_total_ofertado)}</td>
                        <td>{row.numero_oferentes}</td>
                        <td>{row.sector}</td>
                        <td>
                          <span className={`real-badge-sm ${row.resultado_real === 'Seleccionada' ? 'real-sel' : 'real-no'}`}>
                            {row.resultado_real === 'Seleccionada' ? 'Sel' : 'No'}
                          </span>
                        </td>
                        <td>
                          <button className="btn-select-row" onClick={() => selectCsvRow(row)}>
                            Analizar
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="csv-pagination">
                  <button disabled={csvPage <= 1} onClick={() => loadCsvPage(csvPage - 1)}>Anterior</button>
                  <span>Pagina {csvPage} de {totalCsvPages.toLocaleString()} ({csvTotal.toLocaleString()} filas)</span>
                  <button disabled={csvPage >= totalCsvPages} onClick={() => loadCsvPage(csvPage + 1)}>Siguiente</button>
                </div>
              </div>
            )}
          </section>
        </>
      )}

      {activeTab === 'oportunidades' && (
        <section className="oportunidades-section">
          <div className="oport-intro">
            <h2>Licitaciones sin Adjudicar</h2>
            <p>
              Licitaciones reales donde ninguna oferta fue seleccionada.
              Para cada una, el modelo calcula el valor de oferta que maximiza la probabilidad de adjudicacion.
            </p>
          </div>

          {desiertasLoading && (
            <div className="loading-overlay">
              <div className="spinner" />
              <p>Calculando valores optimos...</p>
            </div>
          )}

          {selectedDesierta && (
            <div className="desierta-detail">
              <div className="desierta-detail-header">
                <h3>{selectedDesierta.nombre}</h3>
                <button className="btn-close" onClick={() => setSelectedDesierta(null)}>Cerrar</button>
              </div>
              <div className="desierta-detail-grid">
                <div className="desierta-info-col">
                  <div className="detail-item"><strong>Codigo:</strong> {selectedDesierta.codigo}</div>
                  <div className="detail-item"><strong>Region:</strong> {selectedDesierta.region}</div>
                  <div className="detail-item"><strong>Sector:</strong> {selectedDesierta.sector}</div>
                  <div className="detail-item"><strong>Tipo:</strong> {selectedDesierta.tipo_adquisicion}</div>
                  <div className="detail-item"><strong>Oferentes:</strong> {selectedDesierta.numero_oferentes}</div>
                  <div className="detail-item"><strong>Reclamos:</strong> {selectedDesierta.cantidad_reclamos}</div>
                  <div className="detail-item"><strong>Ofertas rechazadas:</strong> {selectedDesierta.n_ofertas}</div>
                </div>
                <div className="desierta-sugerencia-col">
                  <div className="sug-card">
                    <div className="sug-label">Monto Estimado</div>
                    <div className="sug-monto">{fmtPesos(selectedDesierta.monto_estimado)}</div>
                  </div>
                  <div className="sug-card highlight">
                    <div className="sug-label">Valor de Oferta Sugerido</div>
                    <div className="sug-monto">{fmtPesos(selectedDesierta.valor_sugerido)}</div>
                    <div className="sug-ratio">
                      {(selectedDesierta.valor_sugerido / selectedDesierta.monto_estimado * 100).toFixed(0)}% del monto estimado
                    </div>
                  </div>
                  <div className="sug-card">
                    <div className="sug-label">Probabilidad estimada</div>
                    <div className={`sug-prob ${selectedDesierta.prob_sugerida >= 50 ? 'favorable' : 'unfavorable'}`}>
                      {selectedDesierta.prob_sugerida}%
                    </div>
                  </div>
                  <div className="sug-profiles">
                    <span className={`sug-chip ${selectedDesierta.aprueba_riesgoso ? 'approved' : 'rejected'}`}>
                      Arriesgado: {selectedDesierta.aprueba_riesgoso ? 'SI' : 'NO'}
                    </span>
                    <span className={`sug-chip ${selectedDesierta.aprueba_moderado ? 'approved' : 'rejected'}`}>
                      Equilibrado: {selectedDesierta.aprueba_moderado ? 'SI' : 'NO'}
                    </span>
                    <span className={`sug-chip ${selectedDesierta.aprueba_conservador ? 'approved' : 'rejected'}`}>
                      Cauteloso: {selectedDesierta.aprueba_conservador ? 'SI' : 'NO'}
                    </span>
                  </div>
                </div>
              </div>
              <div className="desierta-actions">
                <button className="btn-fill" onClick={() => {
                  setForm({
                    monto_estimado: String(selectedDesierta.monto_estimado),
                    valor_total_ofertado: String(selectedDesierta.valor_sugerido),
                    numero_oferentes: String(selectedDesierta.numero_oferentes),
                    cantidad_reclamos: String(selectedDesierta.cantidad_reclamos),
                    region: selectedDesierta.region,
                    sector: selectedDesierta.sector,
                    tipo_adquisicion: selectedDesierta.tipo_adquisicion,
                  })
                  setActiveTab('predictor')
                  window.scrollTo({ top: 0, behavior: 'smooth' })
                }}>
                  Simular con valor sugerido en Predictor
                </button>
              </div>
            </div>
          )}

          {desiertas.length > 0 && (
            <div className="csv-table-wrap">
              <table className="csv-table desiertas-table">
                <thead>
                  <tr>
                    <th>Nombre</th>
                    <th>Monto Est.</th>
                    <th>Valor Sugerido</th>
                    <th>Prob.</th>
                    <th>Perfiles</th>
                    <th>Region</th>
                    <th></th>
                  </tr>
                </thead>
                <tbody>
                  {desiertas.map(d => (
                    <tr key={d.codigo} className={selectedDesierta?.codigo === d.codigo ? 'active-row' : ''}>
                      <td className="td-name" title={d.nombre}>{d.nombre.slice(0, 35)}</td>
                      <td>{fmtPesos(d.monto_estimado)}</td>
                      <td className="td-sugerido">{fmtPesos(d.valor_sugerido)}</td>
                      <td>
                        <span className={`prob-chip ${d.prob_sugerida >= 70 ? 'high' : d.prob_sugerida >= 50 ? 'med' : 'low'}`}>
                          {d.prob_sugerida}%
                        </span>
                      </td>
                      <td className="td-profiles">
                        <span className={`dot ${d.aprueba_riesgoso ? 'on' : 'off'}`} title="Arriesgado">R</span>
                        <span className={`dot ${d.aprueba_moderado ? 'on' : 'off'}`} title="Equilibrado">M</span>
                        <span className={`dot ${d.aprueba_conservador ? 'on' : 'off'}`} title="Cauteloso">C</span>
                      </td>
                      <td className="td-region">{d.region.replace('Región ', '').replace('de ', '').slice(0, 20)}</td>
                      <td>
                        <button className="btn-select-row" onClick={() => setSelectedDesierta(d)}>
                          Ver detalle
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="csv-pagination">
                <button disabled={desiertasPage <= 1} onClick={() => loadDesiertasPage(desiertasPage - 1)}>Anterior</button>
                <span>Pagina {desiertasPage} de {totalDesiertasPages.toLocaleString()} ({desiertasTotal.toLocaleString()} licitaciones)</span>
                <button disabled={desiertasPage >= totalDesiertasPages} onClick={() => loadDesiertasPage(desiertasPage + 1)}>Siguiente</button>
              </div>
            </div>
          )}
        </section>
      )}
    </div>
  )
}

export default App
