import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { api } from '../api/client'
import type { ModelMetrics, ModelSummary } from '../types/models'

const resolvePrimaryMetric = (metrics?: ModelMetrics) => {
  if (!metrics) return null
  const priority = ['rmsep', 'rmse', 'r2', 'rpd', 'mae_percent']
  const key = priority.find((metric) => metrics[metric] !== undefined) ?? Object.keys(metrics)[0]
  return key ? { key, value: metrics[key] } : null
}

const Models = () => {
  const [models, setModels] = useState<ModelSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const navigate = useNavigate()

  const fetchModels = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await api.get<ModelSummary[]>('/models')
      setModels(response)
    } catch (err) {
      console.error(err)
      setError(err instanceof Error ? err.message : 'Erro ao carregar modelos')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchModels()
  }, [])

  const subtitle = useMemo(() => {
    if (loading) return 'Carregando modelos...'
    if (error) return 'Não foi possível carregar os modelos registrados.'
    return 'Modelos calibrados disponíveis para predição.'
  }, [error, loading])

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-sm font-semibold uppercase tracking-wide text-indigo-600">Modelos</p>
          <h1 className="text-3xl font-bold text-slate-900">Models</h1>
          <p className="text-slate-600">{subtitle}</p>
        </div>
        <button
          type="button"
          onClick={fetchModels}
          className="inline-flex items-center gap-2 rounded-lg border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
        >
          Atualizar
        </button>
      </div>

      {error ? (
        <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div>
      ) : null}

      {loading ? (
        <div className="flex items-center justify-center rounded-xl border border-dashed border-slate-200 bg-white py-12 text-slate-500">
          Carregando...
        </div>
      ) : models.length === 0 ? (
        <div className="flex items-center justify-center rounded-xl border border-dashed border-slate-200 bg-white py-12 text-slate-500">
          Nenhum modelo cadastrado ainda.
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {models.map((model) => {
            const primaryMetric = resolvePrimaryMetric(model.metrics)
            return (
              <button
                key={model.id}
                type="button"
                onClick={() => navigate(`/models/${model.id}`)}
                className="flex flex-col rounded-2xl border border-slate-200 bg-white p-4 text-left shadow-sm transition hover:-translate-y-0.5 hover:shadow-md focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-xs font-semibold uppercase tracking-wide text-indigo-600">Modelo</p>
                    <h2 className="text-lg font-semibold text-slate-900">{model.name}</h2>
                  </div>
                  {primaryMetric ? (
                    <div className="rounded-lg bg-indigo-50 px-3 py-1 text-xs font-semibold text-indigo-700">
                      {primaryMetric.key}: {primaryMetric.value}
                    </div>
                  ) : null}
                </div>

                <dl className="mt-3 grid grid-cols-2 gap-3 text-sm text-slate-700">
                  <div>
                    <dt className="text-xs font-semibold uppercase tracking-wide text-slate-500">Alvo</dt>
                    <dd className="font-medium text-slate-900">{model.target_name ?? model.target ?? '—'}</dd>
                  </div>
                  <div>
                    <dt className="text-xs font-semibold uppercase tracking-wide text-slate-500">Dataset</dt>
                    <dd className="font-medium text-slate-900">{model.dataset_id ?? '—'}</dd>
                  </div>
                  <div>
                    <dt className="text-xs font-semibold uppercase tracking-wide text-slate-500">Instrumento</dt>
                    <dd className="font-medium text-slate-900">{model.instrument_id ?? '—'}</dd>
                  </div>
                  <div>
                    <dt className="text-xs font-semibold uppercase tracking-wide text-slate-500">Status</dt>
                    <dd className="font-medium text-slate-900">{model.status ?? '—'}</dd>
                  </div>
                </dl>
              </button>
            )
          })}
        </div>
      )}
    </div>
  )
}

export default Models
