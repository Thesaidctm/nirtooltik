import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import type { ModelSummary } from '../types/models'

const API_BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const getPrimaryMetric = (metrics?: Record<string, number>) => {
  if (!metrics) return null
  const preferredOrder = ['r2', 'rmsep', 'rmse', 'mae', 'mae_percent']
  const key = preferredOrder.find((metric) => metrics[metric] !== undefined)
  if (key) {
    return { key, value: metrics[key] }
  }
  const [firstKey] = Object.keys(metrics)
  return firstKey ? { key: firstKey, value: metrics[firstKey] } : null
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
      const response = await fetch(`${API_BASE_URL}/models`)
      if (!response.ok) {
        throw new Error('Não foi possível carregar os modelos')
      }
      const data: ModelSummary[] = await response.json()
      setModels(data)
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

  const hasModels = useMemo(() => models.length > 0, [models])

  return (
    <div className="space-y-6">
      <div className="space-y-2">
        <h1 className="text-3xl font-bold text-slate-900">Modelos Registrados</h1>
        <p className="text-slate-600">Visualize e acompanhe modelos treinados a partir das calibrações.</p>
      </div>

      <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">Lista de modelos</h2>
            <p className="text-sm text-slate-600">Detalhes rápidos sobre cada modelo disponível.</p>
          </div>
          <button
            type="button"
            onClick={fetchModels}
            className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
          >
            Atualizar
          </button>
        </div>

        {error ? (
          <div className="mb-4 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">{error}</div>
        ) : null}

        <div className="overflow-hidden rounded-xl border border-slate-200">
          <table className="min-w-full divide-y divide-slate-200">
            <thead className="bg-slate-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-600">Nome</th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-600">Target</th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-600">Dataset</th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-600">Instrumento</th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-600">Métrica</th>
                <th className="px-4 py-3"></th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 bg-white">
              {loading ? (
                <tr>
                  <td colSpan={6} className="px-4 py-6 text-center text-sm text-slate-500">
                    Carregando modelos...
                  </td>
                </tr>
              ) : !hasModels ? (
                <tr>
                  <td colSpan={6} className="px-4 py-6 text-center text-sm text-slate-500">
                    Nenhum modelo encontrado.
                  </td>
                </tr>
              ) : (
                models.map((model) => {
                  const primaryMetric = getPrimaryMetric(model.metrics)
                  return (
                    <tr key={model.id} className="hover:bg-slate-50/80">
                      <td className="px-4 py-3 text-sm font-medium text-slate-900">{model.name}</td>
                      <td className="px-4 py-3 text-sm text-slate-700">{model.target_name ?? model.target ?? '-'}</td>
                      <td className="px-4 py-3 text-sm text-slate-700">{model.dataset_id ?? '-'}</td>
                      <td className="px-4 py-3 text-sm text-slate-700">{model.instrument_id ?? '-'}</td>
                      <td className="px-4 py-3 text-sm text-slate-700">
                        {primaryMetric ? (
                          <span className="inline-flex items-center gap-2 rounded-full bg-indigo-50 px-3 py-1 text-xs font-semibold text-indigo-700 ring-1 ring-inset ring-indigo-100">
                            <span className="uppercase text-[10px] tracking-wide text-indigo-600">{primaryMetric.key}</span>
                            <span>{primaryMetric.value}</span>
                          </span>
                        ) : (
                          <span className="text-slate-400">—</span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <button
                          type="button"
                          onClick={() => navigate(`/models/${model.id}`)}
                          className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-3 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700"
                        >
                          Ver detalhes
                        </button>
                      </td>
                    </tr>
                  )
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export default Models
