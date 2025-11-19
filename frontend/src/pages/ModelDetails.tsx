import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import type { Dataset } from '../types/datasets'
import type { ModelDetail, ModelEvaluationRecord, ModelMetrics, ModelSpec } from '../types/models'

const API_BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const resolveModelSpec = (model: ModelDetail | null): ModelSpec | undefined => {
  if (!model) return undefined
  const artifactsSpec = model.artifacts?.model_spec as ModelSpec | undefined
  const nestedBestModel = (model.artifacts?.best_model as { model_spec?: ModelSpec } | undefined)?.model_spec
  return model.model_spec ?? artifactsSpec ?? nestedBestModel
}

const formatPreprocess = (preprocess?: ModelSpec['preprocess']) => {
  if (!preprocess) return '—'
  if (Array.isArray(preprocess)) {
    return preprocess.join(', ')
  }
  if (typeof preprocess === 'string') {
    return preprocess
  }
  const steps = Object.keys(preprocess)
  return steps.length > 0 ? steps.join(', ') : '—'
}

const getPrimaryMetricLabel = (metrics?: ModelMetrics) => {
  if (!metrics) return null
  const priority = ['rmsep', 'rmse', 'r2', 'rpd', 'mae_percent']
  const key = priority.find((metric) => metrics[metric] !== undefined) ?? Object.keys(metrics)[0]
  return key ? { key, value: metrics[key] } : null
}

const formatDate = (value?: string) => {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return date.toLocaleString('pt-BR')
}

const EvaluationHistoryTable = ({ evalHistory }: { evalHistory?: ModelEvaluationRecord[] }) => {
  if (!evalHistory?.length) {
    return <p className="text-sm text-slate-500">Nenhuma avaliação externa registrada.</p>
  }

  return (
    <div className="overflow-hidden rounded-xl border border-slate-200">
      <table className="min-w-full divide-y divide-slate-200">
        <thead className="bg-slate-50">
          <tr>
            <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-600">Dataset</th>
            <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-600">Métricas</th>
            <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-600">Data</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-200 bg-white">
          {evalHistory.map((record, idx) => (
            <tr key={`${record.dataset_id}-${idx}`} className="hover:bg-slate-50/70">
              <td className="px-4 py-3 text-sm font-medium text-slate-900">{record.dataset_id ?? '—'}</td>
              <td className="px-4 py-3 text-sm text-slate-700">
                {record.metrics ? (
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(record.metrics).map(([key, value]) => (
                      <span
                        key={key}
                        className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-3 py-1 text-xs font-semibold text-slate-700 ring-1 ring-inset ring-slate-200"
                      >
                        <span className="uppercase text-[10px] tracking-wide text-slate-500">{key}</span>
                        <span>{value}</span>
                      </span>
                    ))}
                  </div>
                ) : (
                  <span className="text-slate-400">—</span>
                )}
              </td>
              <td className="px-4 py-3 text-sm text-slate-700">{formatDate(record.evaluated_at ?? record.created_at)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

const ModelDetails = () => {
  const { modelId } = useParams<{ modelId: string }>()
  const navigate = useNavigate()
  const [model, setModel] = useState<ModelDetail | null>(null)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [evaluationDatasetId, setEvaluationDatasetId] = useState<string>('')
  const [evaluationResult, setEvaluationResult] = useState<ModelMetrics | null>(null)
  const [evaluating, setEvaluating] = useState(false)
  const [evaluationError, setEvaluationError] = useState<string | null>(null)

  const modelSpec = useMemo(() => resolveModelSpec(model), [model])

  const fetchModel = useCallback(async () => {
    if (!modelId) return
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_BASE_URL}/models/${modelId}`)
      if (!response.ok) {
        throw new Error('Não foi possível carregar os detalhes do modelo')
      }
      const data: ModelDetail = await response.json()
      setModel(data)
    } catch (err) {
      console.error(err)
      setError(err instanceof Error ? err.message : 'Erro ao carregar modelo')
    } finally {
      setLoading(false)
    }
  }, [modelId])

  const fetchDatasets = useCallback(async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/datasets`)
      if (!response.ok) {
        throw new Error('Não foi possível carregar os datasets')
      }
      const data: Dataset[] = await response.json()
      setDatasets(data)
    } catch (err) {
      console.error(err)
    }
  }, [])

  const evaluateModel = async () => {
    if (!modelId || !evaluationDatasetId) return
    setEvaluating(true)
    setEvaluationError(null)
    setEvaluationResult(null)
    try {
      const response = await fetch(`${API_BASE_URL}/models/${modelId}/evaluate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dataset_id: Number(evaluationDatasetId), target_index: 0 }),
      })

      if (!response.ok) {
        throw new Error('Falha ao avaliar o modelo no dataset selecionado')
      }

      const data: { metrics: ModelMetrics } = await response.json()
      setEvaluationResult(data.metrics)
    } catch (err) {
      console.error(err)
      setEvaluationError(err instanceof Error ? err.message : 'Erro ao avaliar modelo')
    } finally {
      setEvaluating(false)
    }
  }

  useEffect(() => {
    fetchModel()
    fetchDatasets()
  }, [fetchDatasets, fetchModel])

  if (!modelId) {
    return <div className="py-12 text-center text-slate-500">Nenhum modelo selecionado.</div>
  }

  if (loading) {
    return <div className="py-12 text-center text-slate-500">Carregando detalhes do modelo...</div>
  }

  if (error || !model) {
    return (
      <div className="space-y-4 py-12 text-center">
        <p className="text-slate-600">{error ?? 'Modelo não encontrado.'}</p>
        <button
          type="button"
          onClick={() => navigate('/models')}
          className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700"
        >
          Voltar para modelos
        </button>
      </div>
    )
  }

  const primaryMetric = getPrimaryMetricLabel(model.metrics)

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">{model.name}</h1>
          <p className="text-slate-600">Visualize os detalhes e avalie o modelo em outros datasets.</p>
        </div>
        <button
          type="button"
          onClick={() => navigate('/models')}
          className="inline-flex items-center gap-2 rounded-lg border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-700 shadow-sm transition hover:bg-slate-50"
        >
          ← Voltar
        </button>
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm lg:col-span-2">
          <h2 className="text-lg font-semibold text-slate-900">Informações gerais</h2>
          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <div className="rounded-lg border border-slate-100 bg-slate-50 px-4 py-3">
              <p className="text-sm text-slate-500">Target</p>
              <p className="text-lg font-semibold text-slate-900">{model.target_name ?? model.target ?? '—'}</p>
            </div>
            <div className="rounded-lg border border-slate-100 bg-slate-50 px-4 py-3">
              <p className="text-sm text-slate-500">Dataset de origem</p>
              <p className="text-lg font-semibold text-slate-900">{model.dataset_id ?? '—'}</p>
            </div>
            <div className="rounded-lg border border-slate-100 bg-slate-50 px-4 py-3">
              <p className="text-sm text-slate-500">Instrumento</p>
              <p className="text-lg font-semibold text-slate-900">{model.instrument_id ?? '—'}</p>
            </div>
            <div className="rounded-lg border border-slate-100 bg-slate-50 px-4 py-3">
              <p className="text-sm text-slate-500">Métrica principal</p>
              <p className="text-lg font-semibold text-slate-900">
                {primaryMetric ? (
                  <span className="inline-flex items-center gap-2 rounded-full bg-indigo-50 px-3 py-1 text-xs font-semibold text-indigo-700 ring-1 ring-inset ring-indigo-100">
                    <span className="uppercase text-[10px] tracking-wide text-indigo-600">{primaryMetric.key}</span>
                    <span>{primaryMetric.value}</span>
                  </span>
                ) : (
                  '—'
                )}
              </p>
            </div>
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <h2 className="text-lg font-semibold text-slate-900">Spec do modelo</h2>
          <dl className="mt-4 space-y-3 text-sm text-slate-700">
            <div className="flex items-start justify-between rounded-lg border border-slate-100 bg-slate-50 px-3 py-2">
              <dt className="text-slate-500">Classe</dt>
              <dd className="font-semibold text-slate-900">{modelSpec?.class ?? modelSpec?.class_name ?? modelSpec?.model_class ?? '—'}</dd>
            </div>
            <div className="flex items-start justify-between rounded-lg border border-slate-100 bg-slate-50 px-3 py-2">
              <dt className="text-slate-500">n_components</dt>
              <dd className="font-semibold text-slate-900">{modelSpec?.n_components ?? '—'}</dd>
            </div>
            <div className="flex items-start justify-between rounded-lg border border-slate-100 bg-slate-50 px-3 py-2">
              <dt className="text-slate-500">Preprocess</dt>
              <dd className="text-right font-semibold text-slate-900">{formatPreprocess(modelSpec?.preprocess)}</dd>
            </div>
          </dl>
        </div>
      </div>

      <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="mb-3 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">Histórico de avaliações externas</h2>
            <p className="text-sm text-slate-600">Resultados registrados para outros datasets.</p>
          </div>
        </div>
        <EvaluationHistoryTable evalHistory={model.eval_history} />
      </div>

      <div className="grid gap-4 lg:grid-cols-3">
        <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm lg:col-span-2">
          <h2 className="text-lg font-semibold text-slate-900">Avaliar em outro dataset</h2>
          <p className="text-sm text-slate-600">Selecione um dataset para medir o desempenho externo deste modelo.</p>

          <div className="mt-4 flex flex-col gap-3 sm:flex-row sm:items-end">
            <div className="flex-1">
              <label htmlFor="dataset" className="text-sm font-semibold text-slate-700">
                Dataset
              </label>
              <select
                id="dataset"
                className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-800 shadow-sm focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                value={evaluationDatasetId}
                onChange={(e) => setEvaluationDatasetId(e.target.value)}
              >
                <option value="">Selecione um dataset</option>
                {datasets.map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>
                    {dataset.name} ({dataset.id})
                  </option>
                ))}
              </select>
            </div>
            <button
              type="button"
              disabled={!evaluationDatasetId || evaluating}
              onClick={evaluateModel}
              className="inline-flex items-center justify-center gap-2 rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-emerald-300"
            >
              {evaluating ? 'Avaliando...' : 'Avaliar modelo nesse dataset'}
            </button>
          </div>

          {evaluationError ? (
            <div className="mt-4 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">{evaluationError}</div>
          ) : null}
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
          <h2 className="text-lg font-semibold text-slate-900">Métricas retornadas</h2>
          {evaluationResult ? (
            <div className="mt-4 space-y-3">
              {Object.entries(evaluationResult).map(([key, value]) => (
                <div
                  key={key}
                  className="flex items-center justify-between rounded-lg border border-slate-100 bg-slate-50 px-3 py-2 text-sm font-semibold text-slate-800"
                >
                  <span className="uppercase text-[10px] tracking-wide text-slate-500">{key}</span>
                  <span>{value}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="mt-3 text-sm text-slate-500">As métricas aparecerão aqui após a avaliação.</p>
          )}
        </div>
      </div>
    </div>
  )
}

export default ModelDetails
