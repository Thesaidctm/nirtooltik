import { useEffect, useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { Line, LineChart, CartesianGrid, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'
import type { Dataset } from '../../types/datasets'
import type { PlsLvCurvePoint, Run, RunMetricsResponse, RunPlotsResponse } from '../../types/runs'

interface RunDetailsPageProps {
  runId: string
  apiBaseUrl: string
}

const formatDate = (value?: string) => {
  if (!value) return '-'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleString()
}

const sectionCard = (title: string, children: React.ReactNode) => (
  <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
    <div className="mb-3 flex items-center justify-between">
      <h3 className="text-lg font-semibold text-slate-900">{title}</h3>
    </div>
    {children}
  </div>
)

const RunDetailsPage = ({ runId, apiBaseUrl }: RunDetailsPageProps) => {
  const [run, setRun] = useState<Run | null>(null)
  const [metrics, setMetrics] = useState<RunMetricsResponse | null>(null)
  const [plots, setPlots] = useState<RunPlotsResponse | null>(null)
  const [plsCurve, setPlsCurve] = useState<PlsLvCurvePoint[] | null>(null)
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const datasetName = useMemo(() => {
    if (!run) return '-'
    const match = datasets.find((dataset) => `${dataset.id}` === `${run.dataset_id}`)
    return match?.name ?? run.dataset_id
  }, [datasets, run])

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true)
      setError(null)
      try {
        const [runRes, metricsRes, plotsRes, datasetsRes] = await Promise.all([
          fetch(`${apiBaseUrl}/runs/${runId}`),
          fetch(`${apiBaseUrl}/runs/${runId}/metrics`),
          fetch(`${apiBaseUrl}/runs/${runId}/plots`),
          fetch(`${apiBaseUrl}/datasets`),
        ])

        if (!runRes.ok) throw new Error('Não foi possível carregar o run')
        const runData: Run = await runRes.json()
        setRun(runData)

        if (metricsRes.ok) {
          const metricsData: RunMetricsResponse = await metricsRes.json()
          setMetrics(metricsData)
        }

        if (plotsRes.ok) {
          const plotsData: RunPlotsResponse = await plotsRes.json()
          setPlots(plotsData)
        }

        if (datasetsRes.ok) {
          const datasetData: Dataset[] = await datasetsRes.json()
          setDatasets(datasetData)
        }

        // Try to load optional PLS curve
        try {
          const plsRes = await fetch(`${apiBaseUrl}/runs/${runId}/pls-lv-curve`)
          if (plsRes.ok) {
            const plsData: PlsLvCurvePoint[] = await plsRes.json()
            setPlsCurve(plsData)
          }
        } catch (plsErr) {
          console.warn('PLS curve not available', plsErr)
        }
      } catch (err) {
        console.error(err)
        setError(err instanceof Error ? err.message : 'Erro ao carregar detalhes do run')
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [apiBaseUrl, runId])

  if (loading) {
    return <div className="py-12 text-center text-slate-500">Carregando detalhes...</div>
  }

  if (error || !run) {
    return (
      <div className="py-12 text-center text-slate-500">
        {error ?? 'Não foi possível encontrar esta calibração.'}
        <div className="mt-4">
          <Link to="/runs" className="text-indigo-600 hover:underline">
            Voltar para lista
          </Link>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">{run.name}</h1>
          <p className="text-slate-600">Visualize o progresso e resultados da calibração.</p>
        </div>
        <Link
          to="/runs"
          className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
        >
          ← Voltar
        </Link>
      </div>

      {sectionCard(
        'Resumo',
        <dl className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
            <dt className="text-xs font-semibold uppercase tracking-wide text-slate-500">Dataset</dt>
            <dd className="text-sm font-medium text-slate-900">{datasetName}</dd>
          </div>
          <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
            <dt className="text-xs font-semibold uppercase tracking-wide text-slate-500">Status</dt>
            <dd className="text-sm font-medium text-slate-900">{run.status}</dd>
          </div>
          <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2">
            <dt className="text-xs font-semibold uppercase tracking-wide text-slate-500">Criado em</dt>
            <dd className="text-sm font-medium text-slate-900">{formatDate(run.created_at)}</dd>
          </div>
        </dl>,
      )}

      {sectionCard(
        'Métricas',
        metrics ? (
          <pre className="max-h-96 overflow-auto rounded-lg border border-slate-100 bg-slate-50 p-3 text-sm text-slate-800">
            {JSON.stringify(metrics, null, 2)}
          </pre>
        ) : (
          <p className="text-sm text-slate-600">Nenhuma métrica encontrada para este run.</p>
        ),
      )}

      {sectionCard(
        'Plots',
        plots?.plots?.length ? (
          <ul className="list-disc space-y-1 pl-5 text-sm text-slate-800">
            {plots.plots.map((plot) => (
              <li key={plot}>{plot}</li>
            ))}
          </ul>
        ) : (
          <p className="text-sm text-slate-600">Nenhum plot disponível.</p>
        ),
      )}

      {sectionCard(
        'PLS',
        plsCurve?.length ? (
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={plsCurve} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="n_components" label={{ value: 'LV', position: 'insideBottom', dy: 10 }} />
                <YAxis label={{ value: 'RMSECV', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Line type="monotone" dataKey="rmsecv" stroke="#4f46e5" strokeWidth={2} dot />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <p className="text-sm text-slate-600">Curva LV × RMSECV não disponível.</p>
        ),
      )}
    </div>
  )
}

export default RunDetailsPage
