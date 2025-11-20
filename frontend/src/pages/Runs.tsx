import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import { api } from '../api/client'
import RunListTable from '../components/runs/RunListTable'
import type { Dataset, DatasetApiResponse } from '../types/datasets'
import { normalizeDataset } from '../types/datasets'
import type { Run } from '../types/runs'

interface RunApiResponse extends Run {
  run_id?: number
  created_at?: string
  run_uid?: string
}

const Runs = () => {
  const [runs, setRuns] = useState<Run[]>([])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const navigate = useNavigate()

  const normalizeRun = (run: RunApiResponse): Run => {
    const id = run.run_id ?? run.id ?? run.run_uid ?? ''
    return {
      id,
      name: run.name ?? run.run_uid ?? `Run ${run.run_id ?? id}`,
      dataset_id: run.dataset_id ?? '—',
      status: run.status ?? 'desconhecido',
      created_at: run.created_at,
      artifacts: run.artifacts,
      target_cols: run.target_cols,
      config: run.config,
    }
  }

  const fetchRuns = async () => {
    setLoading(true)
    setError(null)
    try {
      const [runsResponse, datasetsResponse] = await Promise.all([
        api.get<RunApiResponse[]>('/runs'),
        api.get<DatasetApiResponse[]>('/datasets/'),
      ])

      setRuns(runsResponse.map(normalizeRun))
      setDatasets(datasetsResponse.map(normalizeDataset))
    } catch (err) {
      console.error(err)
      setError(err instanceof Error ? err.message : 'Erro ao carregar as calibrações')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchRuns()
  }, [])

  const pageSubtitle = useMemo(() => {
    if (loading) return 'Carregando execuções...'
    if (error) return 'Não foi possível carregar as execuções.'
    return 'Acompanhe as calibrações enviadas e seu status.'
  }, [error, loading])

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-sm font-semibold uppercase tracking-wide text-indigo-600">Execuções</p>
          <h1 className="text-3xl font-bold text-slate-900">Runs</h1>
          <p className="text-slate-600">{pageSubtitle}</p>
        </div>
        <button
          type="button"
          onClick={fetchRuns}
          className="inline-flex items-center gap-2 rounded-lg border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
        >
          Atualizar
        </button>
      </div>

      {error ? (
        <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div>
      ) : null}

      <RunListTable
        runs={runs}
        datasets={datasets}
        loading={loading}
        onView={(run) => navigate(`/runs/${run.id}`)}
      />
    </div>
  )
}

export default Runs
