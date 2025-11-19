import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import RunCreateModal from '../components/runs/RunCreateModal'
import RunListTable from '../components/runs/RunListTable'
import type { Dataset } from '../types/datasets'
import type { Run } from '../types/runs'

const API_BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const Runs = () => {
  const [runs, setRuns] = useState<Run[]>([])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [loadingRuns, setLoadingRuns] = useState(true)
  const [loadingDatasets, setLoadingDatasets] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isCreateOpen, setIsCreateOpen] = useState(false)
  const navigate = useNavigate()

  const loading = useMemo(() => loadingRuns || loadingDatasets, [loadingRuns, loadingDatasets])

  const fetchRuns = async () => {
    setLoadingRuns(true)
    setError(null)
    try {
      const response = await fetch(`${API_BASE_URL}/runs`)
      if (!response.ok) {
        throw new Error('Não foi possível carregar as calibrações')
      }
      const data: Run[] = await response.json()
      setRuns(data)
    } catch (err) {
      console.error(err)
      setError(err instanceof Error ? err.message : 'Erro ao carregar calibrações')
    } finally {
      setLoadingRuns(false)
    }
  }

  const fetchDatasets = async () => {
    setLoadingDatasets(true)
    try {
      const response = await fetch(`${API_BASE_URL}/datasets`)
      if (!response.ok) {
        throw new Error('Não foi possível carregar os datasets')
      }
      const data: Dataset[] = await response.json()
      setDatasets(data)
    } catch (err) {
      console.error(err)
    } finally {
      setLoadingDatasets(false)
    }
  }

  useEffect(() => {
    fetchRuns()
    fetchDatasets()
  }, [])

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Calibrações (Runs)</h1>
          <p className="text-slate-600">Visualize execuções de calibração e seus resultados.</p>
        </div>
        <div className="rounded-xl border border-emerald-100 bg-emerald-50 px-4 py-3 shadow-sm">
          <button
            type="button"
            onClick={() => setIsCreateOpen(true)}
            className="inline-flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-emerald-700"
          >
            <span aria-hidden="true">＋</span> Nova calibração
          </button>
        </div>
      </div>

      <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">Lista de calibrações</h2>
            <p className="text-sm text-slate-600">Todas as execuções de calibração disponíveis.</p>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={fetchRuns}
              className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
            >
              Atualizar
            </button>
          </div>
        </div>

        {error ? (
          <div className="mb-4 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">{error}</div>
        ) : null}

        <RunListTable
          runs={runs}
          datasets={datasets}
          loading={loading}
          onView={(run) => navigate(`/runs/${run.id}`)}
        />
      </div>

      <RunCreateModal
        isOpen={isCreateOpen}
        datasets={datasets}
        apiBaseUrl={API_BASE_URL}
        onClose={() => setIsCreateOpen(false)}
        onCreated={(created) => {
          setRuns((prev) => [created, ...prev])
        }}
      />
    </div>
  )
}

export default Runs
