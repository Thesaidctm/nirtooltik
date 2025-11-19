import type { Dataset } from '../../types/datasets'
import type { Run } from '../../types/runs'

interface RunListTableProps {
  runs: Run[]
  datasets: Dataset[]
  loading?: boolean
  onView: (run: Run) => void
}

const formatDate = (value?: string) => {
  if (!value) return '-'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleString()
}

const statusBadge = (status: string) => {
  const base = 'inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold'
  const normalized = status?.toLowerCase()

  if (normalized === 'completed') {
    return <span className={`${base} bg-emerald-50 text-emerald-700`}>Concluído</span>
  }
  if (normalized === 'running') {
    return <span className={`${base} bg-amber-50 text-amber-700`}>Em execução</span>
  }
  if (normalized === 'failed') {
    return <span className={`${base} bg-rose-50 text-rose-700`}>Falhou</span>
  }
  return <span className={`${base} bg-slate-100 text-slate-700`}>{status || 'Desconhecido'}</span>
}

const resolveDatasetName = (datasets: Dataset[], id: number | string) => {
  const match = datasets.find((dataset) => `${dataset.id}` === `${id}`)
  return match?.name ?? id
}

const RunListTable = ({ runs, datasets, loading, onView }: RunListTableProps) => {
  if (loading) {
    return (
      <div className="flex items-center justify-center py-12 text-slate-500">Carregando calibrações...</div>
    )
  }

  if (!runs.length) {
    return <div className="flex items-center justify-center py-12 text-slate-500">Nenhuma calibração encontrada.</div>
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-slate-200">
        <thead className="bg-slate-50">
          <tr>
            <th scope="col" className="px-4 py-3 text-left text-sm font-semibold text-slate-600">
              Nome
            </th>
            <th scope="col" className="px-4 py-3 text-left text-sm font-semibold text-slate-600">
              Dataset
            </th>
            <th scope="col" className="px-4 py-3 text-left text-sm font-semibold text-slate-600">
              Status
            </th>
            <th scope="col" className="px-4 py-3 text-left text-sm font-semibold text-slate-600">
              Criado em
            </th>
            <th scope="col" className="px-4 py-3 text-left text-sm font-semibold text-slate-600">
              Ação
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-200 bg-white">
          {runs.map((run) => (
            <tr key={run.id} className="hover:bg-slate-50">
              <td className="whitespace-nowrap px-4 py-3 text-sm font-medium text-slate-900">{run.name}</td>
              <td className="whitespace-nowrap px-4 py-3 text-sm text-slate-700">
                {resolveDatasetName(datasets, run.dataset_id)}
              </td>
              <td className="whitespace-nowrap px-4 py-3 text-sm text-slate-700">{statusBadge(run.status)}</td>
              <td className="whitespace-nowrap px-4 py-3 text-sm text-slate-700">{formatDate(run.created_at)}</td>
              <td className="whitespace-nowrap px-4 py-3 text-sm">
                <button
                  type="button"
                  onClick={() => onView(run)}
                  className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm font-semibold text-indigo-600 transition hover:border-indigo-200 hover:bg-indigo-50"
                >
                  Ver detalhes
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export default RunListTable
