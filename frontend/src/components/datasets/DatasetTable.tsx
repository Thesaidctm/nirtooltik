import type { Dataset } from '../../types/datasets'

interface DatasetTableProps {
  datasets: Dataset[]
  loading?: boolean
  onDetails: (dataset: Dataset) => void
}

const formatDate = (value?: string) => {
  if (!value) return '-'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleString()
}

const resolveTargets = (manifestTargets?: string[] | string) => {
  if (!manifestTargets) return '-'
  if (Array.isArray(manifestTargets)) {
    return manifestTargets.length ? manifestTargets.join(', ') : '-'
  }
  return manifestTargets
}

const DatasetTable = ({ datasets, loading, onDetails }: DatasetTableProps) => {
  if (loading) {
    return (
      <div className="flex items-center justify-center py-12 text-slate-500">
        Carregando datasets...
      </div>
    )
  }

  if (!datasets.length) {
    return (
      <div className="flex items-center justify-center py-12 text-slate-500">
        Nenhum dataset encontrado.
      </div>
    )
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
              Instrumento
            </th>
            <th scope="col" className="px-4 py-3 text-left text-sm font-semibold text-slate-600">
              #Amostras
            </th>
            <th scope="col" className="px-4 py-3 text-left text-sm font-semibold text-slate-600">
              #Variáveis
            </th>
            <th scope="col" className="px-4 py-3 text-left text-sm font-semibold text-slate-600">
              Targets
            </th>
            <th scope="col" className="px-4 py-3 text-left text-sm font-semibold text-slate-600">
              Data de criação
            </th>
            <th scope="col" className="px-4 py-3 text-left text-sm font-semibold text-slate-600">
              Ação
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-200 bg-white">
          {datasets.map((dataset) => {
            const { id, name, instrument_name, instrument_id, manifest, created_at } = dataset
            return (
              <tr key={id} className="hover:bg-slate-50">
                <td className="whitespace-nowrap px-4 py-3 text-sm font-medium text-slate-900">{name}</td>
                <td className="whitespace-nowrap px-4 py-3 text-sm text-slate-700">
                  {instrument_name ?? instrument_id ?? '-'}
                </td>
                <td className="whitespace-nowrap px-4 py-3 text-sm text-slate-700">{manifest?.n_samples ?? '-'}</td>
                <td className="whitespace-nowrap px-4 py-3 text-sm text-slate-700">{manifest?.n_features ?? '-'}</td>
                <td className="whitespace-nowrap px-4 py-3 text-sm text-slate-700">
                  {resolveTargets(manifest?.target_cols)}
                </td>
                <td className="whitespace-nowrap px-4 py-3 text-sm text-slate-700">{formatDate(created_at)}</td>
                <td className="whitespace-nowrap px-4 py-3 text-sm">
                  <button
                    type="button"
                    onClick={() => onDetails(dataset)}
                    className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm font-semibold text-indigo-600 transition hover:border-indigo-200 hover:bg-indigo-50"
                  >
                    Detalhes
                  </button>
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

export default DatasetTable
