import type { Dataset } from '../../types/datasets'

interface DatasetTableProps {
  datasets: Dataset[]
  loading?: boolean
  onSelect?: (dataset: Dataset) => void
}

const formatDate = (value?: string | null) => {
  if (!value) return '-'
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return value
  return parsed.toLocaleDateString()
}

const DatasetTable = ({ datasets, loading, onSelect }: DatasetTableProps) => {
  if (loading) {
    return <div className="flex items-center justify-center py-12 text-slate-500">Carregando datasets...</div>
  }

  if (!datasets.length) {
    return <div className="flex items-center justify-center py-12 text-slate-500">Nenhum dataset cadastrado ainda.</div>
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-slate-200">
        <thead className="bg-slate-50">
          <tr>
            <th scope="col" className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-600">
              Nome
            </th>
            <th scope="col" className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-600">
              Amostras
            </th>
            <th scope="col" className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-600">
              Variáveis
            </th>
            <th scope="col" className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-600">
              Criado em
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-200 bg-white">
          {datasets.map((dataset) => {
            const clickable = Boolean(onSelect)
            return (
              <tr
                key={dataset.id}
                onClick={() => onSelect?.(dataset)}
                onKeyDown={(event) => {
                  if (!clickable) return
                  if (event.key === 'Enter' || event.key === ' ') {
                    event.preventDefault()
                    onSelect?.(dataset)
                  }
                }}
                tabIndex={clickable ? 0 : -1}
                className={`transition hover:bg-slate-50 ${clickable ? 'cursor-pointer focus:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500' : ''}`}
              >
                <td className="whitespace-nowrap px-4 py-3 text-sm font-medium text-slate-900">
                  <div className="flex flex-col">
                    <span>{dataset.name}</span>
                    {dataset.description ? (
                      <span className="text-xs font-normal text-slate-500">{dataset.description}</span>
                    ) : null}
                  </div>
                </td>
                <td className="whitespace-nowrap px-4 py-3 text-sm text-slate-700">{dataset.n_samples ?? dataset.manifest?.n_samples ?? '-'}
                </td>
                <td className="whitespace-nowrap px-4 py-3 text-sm text-slate-700">{dataset.n_features ?? dataset.manifest?.n_features ?? '-'}
                </td>
                <td className="whitespace-nowrap px-4 py-3 text-sm text-slate-700">{formatDate(dataset.created_at ?? dataset.updated_at)}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

export default DatasetTable
