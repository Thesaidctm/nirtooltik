import { useMemo } from 'react'
import { useNavigate } from 'react-router-dom'

import type { Dataset } from '../../types/datasets'

interface DatasetDetailsDrawerProps {
  dataset: Dataset | null
  isOpen: boolean
  onClose: () => void
}

const DatasetDetailsDrawer = ({ dataset, isOpen, onClose }: DatasetDetailsDrawerProps) => {
  const navigate = useNavigate()

  const targets = useMemo(() => {
    if (!dataset?.manifest?.target_cols) return '-'
    return Array.isArray(dataset.manifest.target_cols)
      ? dataset.manifest.target_cols.join(', ')
      : dataset.manifest.target_cols
  }, [dataset])

  const handleCreateRun = () => {
    if (!dataset) return
    navigate(`/runs?dataset_id=${dataset.id}`)
  }

  if (!isOpen || !dataset) return null

  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-slate-900/40">
      <div className="flex h-full w-full max-w-lg flex-col bg-white shadow-2xl ring-1 ring-slate-100">
        <div className="flex items-center justify-between border-b border-slate-200 px-6 py-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-indigo-600">Dataset</p>
            <h3 className="text-xl font-semibold text-slate-900">{dataset.name}</h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-full p-2 text-slate-500 transition hover:bg-slate-100 hover:text-slate-700"
          >
            <span className="sr-only">Fechar</span>
            ✕
          </button>
        </div>

        <div className="flex-1 space-y-4 overflow-y-auto px-6 py-4">
          <div className="rounded-xl border border-slate-200 p-4">
            <h4 className="text-sm font-semibold text-slate-800">Informações</h4>
            <dl className="mt-3 space-y-3 text-sm text-slate-700">
              <div className="flex items-start justify-between gap-4">
                <dt className="text-slate-500">Instrumento</dt>
                <dd className="text-right font-medium text-slate-900">
                  {dataset.instrument_name ?? dataset.instrument_id ?? '-'}
                </dd>
              </div>
              <div className="flex items-start justify-between gap-4">
                <dt className="text-slate-500">Amostras</dt>
                <dd className="text-right font-medium text-slate-900">{dataset.manifest?.n_samples ?? '-'}</dd>
              </div>
              <div className="flex items-start justify-between gap-4">
                <dt className="text-slate-500">Variáveis</dt>
                <dd className="text-right font-medium text-slate-900">{dataset.manifest?.n_features ?? '-'}</dd>
              </div>
              <div className="flex items-start justify-between gap-4">
                <dt className="text-slate-500">Targets</dt>
                <dd className="text-right font-medium text-slate-900">{targets}</dd>
              </div>
            </dl>
          </div>

          <div className="rounded-xl border border-indigo-100 bg-indigo-50 p-4">
            <h4 className="text-sm font-semibold text-indigo-900">Próximos passos</h4>
            <p className="mt-1 text-sm text-indigo-800">Crie uma calibração usando este dataset.</p>
            <button
              type="button"
              onClick={handleCreateRun}
              className="mt-3 inline-flex items-center justify-center rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700"
            >
              Criar calibração a partir deste dataset
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default DatasetDetailsDrawer
