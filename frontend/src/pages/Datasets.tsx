import { useEffect, useState } from 'react'

import DatasetDetailsDrawer from '../components/datasets/DatasetDetailsDrawer'
import DatasetTable from '../components/datasets/DatasetTable'
import DatasetUploadModal from '../components/datasets/DatasetUploadModal'
import type { Dataset } from '../types/datasets'

const API_BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const Datasets = () => {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isUploadOpen, setIsUploadOpen] = useState(false)
  const [selectedDataset, setSelectedDataset] = useState<Dataset | null>(null)

  const fetchDatasets = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_BASE_URL}/datasets`)
      if (!response.ok) {
        throw new Error('Não foi possível carregar os datasets')
      }
      const data: Dataset[] = await response.json()
      setDatasets(data)
    } catch (err) {
      console.error(err)
      setError(err instanceof Error ? err.message : 'Erro ao carregar datasets')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchDatasets()
  }, [])

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Gestão de Datasets</h1>
          <p className="text-slate-600">Organize e visualize coleções de dados para calibração.</p>
        </div>
        <div className="rounded-xl border border-indigo-100 bg-indigo-50 px-4 py-3 shadow-sm">
          <button
            type="button"
            onClick={() => setIsUploadOpen(true)}
            className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700"
          >
            <span aria-hidden="true">＋</span> Novo Dataset
          </button>
        </div>
      </div>

      <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">Lista de datasets</h2>
            <p className="text-sm text-slate-600">Visualize os conjuntos de dados já disponíveis.</p>
          </div>
          <button
            type="button"
            onClick={fetchDatasets}
            className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
          >
            Atualizar
          </button>
        </div>

        {error ? (
          <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div>
        ) : null}

        <DatasetTable datasets={datasets} loading={loading} onDetails={setSelectedDataset} />
      </div>

      <DatasetUploadModal
        isOpen={isUploadOpen}
        apiBaseUrl={API_BASE_URL}
        onClose={() => setIsUploadOpen(false)}
        onUploaded={fetchDatasets}
      />

      <DatasetDetailsDrawer
        dataset={selectedDataset}
        isOpen={selectedDataset !== null}
        onClose={() => setSelectedDataset(null)}
      />
    </div>
  )
}

export default Datasets
