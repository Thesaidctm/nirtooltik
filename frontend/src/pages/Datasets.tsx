import { useEffect, useState } from 'react'

import DatasetDetailsDrawer from '../components/datasets/DatasetDetailsDrawer'
import DatasetTable from '../components/datasets/DatasetTable'
import DatasetUploadModal from '../components/datasets/DatasetUploadModal'
import { api } from '../api/client'
import type { Dataset, DatasetApiResponse } from '../types/datasets'
import { normalizeDataset } from '../types/datasets'

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
      const response = await api.get<DatasetApiResponse[]>('/datasets/')
      setDatasets(response.map(normalizeDataset))
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
          <p className="text-sm font-semibold uppercase tracking-wide text-indigo-600">Dados NIR</p>
          <h1 className="text-3xl font-bold text-slate-900">Datasets</h1>
          <p className="text-slate-600">Gerencie coleções de espectros e envie novos arquivos.</p>
        </div>
        <button
          type="button"
          onClick={() => setIsUploadOpen(true)}
          className="inline-flex items-center gap-2 rounded-xl bg-indigo-600 px-5 py-3 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700"
        >
          <span className="text-lg" aria-hidden>
            ＋
          </span>
          Novo dataset
        </button>
      </div>

      <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="mb-4 flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">Lista de datasets</h2>
            <p className="text-sm text-slate-600">Os dados disponíveis serão mostrados abaixo.</p>
          </div>
          <button
            type="button"
            onClick={fetchDatasets}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
          >
            Atualizar
          </button>
        </div>

        {error ? (
          <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div>
        ) : null}

        <DatasetTable datasets={datasets} loading={loading} onSelect={setSelectedDataset} />
      </div>

      <DatasetUploadModal isOpen={isUploadOpen} onClose={() => setIsUploadOpen(false)} onUploaded={fetchDatasets} />

      <DatasetDetailsDrawer dataset={selectedDataset} isOpen={Boolean(selectedDataset)} onClose={() => setSelectedDataset(null)} />
    </div>
  )
}

export default Datasets
