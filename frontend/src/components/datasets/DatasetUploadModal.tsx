import { useEffect, useMemo, useState } from 'react'

import { API_BASE_URL } from '../../api/client'
import type { DatasetApiResponse, Instrument } from '../../types/datasets'
import { normalizeDataset } from '../../types/datasets'

interface DatasetUploadModalProps {
  isOpen: boolean
  onClose: () => void
  onUploaded: () => void
}

const DatasetUploadModal = ({ isOpen, onClose, onUploaded }: DatasetUploadModalProps) => {
  const [name, setName] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [instruments, setInstruments] = useState<Instrument[]>([])
  const [selectedInstrumentId, setSelectedInstrumentId] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [loadingInstruments, setLoadingInstruments] = useState(false)
  const [instrumentsError, setInstrumentsError] = useState<string | null>(null)

  const canSubmit = useMemo(() => Boolean(name.trim() && file), [file, name])

  const loadInstruments = async () => {
    setLoadingInstruments(true)
    setInstrumentsError(null)
    try {
      const response = await fetch(`${API_BASE_URL}/instruments`)
      if (!response.ok) {
        throw new Error(await response.text())
      }
      const data: Instrument[] = await response.json()
      setInstruments(data)
    } catch (err) {
      console.error(err)
      setInstrumentsError(
        'Não foi possível carregar a lista de instrumentos. Você ainda pode cadastrar o dataset sem informar o instrumento.',
      )
    } finally {
      setLoadingInstruments(false)
    }
  }

  const resetForm = () => {
    setName('')
    setFile(null)
    setError(null)
    setSelectedInstrumentId('')
  }

  const handleClose = () => {
    if (submitting) return
    resetForm()
    onClose()
  }

  useEffect(() => {
    if (isOpen) {
      loadInstruments()
    }
  }, [isOpen])

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!canSubmit || !file) {
      setError('Informe o nome do dataset e selecione um arquivo .csv, .xlsx ou .xls')
      return
    }

    setSubmitting(true)
    setError(null)

    try {
      const formData = new FormData()
      formData.append('name', name.trim())
      if (selectedInstrumentId) {
        formData.append('instrument_id', selectedInstrumentId)
      }
      formData.append('file', file)

      const response = await fetch(`${API_BASE_URL}/datasets`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const text = await response.text()
        setError(text || `Erro ${response.status}`)
        return
      }

      const createdResponse: DatasetApiResponse = await response.json()
      normalizeDataset(createdResponse)

      onUploaded()
      resetForm()
      onClose()
    } catch (err) {
      console.error(err)
      setError('Erro ao conectar com a API. Verifique se o servidor está rodando.')
    } finally {
      setSubmitting(false)
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 p-4">
      <div className="w-full max-w-lg rounded-2xl bg-white shadow-xl ring-1 ring-slate-100">
        <div className="flex items-center justify-between border-b border-slate-200 px-6 py-4">
            <div>
              <h3 className="text-lg font-semibold text-slate-900">Novo dataset</h3>
              <p className="text-sm text-slate-600">Crie um registro e envie o arquivo de espectros (.csv, .xlsx ou .xls).</p>
            </div>
          <button
            type="button"
            onClick={handleClose}
            className="rounded-full p-2 text-slate-500 transition hover:bg-slate-100 hover:text-slate-700"
          >
            <span className="sr-only">Fechar</span>
            ✕
          </button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4 px-6 py-4">
          {instrumentsError ? (
            <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
              {instrumentsError}
            </div>
          ) : null}
          {error ? <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div> : null}

          <div className="space-y-2">
            <label htmlFor="dataset-name" className="text-sm font-medium text-slate-800">
              Nome do dataset
            </label>
            <input
              id="dataset-name"
              type="text"
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="Ex: Soja 2024"
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-900 shadow-sm transition focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              required
            />
          </div>

          <div className="space-y-2">
            <label htmlFor="dataset-instrument" className="text-sm font-medium text-slate-800">
              Instrumento
            </label>
            <select
              id="dataset-instrument"
              value={selectedInstrumentId}
              onChange={(event) => setSelectedInstrumentId(event.target.value)}
              disabled={loadingInstruments}
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-900 shadow-sm transition focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:cursor-not-allowed disabled:bg-slate-100"
            >
              <option value="">
                {loadingInstruments ? 'Carregando instrumentos...' : 'Selecione um instrumento (opcional)'}
              </option>
              {instruments.map((instrument) => (
                <option key={instrument.id} value={instrument.id}>
                  {instrument.name}
                </option>
              ))}
            </select>
            <p className="text-xs text-slate-500">Selecione o espectrômetro associado a este dataset (opcional).</p>
          </div>

          <div className="space-y-2">
            <label htmlFor="dataset-file" className="text-sm font-medium text-slate-800">
              Arquivo (.csv, .xlsx ou .xls)
            </label>
            <input
              id="dataset-file"
              type="file"
              accept=".csv,.xlsx,.xls"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              className="w-full rounded-lg border border-dashed border-slate-300 px-3 py-10 text-sm text-slate-600 file:mr-4 file:rounded-md file:border-0 file:bg-indigo-50 file:px-3 file:py-2 file:text-sm file:font-semibold file:text-indigo-700 hover:border-indigo-200"
              required
            />
            <p className="text-xs text-slate-500">O arquivo será anexado ao dataset após a criação do registro.</p>
          </div>

          <div className="flex items-center justify-end gap-2 pt-2">
            <button
              type="button"
              onClick={handleClose}
              className="rounded-lg border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
            >
              Cancelar
            </button>
            <button
              type="submit"
              disabled={!canSubmit || submitting}
              className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-indigo-300"
            >
              {submitting ? 'Enviando...' : 'Salvar e enviar'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default DatasetUploadModal
