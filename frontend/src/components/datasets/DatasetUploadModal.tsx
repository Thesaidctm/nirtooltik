import { useEffect, useMemo, useState } from 'react'

import type { Instrument } from '../../types/datasets'

interface DatasetUploadModalProps {
  isOpen: boolean
  apiBaseUrl: string
  onClose: () => void
  onUploaded: () => void
}

const DatasetUploadModal = ({ isOpen, apiBaseUrl, onClose, onUploaded }: DatasetUploadModalProps) => {
  const [name, setName] = useState('')
  const [instrumentId, setInstrumentId] = useState('')
  const [targetCols, setTargetCols] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [instruments, setInstruments] = useState<Instrument[]>([])
  const [loadingInstruments, setLoadingInstruments] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const showModal = isOpen

  useEffect(() => {
    const fetchInstruments = async () => {
      setLoadingInstruments(true)
      setError(null)
      try {
        const response = await fetch(`${apiBaseUrl}/instruments`)
        if (!response.ok) {
          throw new Error('Não foi possível carregar os instrumentos')
        }
        const data: Instrument[] = await response.json()
        setInstruments(data)
      } catch (err) {
        console.error(err)
        setError(err instanceof Error ? err.message : 'Erro ao carregar instrumentos')
      } finally {
        setLoadingInstruments(false)
      }
    }

    if (showModal) {
      fetchInstruments()
    }
  }, [apiBaseUrl, showModal])

  const resetForm = () => {
    setName('')
    setInstrumentId('')
    setTargetCols('')
    setFile(null)
    setError(null)
  }

  const handleClose = () => {
    resetForm()
    onClose()
  }

  const canSubmit = useMemo(() => Boolean(name.trim() && instrumentId && file), [file, instrumentId, name])

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault()
    if (!canSubmit) {
      setError('Preencha nome, instrumento e selecione um arquivo')
      return
    }

    const formData = new FormData()
    formData.append('name', name.trim())
    formData.append('instrument_id', instrumentId)
    if (targetCols.trim()) {
      formData.append('target_cols', targetCols.trim())
    }
    if (file) {
      formData.append('file', file)
    }

    setSubmitting(true)
    setError(null)

    try {
      const response = await fetch(`${apiBaseUrl}/datasets`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        throw new Error('Não foi possível enviar o dataset')
      }

      onUploaded()
      handleClose()
    } catch (err) {
      console.error(err)
      setError(err instanceof Error ? err.message : 'Erro ao enviar dataset')
    } finally {
      setSubmitting(false)
    }
  }

  if (!showModal) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 p-4">
      <div className="w-full max-w-2xl rounded-2xl bg-white shadow-xl ring-1 ring-slate-100">
        <div className="flex items-center justify-between border-b border-slate-200 px-6 py-4">
          <div>
            <h3 className="text-lg font-semibold text-slate-900">Novo Dataset</h3>
            <p className="text-sm text-slate-600">Preencha os dados e envie um arquivo CSV ou XLSX.</p>
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
          {error ? (
            <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div>
          ) : null}

          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            <div className="space-y-2">
              <label htmlFor="dataset-name" className="text-sm font-medium text-slate-800">
                Nome
              </label>
              <input
                id="dataset-name"
                type="text"
                value={name}
                onChange={(event) => setName(event.target.value)}
                className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-900 shadow-sm transition focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                placeholder="Ex: Amostras de milho 2024"
                required
              />
            </div>

            <div className="space-y-2">
              <label htmlFor="instrument" className="text-sm font-medium text-slate-800">
                Instrumento
              </label>
              <select
                id="instrument"
                value={instrumentId}
                onChange={(event) => setInstrumentId(event.target.value)}
                disabled={loadingInstruments}
                className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-900 shadow-sm transition focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 disabled:cursor-not-allowed disabled:bg-slate-100"
                required
              >
                <option value="">Selecione</option>
                {instruments.map((instrument) => (
                  <option key={instrument.id} value={instrument.id}>
                    {instrument.name}
                  </option>
                ))}
              </select>
              {loadingInstruments ? (
                <p className="text-xs text-slate-500">Carregando instrumentos...</p>
              ) : null}
            </div>
          </div>

          <div className="space-y-2">
            <label htmlFor="file" className="text-sm font-medium text-slate-800">
              Arquivo (CSV ou XLSX)
            </label>
            <input
              id="file"
              type="file"
              accept=".csv,.xlsx,.xls"
              onChange={(event) => setFile(event.target.files?.[0] ?? null)}
              className="w-full rounded-lg border border-dashed border-slate-300 px-3 py-10 text-sm text-slate-600 file:mr-4 file:rounded-md file:border-0 file:bg-indigo-50 file:px-3 file:py-2 file:text-sm file:font-semibold file:text-indigo-700 hover:border-indigo-200"
              required
            />
          </div>

          <div className="space-y-2">
            <label htmlFor="target-cols" className="text-sm font-medium text-slate-800">
              Targets (opcional)
            </label>
            <input
              id="target-cols"
              type="text"
              value={targetCols}
              onChange={(event) => setTargetCols(event.target.value)}
              className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-900 shadow-sm transition focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
              placeholder="Ex: protein,moisture"
            />
            <p className="text-xs text-slate-500">Separe múltiplos alvos com vírgula.</p>
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
              disabled={submitting || !canSubmit}
              className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-indigo-300"
            >
              {submitting ? 'Enviando...' : 'Confirmar envio'}
            </button>
          </div>
        </form>
      </div>
    </div>
  )
}

export default DatasetUploadModal
