import { useEffect, useState } from 'react'

import { api } from '../api/client'
import type { ModelSummary } from '../types/models'
import type { CalibrationTransfer } from '../types/transfers'
import { parseSpectraFile, parseSpectraFromText } from '../utils/spectraParsers'

const Predict = () => {
  const [models, setModels] = useState<ModelSummary[]>([])
  const [transfers, setTransfers] = useState<CalibrationTransfer[]>([])

  const [selectedModelId, setSelectedModelId] = useState('')
  const [modelFile, setModelFile] = useState<File | null>(null)
  const [modelText, setModelText] = useState('')
  const [modelPredictions, setModelPredictions] = useState<number[]>([])
  const [modelError, setModelError] = useState<string | null>(null)
  const [predictingModel, setPredictingModel] = useState(false)

  const [selectedTransferId, setSelectedTransferId] = useState('')
  const [transferFile, setTransferFile] = useState<File | null>(null)
  const [transferText, setTransferText] = useState('')
  const [transferPredictions, setTransferPredictions] = useState<number[]>([])
  const [transferError, setTransferError] = useState<string | null>(null)
  const [predictingTransfer, setPredictingTransfer] = useState(false)

  const fetchModels = async () => {
    try {
      const data = await api.get<ModelSummary[]>('/models')
      setModels(data)
    } catch (err) {
      console.error(err)
    }
  }

  const fetchTransfers = async () => {
    try {
      const data = await api.get<CalibrationTransfer[]>('/calibration-transfers')
      setTransfers(data)
    } catch (err) {
      console.error(err)
    }
  }

  useEffect(() => {
    fetchModels()
    fetchTransfers()
  }, [])

  const parseInput = async (file: File | null, text: string) => {
    if (file) {
      return parseSpectraFile(file)
    }
    if (text.trim()) {
      return parseSpectraFromText(text)
    }
    throw new Error('Envie um arquivo ou cole os espectros para continuar')
  }

  const handleModelPredict = async () => {
    setModelError(null)
    setModelPredictions([])

    if (!selectedModelId) {
      setModelError('Selecione um modelo para predição direta')
      return
    }

    setPredictingModel(true)
    try {
      const spectra = await parseInput(modelFile, modelText)
      if (spectra.length === 0) {
        throw new Error('Nenhuma linha válida de espectros foi encontrada')
      }

      const data = await api.post<{ y_hat: number[] }>(`/models/${selectedModelId}/predict`, {
        spectra,
      })
      setModelPredictions(data.y_hat ?? [])
    } catch (err) {
      console.error(err)
      setModelError(err instanceof Error ? err.message : 'Erro ao processar predição')
    } finally {
      setPredictingModel(false)
    }
  }

  const handleTransferPredict = async () => {
    setTransferError(null)
    setTransferPredictions([])

    if (!selectedTransferId) {
      setTransferError('Selecione uma transferência para predição')
      return
    }

    setPredictingTransfer(true)
    try {
      const spectra = await parseInput(transferFile, transferText)
      if (spectra.length === 0) {
        throw new Error('Nenhuma linha válida de espectros foi encontrada')
      }

      const data = await api.post<{ y_hat: number[] }>(
        `/calibration-transfers/${selectedTransferId}/predict`,
        {
          spectra,
        },
      )
      setTransferPredictions(data.y_hat ?? [])
    } catch (err) {
      console.error(err)
      setTransferError(err instanceof Error ? err.message : 'Erro ao processar predição com transferência')
    } finally {
      setPredictingTransfer(false)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-slate-900">Predição Online</h1>
        <p className="text-slate-600">Realize inferências em tempo real com modelos registrados ou aplicando transferência de calibração.</p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-slate-900">Predição direta (modelo registrado)</h2>
              <p className="text-sm text-slate-600">Selecione um modelo e envie espectros para obter previsões imediatas.</p>
            </div>
          </div>

          <div className="mt-4 space-y-3">
            <label className="space-y-1 text-sm text-slate-700">
              Modelo
              <select
                value={selectedModelId}
                onChange={(event) => setSelectedModelId(event.target.value)}
                className="w-full rounded-lg border border-slate-200 px-3 py-2 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
              >
                <option value="">Selecione um modelo</option>
                {models.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name}
                  </option>
                ))}
              </select>
            </label>

            <label className="space-y-1 text-sm text-slate-700">
              Upload de espectros (CSV/XLSX)
              <input
                type="file"
                accept=".csv,.xlsx,.xls,.txt"
                onChange={(event) => setModelFile(event.target.files?.[0] ?? null)}
                className="w-full rounded-lg border border-slate-200 px-3 py-2 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
              />
            </label>

            <label className="space-y-1 text-sm text-slate-700">
              Ou cole os dados
              <textarea
                value={modelText}
                onChange={(event) => setModelText(event.target.value)}
                rows={6}
                placeholder="10.1, 0.2, 0.3\n11.5, 0.1, 0.4"
                className="w-full rounded-lg border border-slate-200 px-3 py-2 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
              />
            </label>

            {modelError ? <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{modelError}</div> : null}

            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={handleModelPredict}
                disabled={predictingModel}
                className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-indigo-300"
              >
                {predictingModel ? 'Prevendo...' : 'Prever'}
              </button>
              <button
                type="button"
                onClick={() => {
                  setModelFile(null)
                  setModelText('')
                  setModelPredictions([])
                  setModelError(null)
                }}
                className="text-sm font-semibold text-slate-600 hover:underline"
              >
                Limpar
              </button>
            </div>

            {modelPredictions.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="mt-3 min-w-full divide-y divide-slate-200 text-sm">
                  <thead className="bg-slate-50 text-left text-xs font-semibold uppercase tracking-wide text-slate-600">
                    <tr>
                      <th className="px-4 py-2">#</th>
                      <th className="px-4 py-2">ŷ</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {modelPredictions.map((value, index) => (
                      <tr key={index}>
                        <td className="px-4 py-2 text-slate-700">{index + 1}</td>
                        <td className="px-4 py-2 font-semibold text-slate-900">{value}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
          </div>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-slate-900">Predição com Transferência de Calibração</h2>
              <p className="text-sm text-slate-600">Selecione uma transferência mestre → escravo para aplicar DS automaticamente.</p>
            </div>
          </div>

          <div className="mt-4 space-y-3">
            <label className="space-y-1 text-sm text-slate-700">
              Transferência
              <select
                value={selectedTransferId}
                onChange={(event) => setSelectedTransferId(event.target.value)}
                className="w-full rounded-lg border border-slate-200 px-3 py-2 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
              >
                <option value="">Selecione uma transferência</option>
                {transfers.map((transfer) => (
                  <option key={transfer.id} value={transfer.id}>
                    {transfer.name}
                  </option>
                ))}
              </select>
            </label>

            <label className="space-y-1 text-sm text-slate-700">
              Upload de espectros (escravo)
              <input
                type="file"
                accept=".csv,.xlsx,.xls,.txt"
                onChange={(event) => setTransferFile(event.target.files?.[0] ?? null)}
                className="w-full rounded-lg border border-slate-200 px-3 py-2 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
              />
            </label>

            <label className="space-y-1 text-sm text-slate-700">
              Ou cole os dados (escravo)
              <textarea
                value={transferText}
                onChange={(event) => setTransferText(event.target.value)}
                rows={6}
                placeholder="10.1, 0.2, 0.3\n11.5, 0.1, 0.4"
                className="w-full rounded-lg border border-slate-200 px-3 py-2 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
              />
            </label>

            {transferError ? <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{transferError}</div> : null}

            <div className="flex items-center gap-3">
              <button
                type="button"
                onClick={handleTransferPredict}
                disabled={predictingTransfer}
                className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-indigo-300"
              >
                {predictingTransfer ? 'Prevendo...' : 'Prever com transferência'}
              </button>
              <button
                type="button"
                onClick={() => {
                  setTransferFile(null)
                  setTransferText('')
                  setTransferPredictions([])
                  setTransferError(null)
                }}
                className="text-sm font-semibold text-slate-600 hover:underline"
              >
                Limpar
              </button>
            </div>

            {transferPredictions.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="mt-3 min-w-full divide-y divide-slate-200 text-sm">
                  <thead className="bg-slate-50 text-left text-xs font-semibold uppercase tracking-wide text-slate-600">
                    <tr>
                      <th className="px-4 py-2">#</th>
                      <th className="px-4 py-2">ŷ</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {transferPredictions.map((value, index) => (
                      <tr key={index}>
                        <td className="px-4 py-2 text-slate-700">{index + 1}</td>
                        <td className="px-4 py-2 font-semibold text-slate-900">{value}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  )
}

export default Predict
