import { useCallback, useEffect, useMemo, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'

import type { CalibrationTransfer } from '../types/transfers'
import { parseSpectraFile, parseSpectraFromText } from '../utils/spectraParsers'

const API_BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const TransferDetails = () => {
  const { transferId } = useParams()
  const navigate = useNavigate()

  const [transfer, setTransfer] = useState<CalibrationTransfer | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const [spectraText, setSpectraText] = useState('')
  const [spectraFile, setSpectraFile] = useState<File | null>(null)
  const [predicting, setPredicting] = useState(false)
  const [predictError, setPredictError] = useState<string | null>(null)
  const [predictions, setPredictions] = useState<number[]>([])

  const fetchTransfer = useCallback(async () => {
    if (!transferId) return
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_BASE_URL}/calibration-transfers/${transferId}`)
      if (!response.ok) {
        throw new Error('Não foi possível carregar a transferência')
      }
      const data: CalibrationTransfer = await response.json()
      setTransfer(data)
    } catch (err) {
      console.error(err)
      setError(err instanceof Error ? err.message : 'Erro ao carregar transferência')
    } finally {
      setLoading(false)
    }
  }, [transferId])

  useEffect(() => {
    fetchTransfer()
  }, [fetchTransfer])

  const dsStatus = useMemo(() => (transfer?.artifacts && 'F_ds' in transfer.artifacts ? 'DS ajustado' : 'Sem DS ajustado'), [transfer?.artifacts])

  const parseSpectraInput = async (): Promise<number[][]> => {
    if (spectraFile) {
      return parseSpectraFile(spectraFile)
    }

    if (spectraText.trim()) {
      return parseSpectraFromText(spectraText)
    }

    throw new Error('Forneça um arquivo ou cole os espectros no campo de texto')
  }

  const handlePredict = async () => {
    if (!transferId) return
    setPredictError(null)
    setPredicting(true)

    try {
      const spectra = await parseSpectraInput()
      if (spectra.length === 0) {
        throw new Error('Não foi possível extrair espectros válidos do arquivo/texto informado')
      }

      const response = await fetch(`${API_BASE_URL}/calibration-transfers/${transferId}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ spectra }),
      })

      if (!response.ok) {
        throw new Error('Falha ao realizar predição')
      }

      const data: { y_hat: number[] } = await response.json()
      setPredictions(data.y_hat ?? [])
    } catch (err) {
      console.error(err)
      setPredictError(err instanceof Error ? err.message : 'Erro ao executar predição')
    } finally {
      setPredicting(false)
    }
  }

  if (loading) {
    return <p className="text-slate-600">Carregando transferência...</p>
  }

  if (error) {
    return (
      <div className="space-y-3">
        <button
          type="button"
          onClick={() => navigate(-1)}
          className="text-sm font-semibold text-indigo-600 hover:underline"
        >
          ← Voltar
        </button>
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div>
      </div>
    )
  }

  if (!transfer) return null

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-sm font-semibold text-indigo-700">Transferência #{transfer.id}</p>
          <h1 className="text-3xl font-bold text-slate-900">{transfer.name}</h1>
          <p className="text-slate-600">{transfer.method?.toUpperCase()} • {dsStatus}</p>
        </div>
        <button
          type="button"
          onClick={() => navigate('/transfers')}
          className="inline-flex items-center gap-2 rounded-lg border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
        >
          ← Voltar para lista
        </button>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
          <h2 className="text-lg font-semibold text-slate-900">Configuração</h2>
          <dl className="mt-3 grid grid-cols-2 gap-3 text-sm text-slate-700">
            <div>
              <dt className="text-slate-500">Modelo mestre</dt>
              <dd className="font-semibold">{transfer.master_model_name ?? transfer.master_model_id ?? 'N/D'}</dd>
            </div>
            <div>
              <dt className="text-slate-500">Método</dt>
              <dd className="font-semibold">{transfer.method?.toUpperCase() ?? 'N/D'}</dd>
            </div>
            <div>
              <dt className="text-slate-500">Instrumento mestre</dt>
              <dd className="font-semibold">{transfer.master_instrument_name ?? transfer.master_instrument_id ?? 'N/D'}</dd>
            </div>
            <div>
              <dt className="text-slate-500">Instrumento escravo</dt>
              <dd className="font-semibold">{transfer.slave_instrument_name ?? transfer.slave_instrument_id ?? 'N/D'}</dd>
            </div>
            <div>
              <dt className="text-slate-500">Dataset padrão (mestre)</dt>
              <dd className="font-semibold">{transfer.master_dataset_name ?? transfer.master_dataset_id ?? 'N/D'}</dd>
            </div>
            <div>
              <dt className="text-slate-500">Dataset padrão (escravo)</dt>
              <dd className="font-semibold">{transfer.slave_dataset_name ?? transfer.slave_dataset_id ?? 'N/D'}</dd>
            </div>
          </dl>
        </div>

        <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
          <h2 className="text-lg font-semibold text-slate-900">Artefatos</h2>
          {transfer.artifacts && 'F_ds' in transfer.artifacts ? (
            <div className="mt-3 space-y-2 text-sm text-slate-700">
              <p className="flex items-center gap-2 font-semibold text-emerald-700">
                <span className="h-2 w-2 rounded-full bg-emerald-500" aria-hidden /> DS ajustado
              </p>
              {transfer.artifacts.base_dir ? <p>base_dir: {transfer.artifacts.base_dir}</p> : null}
              {transfer.artifacts.shape ? <p>shape: {Array.isArray(transfer.artifacts.shape) ? transfer.artifacts.shape.join(' x ') : String(transfer.artifacts.shape)}</p> : null}
            </div>
          ) : (
            <p className="mt-3 text-sm text-amber-700">Nenhum F_ds salvo para esta transferência.</p>
          )}
        </div>
      </div>

      <div className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
        <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">Teste rápido</h2>
            <p className="text-sm text-slate-600">Envie espectros do instrumento escravo para obter previsões pelo modelo transferido.</p>
          </div>
          <div className="text-sm text-slate-500">Formato: CSV ou XLSX com valores numéricos</div>
        </div>

        <div className="mt-4 grid gap-4 md:grid-cols-2">
          <label className="space-y-2 text-sm text-slate-700">
            Upload de arquivo
            <input
              type="file"
              accept=".csv,.xlsx,.xls,.txt"
              onChange={(event) => setSpectraFile(event.target.files?.[0] ?? null)}
              className="w-full rounded-lg border border-slate-200 px-3 py-2 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
            />
          </label>

          <label className="space-y-2 text-sm text-slate-700">
            Ou cole os espectros
            <textarea
              value={spectraText}
              onChange={(event) => setSpectraText(event.target.value)}
              rows={6}
              placeholder="10.1, 0.2, 0.3\n11.5, 0.1, 0.4"
              className="w-full rounded-lg border border-slate-200 px-3 py-2 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
            />
          </label>
        </div>

        {predictError ? <div className="mt-3 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{predictError}</div> : null}

        <div className="mt-4 flex items-center gap-3">
          <button
            type="button"
            onClick={handlePredict}
            disabled={predicting}
            className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-indigo-300"
          >
            {predicting ? 'Gerando previsões...' : 'Prever'}
          </button>
          <button
            type="button"
            onClick={() => {
              setSpectraFile(null)
              setSpectraText('')
              setPredictions([])
              setPredictError(null)
            }}
            className="text-sm font-semibold text-slate-600 hover:underline"
          >
            Limpar
          </button>
        </div>

        {predictions.length > 0 ? (
          <div className="mt-4 overflow-x-auto">
            <table className="min-w-full divide-y divide-slate-200 text-sm">
              <thead className="bg-slate-50 text-left text-xs font-semibold uppercase tracking-wide text-slate-600">
                <tr>
                  <th className="px-4 py-2">#</th>
                  <th className="px-4 py-2">ŷ</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {predictions.map((value, index) => (
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
  )
}

export default TransferDetails
