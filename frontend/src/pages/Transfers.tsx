import { useEffect, useState } from 'react'

import { api } from '../api/client'
import type { CalibrationTransfer } from '../types/transfers'

const badge = (label: string, variant: 'success' | 'warning') => {
  const colors =
    variant === 'success'
      ? 'bg-emerald-50 text-emerald-700'
      : 'bg-amber-50 text-amber-700'
  return <span className={`inline-flex items-center rounded-full px-2.5 py-1 text-xs font-semibold ${colors}`}>{label}</span>
}

const Transfers = () => {
  const [transfers, setTransfers] = useState<CalibrationTransfer[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [fittingId, setFittingId] = useState<number | string | null>(null)

  const fetchTransfers = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await api.get<CalibrationTransfer[]>('/calibration-transfers')
      setTransfers(response)
    } catch (err) {
      console.error(err)
      setError(err instanceof Error ? err.message : 'Erro ao carregar transferências')
    } finally {
      setLoading(false)
    }
  }

  const handleFitDs = async (transferId: number | string) => {
    setFittingId(transferId)
    try {
      await api.post(`/calibration-transfers/${transferId}/fit-ds`)
      await fetchTransfers()
    } catch (err) {
      console.error(err)
      setError(err instanceof Error ? err.message : 'Falha ao ajustar DS')
    } finally {
      setFittingId(null)
    }
  }

  useEffect(() => {
    fetchTransfers()
  }, [])

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-sm font-semibold uppercase tracking-wide text-indigo-600">Transferências</p>
          <h1 className="text-3xl font-bold text-slate-900">Transfers</h1>
          <p className="text-slate-600">Aplique e acompanhe transferências de calibração.</p>
        </div>
        <button
          type="button"
          onClick={fetchTransfers}
          className="inline-flex items-center gap-2 rounded-lg border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
        >
          Atualizar
        </button>
      </div>

      {error ? (
        <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div>
      ) : null}

      {loading ? (
        <div className="flex items-center justify-center rounded-xl border border-dashed border-slate-200 bg-white py-12 text-slate-500">
          Carregando transferências...
        </div>
      ) : transfers.length === 0 ? (
        <div className="flex items-center justify-center rounded-xl border border-dashed border-slate-200 bg-white py-12 text-slate-500">
          Nenhuma transferência cadastrada ainda.
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-slate-200">
            <thead className="bg-slate-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-600">Nome</th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-600">Método</th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-600">Modelo mestre</th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-600">Instrumentos</th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-600">DS</th>
                <th className="px-4 py-3 text-left text-xs font-semibold uppercase tracking-wide text-slate-600">Ações</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 bg-white">
              {transfers.map((transfer) => {
                const hasDs = Boolean(transfer.artifacts?.F_ds)
                return (
                  <tr key={transfer.id}>
                    <td className="whitespace-nowrap px-4 py-3 text-sm font-semibold text-slate-900">{transfer.name}</td>
                    <td className="whitespace-nowrap px-4 py-3 text-sm text-slate-700">{transfer.method ? transfer.method.toUpperCase() : '—'}</td>
                    <td className="whitespace-nowrap px-4 py-3 text-sm text-slate-700">{transfer.master_model_id ?? '—'}</td>
                    <td className="whitespace-nowrap px-4 py-3 text-sm text-slate-700">
                      <div className="flex flex-col">
                        <span>Mestre: {transfer.master_instrument_id ?? '—'}</span>
                        <span>Escravo: {transfer.slave_instrument_id ?? '—'}</span>
                      </div>
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 text-sm text-slate-700">
                      {hasDs ? badge('DS ajustada', 'success') : badge('DS não ajustada', 'warning')}
                    </td>
                    <td className="whitespace-nowrap px-4 py-3 text-sm">
                      <div className="flex items-center gap-2">
                        <button
                          type="button"
                          onClick={() => handleFitDs(transfer.id)}
                          disabled={fittingId === transfer.id}
                          className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm font-semibold text-indigo-600 transition hover:border-indigo-200 hover:bg-indigo-50 disabled:cursor-not-allowed disabled:opacity-70"
                        >
                          {fittingId === transfer.id ? 'Ajustando...' : 'Ajustar DS'}
                        </button>
                      </div>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

export default Transfers
