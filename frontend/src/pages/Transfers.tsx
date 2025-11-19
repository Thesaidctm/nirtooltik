import { useEffect, useMemo, useState } from 'react'
import { useNavigate } from 'react-router-dom'

import type { Dataset, Instrument } from '../types/datasets'
import type { ModelSummary } from '../types/models'
import type { CalibrationTransfer } from '../types/transfers'

const API_BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const Transfers = () => {
  const [transfers, setTransfers] = useState<CalibrationTransfer[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [fittingId, setFittingId] = useState<string | number | null>(null)

  const [isModalOpen, setIsModalOpen] = useState(false)
  const [models, setModels] = useState<ModelSummary[]>([])
  const [instruments, setInstruments] = useState<Instrument[]>([])
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [loadingOptions, setLoadingOptions] = useState(false)
  const [formError, setFormError] = useState<string | null>(null)
  const [creating, setCreating] = useState(false)

  const [name, setName] = useState('')
  const [masterModelId, setMasterModelId] = useState('')
  const [masterInstrumentId, setMasterInstrumentId] = useState('')
  const [slaveInstrumentId, setSlaveInstrumentId] = useState('')
  const [masterDatasetId, setMasterDatasetId] = useState('')
  const [slaveDatasetId, setSlaveDatasetId] = useState('')
  const [method, setMethod] = useState('ds')

  const navigate = useNavigate()

  const fetchTransfers = async () => {
    setLoading(true)
    setError(null)
    try {
      const response = await fetch(`${API_BASE_URL}/calibration-transfers`)
      if (!response.ok) {
        throw new Error('Não foi possível carregar as transferências')
      }
      const data: CalibrationTransfer[] = await response.json()
      setTransfers(data)
    } catch (err) {
      console.error(err)
      setError(err instanceof Error ? err.message : 'Erro ao carregar transferências')
    } finally {
      setLoading(false)
    }
  }

  const fetchOptions = async () => {
    setLoadingOptions(true)
    setFormError(null)
    try {
      const [modelsResponse, instrumentsResponse, datasetsResponse] = await Promise.all([
        fetch(`${API_BASE_URL}/models`),
        fetch(`${API_BASE_URL}/instruments`),
        fetch(`${API_BASE_URL}/datasets`),
      ])

      if (!modelsResponse.ok || !instrumentsResponse.ok || !datasetsResponse.ok) {
        throw new Error('Não foi possível carregar listas de modelos, instrumentos ou datasets')
      }

      const modelsData: ModelSummary[] = await modelsResponse.json()
      const instrumentsData: Instrument[] = await instrumentsResponse.json()
      const datasetsData: Dataset[] = await datasetsResponse.json()

      setModels(modelsData)
      setInstruments(instrumentsData)
      setDatasets(datasetsData)
    } catch (err) {
      console.error(err)
      setFormError(err instanceof Error ? err.message : 'Erro ao carregar listas')
    } finally {
      setLoadingOptions(false)
    }
  }

  useEffect(() => {
    fetchTransfers()
  }, [])

  useEffect(() => {
    if (isModalOpen) {
      fetchOptions()
    }
  }, [isModalOpen])

  const hasDsAdjusted = (transfer: CalibrationTransfer) => Boolean(transfer.artifacts && 'F_ds' in transfer.artifacts)

  const resetForm = () => {
    setName('')
    setMasterModelId('')
    setMasterInstrumentId('')
    setSlaveInstrumentId('')
    setMasterDatasetId('')
    setSlaveDatasetId('')
    setMethod('ds')
    setFormError(null)
  }

  const handleCreateTransfer = async (event: React.FormEvent) => {
    event.preventDefault()

    if (!name.trim() || !masterModelId || !masterInstrumentId || !slaveInstrumentId || !masterDatasetId || !slaveDatasetId) {
      setFormError('Preencha todos os campos obrigatórios antes de salvar')
      return
    }

    const payload = {
      name: name.trim(),
      master_model_id: masterModelId,
      master_instrument_id: masterInstrumentId,
      slave_instrument_id: slaveInstrumentId,
      master_dataset_id: masterDatasetId,
      slave_dataset_id: slaveDatasetId,
      method,
    }

    setCreating(true)
    setFormError(null)

    try {
      const response = await fetch(`${API_BASE_URL}/calibration-transfers`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        throw new Error('Não foi possível criar a transferência')
      }

      resetForm()
      setIsModalOpen(false)
      fetchTransfers()
    } catch (err) {
      console.error(err)
      setFormError(err instanceof Error ? err.message : 'Erro ao criar transferência')
    } finally {
      setCreating(false)
    }
  }

  const handleFitDs = async (transferId: string | number) => {
    setFittingId(transferId)
    try {
      const response = await fetch(`${API_BASE_URL}/calibration-transfers/${transferId}/fit-ds`, {
        method: 'POST',
      })

      if (!response.ok) {
        throw new Error('Falha ao ajustar DS')
      }

      fetchTransfers()
    } catch (err) {
      console.error(err)
      setError(err instanceof Error ? err.message : 'Erro ao ajustar DS')
    } finally {
      setFittingId(null)
    }
  }

  const statusBadge = (transfer: CalibrationTransfer) => {
    const hasDs = hasDsAdjusted(transfer)
    const label = hasDs ? 'DS ajustado' : 'Sem DS ajustado'
    const classes = hasDs ? 'bg-emerald-50 text-emerald-700 border-emerald-200' : 'bg-amber-50 text-amber-700 border-amber-200'
    return <span className={`rounded-full border px-3 py-1 text-xs font-semibold ${classes}`}>{label}</span>
  }

  const methodLabel = (value?: string) => {
    if (!value) return 'N/D'
    return value.toUpperCase()
  }

  const canSubmit = useMemo(
    () => Boolean(name.trim() && masterModelId && masterInstrumentId && slaveInstrumentId && masterDatasetId && slaveDatasetId && method),
    [masterDatasetId, masterInstrumentId, masterModelId, method, name, slaveDatasetId, slaveInstrumentId],
  )

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-900">Transferência de Calibração</h1>
          <p className="text-slate-600">Gerencie ajustes de DS e registre novas transferências mestre → escravo.</p>
        </div>
        <button
          type="button"
          onClick={() => setIsModalOpen(true)}
          className="inline-flex items-center gap-2 rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700"
        >
          <span aria-hidden="true">＋</span> Nova transferência
        </button>
      </div>

      <div className="rounded-2xl border border-slate-200 bg-white shadow-sm">
        <div className="flex items-center justify-between border-b border-slate-100 px-4 py-3">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">Lista de transferências</h2>
            <p className="text-sm text-slate-600">Veja o status de DS e navegue até os detalhes.</p>
          </div>
          <button
            type="button"
            onClick={fetchTransfers}
            className="rounded-lg border border-slate-200 px-3 py-1.5 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
          >
            Atualizar
          </button>
        </div>

        {error ? <div className="m-4 rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{error}</div> : null}

        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-slate-200 text-sm">
            <thead className="bg-slate-50 text-left text-xs font-semibold uppercase tracking-wide text-slate-600">
              <tr>
                <th className="px-4 py-3">Nome</th>
                <th className="px-4 py-3">Método</th>
                <th className="px-4 py-3">Modelo mestre</th>
                <th className="px-4 py-3">Instrumentos</th>
                <th className="px-4 py-3">Status</th>
                <th className="px-4 py-3 text-right">Ações</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 bg-white">
              {loading ? (
                <tr>
                  <td className="px-4 py-4 text-center text-slate-500" colSpan={6}>
                    Carregando transferências...
                  </td>
                </tr>
              ) : transfers.length === 0 ? (
                <tr>
                  <td className="px-4 py-4 text-center text-slate-500" colSpan={6}>
                    Nenhuma transferência cadastrada.
                  </td>
                </tr>
              ) : (
                transfers.map((transfer) => (
                  <tr key={transfer.id} className="hover:bg-slate-50/60">
                    <td className="px-4 py-3 font-semibold text-slate-900">{transfer.name}</td>
                    <td className="px-4 py-3 text-slate-700">{methodLabel(transfer.method)}</td>
                    <td className="px-4 py-3 text-slate-700">{transfer.master_model_name ?? transfer.master_model_id ?? 'N/D'}</td>
                    <td className="px-4 py-3 text-slate-700">
                      {transfer.master_instrument_name ?? transfer.master_instrument_id ?? 'N/D'} →{' '}
                      {transfer.slave_instrument_name ?? transfer.slave_instrument_id ?? 'N/D'}
                    </td>
                    <td className="px-4 py-3">{statusBadge(transfer)}</td>
                    <td className="px-4 py-3">
                      <div className="flex items-center justify-end gap-2">
                        <button
                          type="button"
                          onClick={() => navigate(`/transfers/${transfer.id}`)}
                          className="rounded-lg border border-slate-200 px-3 py-1.5 text-xs font-semibold text-slate-700 transition hover:bg-slate-50"
                        >
                          Ver detalhes
                        </button>
                        <button
                          type="button"
                          disabled={fittingId === transfer.id}
                          onClick={() => handleFitDs(transfer.id)}
                          className="rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white shadow-sm transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-indigo-300"
                        >
                          {fittingId === transfer.id ? 'Ajustando...' : 'Ajustar DS'}
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {isModalOpen ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 p-4">
          <div className="w-full max-w-3xl rounded-2xl bg-white shadow-xl ring-1 ring-slate-100">
            <div className="flex items-start justify-between border-b border-slate-200 px-6 py-4">
              <div>
                <h3 className="text-lg font-semibold text-slate-900">Nova transferência</h3>
                <p className="text-sm text-slate-600">Defina modelo mestre, instrumentos e datasets padrão.</p>
              </div>
              <button
                type="button"
                onClick={() => {
                  resetForm()
                  setIsModalOpen(false)
                }}
                className="rounded-full p-2 text-slate-500 transition hover:bg-slate-100 hover:text-slate-700"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleCreateTransfer} className="space-y-4 px-6 py-4">
              {formError ? <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">{formError}</div> : null}

              <div className="grid gap-4 md:grid-cols-2">
                <label className="space-y-1 text-sm text-slate-700">
                  Nome
                  <input
                    type="text"
                    value={name}
                    onChange={(event) => setName(event.target.value)}
                    className="w-full rounded-lg border border-slate-200 px-3 py-2 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
                    placeholder="Ex.: Transferência DS"
                  />
                </label>

                <label className="space-y-1 text-sm text-slate-700">
                  Método
                  <select
                    value={method}
                    onChange={(event) => setMethod(event.target.value)}
                    className="w-full rounded-lg border border-slate-200 px-3 py-2 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
                  >
                    <option value="ds">DS</option>
                  </select>
                </label>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <label className="space-y-1 text-sm text-slate-700">
                  Modelo mestre
                  <select
                    value={masterModelId}
                    onChange={(event) => setMasterModelId(event.target.value)}
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
                  Dataset padrão (mestre)
                  <select
                    value={masterDatasetId}
                    onChange={(event) => setMasterDatasetId(event.target.value)}
                    className="w-full rounded-lg border border-slate-200 px-3 py-2 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
                  >
                    <option value="">Selecione um dataset</option>
                    {datasets.map((dataset) => (
                      <option key={dataset.id} value={dataset.id}>
                        {dataset.name}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <label className="space-y-1 text-sm text-slate-700">
                  Instrumento mestre
                  <select
                    value={masterInstrumentId}
                    onChange={(event) => setMasterInstrumentId(event.target.value)}
                    className="w-full rounded-lg border border-slate-200 px-3 py-2 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
                  >
                    <option value="">Selecione um instrumento</option>
                    {instruments.map((instrument) => (
                      <option key={instrument.id} value={instrument.id}>
                        {instrument.name}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="space-y-1 text-sm text-slate-700">
                  Instrumento escravo
                  <select
                    value={slaveInstrumentId}
                    onChange={(event) => setSlaveInstrumentId(event.target.value)}
                    className="w-full rounded-lg border border-slate-200 px-3 py-2 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
                  >
                    <option value="">Selecione um instrumento</option>
                    {instruments.map((instrument) => (
                      <option key={instrument.id} value={instrument.id}>
                        {instrument.name}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              <div className="grid gap-4 md:grid-cols-2">
                <label className="space-y-1 text-sm text-slate-700">
                  Dataset padrão (escravo)
                  <select
                    value={slaveDatasetId}
                    onChange={(event) => setSlaveDatasetId(event.target.value)}
                    className="w-full rounded-lg border border-slate-200 px-3 py-2 focus:border-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-100"
                  >
                    <option value="">Selecione um dataset</option>
                    {datasets.map((dataset) => (
                      <option key={dataset.id} value={dataset.id}>
                        {dataset.name}
                      </option>
                    ))}
                  </select>
                </label>
              </div>

              <div className="flex items-center justify-end gap-3 border-t border-slate-200 pt-4">
                <button
                  type="button"
                  onClick={() => {
                    resetForm()
                    setIsModalOpen(false)
                  }}
                  className="rounded-lg border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-50"
                >
                  Cancelar
                </button>
                <button
                  type="submit"
                  disabled={!canSubmit || creating}
                  className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-indigo-700 disabled:cursor-not-allowed disabled:bg-indigo-300"
                >
                  {creating ? 'Salvando...' : 'Criar transferência'}
                </button>
              </div>

              {loadingOptions ? (
                <p className="text-sm text-slate-500">Carregando listas de modelos, instrumentos e datasets...</p>
              ) : null}
            </form>
          </div>
        </div>
      ) : null}
    </div>
  )
}

export default Transfers
