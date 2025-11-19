import { Fragment, useEffect, useMemo, useState } from 'react'
import type { Dataset } from '../../types/datasets'
import type { Run } from '../../types/runs'

interface RunCreateModalProps {
  isOpen: boolean
  datasets: Dataset[]
  apiBaseUrl: string
  onClose: () => void
  onCreated: (run: Run) => void
}

type ModelProfile = 'NIR_REGRESSION' | 'PLS_ONLY' | 'ML_ONLY'

const profileModels: Record<ModelProfile, { name: string; params: Record<string, unknown> }> = {
  NIR_REGRESSION: { name: 'PLS_REG', params: { n_components: 8 } },
  PLS_ONLY: { name: 'PLS_REG', params: { n_components: 8 } },
  ML_ONLY: { name: 'RF_REG', params: {} },
}

const stepTitles = ['Dataset', 'Targets', 'Perfil de modelos', 'Configuração de CV']

const RunCreateModal = ({ isOpen, datasets, apiBaseUrl, onClose, onCreated }: RunCreateModalProps) => {
  const [currentStep, setCurrentStep] = useState(0)
  const [selectedDatasetId, setSelectedDatasetId] = useState<string>('')
  const [selectedTargets, setSelectedTargets] = useState<string[]>([])
  const [profile, setProfile] = useState<ModelProfile>('NIR_REGRESSION')
  const [nSplits, setNSplits] = useState<number>(5)
  const [cvMethod, setCvMethod] = useState('kfold')
  const [name, setName] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const selectedDataset = useMemo(
    () => datasets.find((dataset) => `${dataset.id}` === `${selectedDatasetId}`),
    [datasets, selectedDatasetId],
  )

  const targetOptions = useMemo(() => {
    if (!selectedDataset?.manifest?.target_cols) return []
    const { target_cols: manifestTargets } = selectedDataset.manifest
    if (Array.isArray(manifestTargets)) {
      return manifestTargets
    }
    return manifestTargets ? [manifestTargets] : []
  }, [selectedDataset])

  useEffect(() => {
    if (!isOpen) {
      setCurrentStep(0)
      setSelectedDatasetId('')
      setSelectedTargets([])
      setProfile('NIR_REGRESSION')
      setNSplits(5)
      setCvMethod('kfold')
      setName('')
      setError(null)
    }
  }, [isOpen])

  const toggleTarget = (value: string) => {
    setSelectedTargets((prev) =>
      prev.includes(value) ? prev.filter((item) => item !== value) : [...prev, value],
    )
  }

  const canGoNext = () => {
    if (currentStep === 0) return Boolean(selectedDatasetId)
    if (currentStep === 1) return selectedTargets.length > 0
    return true
  }

  const handleSubmit = async () => {
    if (!selectedDatasetId || !selectedTargets.length) return
    setSubmitting(true)
    setError(null)

    const model = profileModels[profile]
    const payload = {
      dataset_id: selectedDatasetId,
      name: name || `Calibração ${selectedDataset?.name ?? ''}`.trim(),
      target_cols: selectedTargets,
      config: {
        data: {},
        preprocess: {},
        cv: { method: cvMethod, n_splits: nSplits },
        models: [model],
      },
    }

    try {
      const response = await fetch(`${apiBaseUrl}/runs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })

      if (!response.ok) {
        throw new Error('Não foi possível criar a calibração')
      }

      const created: Run = await response.json()
      onCreated(created)
      onClose()
    } catch (err) {
      console.error(err)
      setError(err instanceof Error ? err.message : 'Erro ao criar calibração')
    } finally {
      setSubmitting(false)
    }
  }

  if (!isOpen) return null

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 px-4">
      <div className="w-full max-w-2xl rounded-2xl bg-white shadow-xl">
        <div className="flex items-center justify-between border-b border-slate-200 px-6 py-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-wide text-indigo-600">Nova calibração</p>
            <h3 className="text-xl font-bold text-slate-900">Configurar execução</h3>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="rounded-lg px-2 py-1 text-sm font-semibold text-slate-500 transition hover:bg-slate-100 hover:text-slate-700"
          >
            Fechar
          </button>
        </div>

        <div className="border-b border-slate-200 px-6 py-3">
          <div className="flex items-center gap-3 text-sm font-semibold text-slate-700">
            {stepTitles.map((title, index) => (
              <Fragment key={title}>
                <div className="flex items-center gap-2">
                  <span
                    className={`flex h-7 w-7 items-center justify-center rounded-full border text-xs ${
                      index === currentStep
                        ? 'border-indigo-500 bg-indigo-50 text-indigo-600'
                        : index < currentStep
                          ? 'border-emerald-500 bg-emerald-50 text-emerald-600'
                          : 'border-slate-200 bg-slate-50 text-slate-500'
                    }`}
                  >
                    {index + 1}
                  </span>
                  <span>{title}</span>
                </div>
                {index < stepTitles.length - 1 ? <span className="text-slate-300">→</span> : null}
              </Fragment>
            ))}
          </div>
        </div>

        <div className="px-6 py-5">
          {currentStep === 0 ? (
            <div className="space-y-3">
              <label className="block text-sm font-medium text-slate-700">Selecione um dataset</label>
              <select
                value={selectedDatasetId}
                onChange={(e) => setSelectedDatasetId(e.target.value)}
                className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-800 shadow-sm focus:border-indigo-500 focus:outline-none"
              >
                <option value="">Escolha um dataset</option>
                {datasets.map((dataset) => (
                  <option key={dataset.id} value={dataset.id}>
                    {dataset.name}
                  </option>
                ))}
              </select>
              <p className="text-xs text-slate-500">Targets serão carregados a partir do manifest do dataset.</p>
            </div>
          ) : null}

          {currentStep === 1 ? (
            <div className="space-y-3">
              <p className="text-sm font-medium text-slate-700">Escolha os targets da calibração</p>
              {!targetOptions.length ? (
                <p className="rounded-lg border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-700">
                  Nenhum target encontrado no manifest deste dataset.
                </p>
              ) : (
                <div className="grid grid-cols-1 gap-2 sm:grid-cols-2">
                  {targetOptions.map((target) => (
                    <label
                      key={target}
                      className="flex cursor-pointer items-center gap-2 rounded-lg border border-slate-200 px-3 py-2 text-sm font-medium text-slate-700 shadow-sm transition hover:border-indigo-200 hover:bg-indigo-50"
                    >
                      <input
                        type="checkbox"
                        className="h-4 w-4 text-indigo-600"
                        checked={selectedTargets.includes(target)}
                        onChange={() => toggleTarget(target)}
                      />
                      {target}
                    </label>
                  ))}
                </div>
              )}
            </div>
          ) : null}

          {currentStep === 2 ? (
            <div className="space-y-3">
              <label className="block text-sm font-medium text-slate-700">Perfil de modelos</label>
              <select
                value={profile}
                onChange={(e) => setProfile(e.target.value as ModelProfile)}
                className="w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-800 shadow-sm focus:border-indigo-500 focus:outline-none"
              >
                <option value="NIR_REGRESSION">NIR_REGRESSION</option>
                <option value="PLS_ONLY">PLS_ONLY</option>
                <option value="ML_ONLY">ML_ONLY</option>
              </select>
              <p className="text-xs text-slate-500">Este perfil controla a lista de modelos enviada ao backend.</p>
            </div>
          ) : null}

          {currentStep === 3 ? (
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-slate-700">Nome da calibração</label>
                <input
                  type="text"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  placeholder="Ex: Calibração densidade básica"
                  className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-800 shadow-sm focus:border-indigo-500 focus:outline-none"
                />
              </div>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                <div>
                  <label className="block text-sm font-medium text-slate-700">n_splits</label>
                  <input
                    type="number"
                    min={2}
                    value={nSplits}
                    onChange={(e) => setNSplits(Number(e.target.value))}
                    className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-800 shadow-sm focus:border-indigo-500 focus:outline-none"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-slate-700">cv_method</label>
                  <select
                    value={cvMethod}
                    onChange={(e) => setCvMethod(e.target.value)}
                    className="mt-1 w-full rounded-lg border border-slate-200 px-3 py-2 text-sm text-slate-800 shadow-sm focus:border-indigo-500 focus:outline-none"
                  >
                    <option value="kfold">kfold</option>
                  </select>
                </div>
              </div>
            </div>
          ) : null}

          {error ? (
            <div className="mt-4 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-700">{error}</div>
          ) : null}
        </div>

        <div className="flex items-center justify-between rounded-b-2xl border-t border-slate-200 bg-slate-50 px-6 py-4">
          <div className="text-sm text-slate-500">
            Passo {currentStep + 1} de {stepTitles.length}
          </div>
          <div className="flex items-center gap-3">
            {currentStep > 0 ? (
              <button
                type="button"
                onClick={() => setCurrentStep((step) => Math.max(0, step - 1))}
                className="rounded-lg border border-slate-200 px-4 py-2 text-sm font-semibold text-slate-700 transition hover:bg-slate-100"
                disabled={submitting}
              >
                Voltar
              </button>
            ) : null}

            {currentStep < stepTitles.length - 1 ? (
              <button
                type="button"
                onClick={() => setCurrentStep((step) => step + 1)}
                disabled={!canGoNext()}
                className={`rounded-lg px-4 py-2 text-sm font-semibold text-white shadow-sm transition ${
                  canGoNext()
                    ? 'bg-indigo-600 hover:bg-indigo-700'
                    : 'cursor-not-allowed bg-slate-300 text-slate-600'
                }`}
              >
                Próximo
              </button>
            ) : (
              <button
                type="button"
                onClick={handleSubmit}
                disabled={submitting || !selectedDatasetId || !selectedTargets.length}
                className="inline-flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:bg-slate-300"
              >
                {submitting ? 'Enviando...' : 'Criar calibração'}
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default RunCreateModal
