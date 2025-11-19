export interface DatasetManifest {
  n_samples?: number | null
  n_features?: number | null
  target_cols?: string[] | string | null
}

export interface DatasetApiResponse {
  dataset_id?: number
  id?: number
  name: string
  description?: string | null
  version?: string | null
  instrument_id?: number | null
  instrument_name?: string | null
  n_samples?: number | null
  n_features?: number | null
  created_at?: string | null
  updated_at?: string | null
  manifest?: DatasetManifest
}

export interface Dataset {
  id: number
  name: string
  description?: string | null
  version?: string | null
  instrument_id?: number | null
  instrument_name?: string | null
  n_samples?: number | null
  n_features?: number | null
  created_at?: string | null
  updated_at?: string | null
  manifest?: DatasetManifest
}

export const normalizeDataset = (dataset: DatasetApiResponse): Dataset => {
  const id = dataset.dataset_id ?? dataset.id
  if (typeof id !== 'number') {
    throw new Error('Dataset response is missing the identifier field')
  }

  const manifestSamples = dataset.manifest?.n_samples ?? null
  const manifestFeatures = dataset.manifest?.n_features ?? null

  return {
    id,
    name: dataset.name,
    description: dataset.description ?? undefined,
    version: dataset.version ?? undefined,
    instrument_id: dataset.instrument_id ?? undefined,
    instrument_name: dataset.instrument_name ?? undefined,
    n_samples: dataset.n_samples ?? manifestSamples ?? undefined,
    n_features: dataset.n_features ?? manifestFeatures ?? undefined,
    created_at: dataset.created_at ?? undefined,
    updated_at: dataset.updated_at ?? undefined,
    manifest: dataset.manifest,
  }
}

export interface Instrument {
  id: number
  name: string
  code?: string
}
