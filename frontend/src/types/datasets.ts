export interface DatasetManifest {
  n_samples?: number
  n_features?: number
  target_cols?: string[] | string
}

export interface Dataset {
  id: string | number
  name: string
  instrument_id?: string | number
  instrument_name?: string
  created_at?: string
  manifest?: DatasetManifest
}

export interface Instrument {
  id: string | number
  name: string
}
