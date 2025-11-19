export interface ModelMetrics {
  [key: string]: number
}

export interface ModelSummary {
  id: number | string
  name: string
  target?: string
  target_name?: string
  dataset_id?: number | string | null
  instrument_id?: number | string | null
  metrics?: ModelMetrics
  status?: string
}

export interface ModelSpec {
  class?: string
  class_name?: string
  model_class?: string
  n_components?: number
  preprocess?: string[] | string | Record<string, unknown>
  [key: string]: unknown
}

export interface ModelEvaluationRecord {
  dataset_id?: number | string
  metrics?: ModelMetrics
  evaluated_at?: string
  created_at?: string
}

export interface ModelDetail extends ModelSummary {
  model_spec?: ModelSpec
  artifacts?: Record<string, unknown>
  eval_history?: ModelEvaluationRecord[]
}
