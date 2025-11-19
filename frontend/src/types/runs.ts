export type RunStatus = 'running' | 'completed' | 'failed' | string

export interface RunConfig {
  data?: Record<string, unknown>
  preprocess?: Record<string, unknown>
  cv?: {
    method?: string
    n_splits?: number
    [key: string]: unknown
  }
  models?: Array<{
    name: string
    params?: Record<string, unknown>
  }>
  [key: string]: unknown
}

export interface Run {
  id: number | string
  name: string
  dataset_id: number | string
  status: RunStatus
  created_at?: string
  artifacts?: unknown
  target_cols?: string[]
  config?: RunConfig
}

export interface RunMetricsResponse {
  [key: string]: unknown
}

export interface RunPlotsResponse {
  plots?: string[]
  [key: string]: unknown
}

export interface PlsLvCurvePoint {
  n_components: number
  rmsecv: number
}
