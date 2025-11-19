export interface CalibrationTransferArtifacts {
  F_ds?: unknown
  base_dir?: string
  shape?: number[] | [number, number]
  [key: string]: unknown
}

export interface CalibrationTransfer {
  id: number | string
  name: string
  master_model_id?: number | string | null
  master_model_name?: string
  master_instrument_id?: number | string | null
  master_instrument_name?: string
  slave_instrument_id?: number | string | null
  slave_instrument_name?: string
  master_dataset_id?: number | string | null
  master_dataset_name?: string
  slave_dataset_id?: number | string | null
  slave_dataset_name?: string
  method?: string
  artifacts?: CalibrationTransferArtifacts
}
