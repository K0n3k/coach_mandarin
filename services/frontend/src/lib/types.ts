// Types canoniques — basés sur le schéma §4 de copilot-instructions.md

export interface PhonemeScore {
  attendu: string
  detecte: string
  score: number
  type_confusion: string | null
  start_ms: number
  end_ms: number
}

export interface SyllableScore {
  hanzi: string
  pinyin: string
  traduction: string
  ton_attendu: number
  ton_detecte: number
  start_s: number
  end_s: number
  scores: {
    global: number
    ton: number
    initiale: number
    finale: number
  }
  f0_attendu: number[]
  f0_produit: number[]
  erreur: string | null
  phonemes: PhonemeScore[]
}

export interface ScoringResult {
  session_id: string
  mode: 'guidé' | 'libre'
  enonce: string
  traduction: string
  duration_total_s: number
  score_global: number
  score_fluidite: number
  score_rythme: number
  debit_mots_par_min: number
  syllabes: SyllableScore[]
}

export interface TtsSegment {
  segment_id: string
  lang: string
  audio_b64: string
}

// --- Training types (§7 WS events) ---

export interface GpuStats {
  util_pct: number
  temp_c: number
  vram_used_gb: number
  vram_total_gb: number
  power_w: number
}

export interface StepEvent {
  type: 'step'
  run_id: string
  epoch: number
  step: number
  steps_per_epoch: number
  loss: number
  loss_ma5: number
  speed_bps: number
  grad_norm: number
  lr: number
  eta_epoch_s: number
  eta_global_s: number
  gpu: GpuStats
}

export interface ToneAccs {
  T1: number
  T2: number
  T3: number
  T4: number
  T5: number
}

export interface EpochEndEvent {
  type: 'epoch_end'
  run_id: string
  epoch: number
  loss_train: number
  loss_val: number
  acc_train: number
  acc_val: number
  tone_accs: ToneAccs
  is_best: boolean
}

export interface CheckpointEvent {
  type: 'checkpoint'
  run_id: string
  epoch: number
  path: string
  val_acc: number
  is_best: boolean
}

export type TrainingEvent = StepEvent | EpochEndEvent | CheckpointEvent | ConfigEvent

export interface LogEntry {
  epoch: number
  step: number
  steps_per_epoch: number
  loss: string
  eta: string
}

export interface DatasetInfo {
  name: string
  active: boolean
  phases: number[]
  samples?: number
}

export interface TrainingConfig {
  model_name: string
  phase: number
  device: string
  amp: boolean
  batch_size: number
  lr: number
  total_epochs: number
  warmup_steps: number
  datasets: DatasetInfo[]
  speakers: number
  total_samples: number
  val_samples: number
}

export interface ConfigEvent {
  type: 'config'
  run_id: string
  config: TrainingConfig
}

export interface TrainingStore {
  status: 'idle' | 'connecting' | 'running' | 'done' | 'failed' | 'stopped'
  run_id: string | null
  config: TrainingConfig

  // Current step state
  epoch: number
  step: number
  steps_per_epoch: number
  loss: number
  loss_ma5: number
  speed_bps: number
  grad_norm: number
  lr: number
  eta_epoch_s: number
  eta_global_s: number
  started_at: number       // timestamp ms
  last_update_at: number   // timestamp ms

  // GPU
  gpu: GpuStats

  // History arrays (rolling window)
  loss_history: number[]
  loss_ma_history: number[]
  gpu_use_history: number[]
  gpu_temp_history: number[]
  gpu_vram_history: number[]

  // Epoch-level metrics
  loss_epoch_train: number[]
  loss_epoch_val: number[]
  acc_epoch_train: number[]
  acc_epoch_val: number[]
  tone_epoch: { T1: number[]; T2: number[]; T3: number[]; T4: number[]; T5: number[] }

  // Best
  best_acc: number
  best_acc_epoch: number
  best_loss: number
  best_loss_epoch: number

  // Logs
  logs: LogEntry[]
}
