import { writable } from 'svelte/store'
import type { TrainingStore, TrainingConfig, StepEvent, EpochEndEvent } from '$lib/types'

const HISTORY_LEN = 140
const LOG_MAX = 20

function defaultConfig(): TrainingStore['config'] {
  return {
    model_name: '',
    phase: 0,
    device: '',
    amp: false,
    batch_size: 0,
    lr: 0,
    total_epochs: 0,
    warmup_steps: 0,
    datasets: [],
    speakers: 0,
    total_samples: 0,
    val_samples: 0
  }
}

function initial(): TrainingStore {
  return {
    status: 'idle',
    run_id: null,
    config: defaultConfig(),
    epoch: 0,
    step: 0,
    steps_per_epoch: 0,
    loss: 0,
    loss_ma5: 0,
    speed_bps: 0,
    grad_norm: 0,
    lr: 0,
    eta_epoch_s: 0,
    eta_global_s: 0,
    started_at: 0,
    last_update_at: 0,
    gpu: { util_pct: 0, temp_c: 0, vram_used_gb: 0, vram_total_gb: 0, power_w: 0 },
    loss_history: [],
    loss_ma_history: [],
    gpu_use_history: [],
    gpu_temp_history: [],
    gpu_vram_history: [],
    loss_epoch_train: [],
    loss_epoch_val: [],
    acc_epoch_train: [],
    acc_epoch_val: [],
    tone_epoch: { T1: [], T2: [], T3: [], T4: [], T5: [] },
    best_acc: 0,
    best_acc_epoch: 0,
    best_loss: Infinity,
    best_loss_epoch: 0,
    pcc_epoch_train: [],
    pcc_epoch_val: [],
    mse_epoch: { score_global: [], score_ton: [], score_initiale: [], score_finale: [] },
    best_pcc: 0,
    best_pcc_epoch: 0,
    score_dist_mean: [],
    score_dist_std: [],
    logs: []
  }
}

function pushRolling(arr: number[], val: number): number[] {
  const next = [...arr, val]
  return next.length > HISTORY_LEN ? next.slice(-HISTORY_LEN) : next
}

export const training = writable<TrainingStore>(initial())

export function applyConfigEvent(config: TrainingConfig, run_id: string) {
  training.update(s => ({
    ...s,
    config,
    run_id,
    started_at: s.started_at || Date.now()
  }))
}

export function applyStepEvent(ev: StepEvent) {
  training.update(s => ({
    ...s,
    status: 'running',
    run_id: ev.run_id,
    epoch: ev.epoch,
    step: ev.step,
    steps_per_epoch: ev.steps_per_epoch,
    loss: ev.loss,
    loss_ma5: ev.loss_ma5,
    speed_bps: ev.speed_bps,
    grad_norm: ev.grad_norm,
    lr: ev.lr,
    eta_epoch_s: ev.eta_epoch_s,
    eta_global_s: ev.eta_global_s,
    gpu: ev.gpu,
    last_update_at: Date.now(),
    loss_history: pushRolling(s.loss_history, ev.loss),
    loss_ma_history: pushRolling(s.loss_ma_history, ev.loss_ma5),
    gpu_use_history: pushRolling(s.gpu_use_history, ev.gpu.util_pct),
    gpu_temp_history: pushRolling(s.gpu_temp_history, ev.gpu.temp_c),
    gpu_vram_history: pushRolling(s.gpu_vram_history, ev.gpu.vram_used_gb),
    logs: [
      ...s.logs.slice(-(LOG_MAX - 1)),
      {
        epoch: ev.epoch,
        step: ev.step,
        steps_per_epoch: ev.steps_per_epoch,
        loss: ev.loss.toFixed(4),
        eta: fmtTime(ev.eta_epoch_s)
      }
    ]
  }))
}

export function applyEpochEndEvent(ev: EpochEndEvent) {
  training.update(s => {
    const newBestAcc = ev.acc_val > s.best_acc
    const newBestLoss = ev.loss_val < s.best_loss

    // Phase 3 regression fields
    const hasPcc = ev.pcc_val !== undefined
    const newBestPcc = hasPcc && ev.pcc_val! > s.best_pcc

    return {
      ...s,
      loss_epoch_train: [...s.loss_epoch_train, ev.loss_train],
      loss_epoch_val: [...s.loss_epoch_val, ev.loss_val],
      acc_epoch_train: [...s.acc_epoch_train, ev.acc_train],
      acc_epoch_val: [...s.acc_epoch_val, ev.acc_val],
      tone_epoch: {
        T1: [...s.tone_epoch.T1, ev.tone_accs.T1],
        T2: [...s.tone_epoch.T2, ev.tone_accs.T2],
        T3: [...s.tone_epoch.T3, ev.tone_accs.T3],
        T4: [...s.tone_epoch.T4, ev.tone_accs.T4],
        T5: [...s.tone_epoch.T5, ev.tone_accs.T5]
      },
      best_acc: newBestAcc ? ev.acc_val : s.best_acc,
      best_acc_epoch: newBestAcc ? ev.epoch : s.best_acc_epoch,
      best_loss: newBestLoss ? ev.loss_val : s.best_loss,
      best_loss_epoch: newBestLoss ? ev.epoch : s.best_loss_epoch,
      // Phase 3
      pcc_epoch_train: ev.pcc_train !== undefined ? [...s.pcc_epoch_train, ev.pcc_train] : s.pcc_epoch_train,
      pcc_epoch_val: hasPcc ? [...s.pcc_epoch_val, ev.pcc_val!] : s.pcc_epoch_val,
      mse_epoch: ev.mse_per_head ? {
        score_global: [...s.mse_epoch.score_global, ev.mse_per_head.score_global],
        score_ton: [...s.mse_epoch.score_ton, ev.mse_per_head.score_ton],
        score_initiale: [...s.mse_epoch.score_initiale, ev.mse_per_head.score_initiale],
        score_finale: [...s.mse_epoch.score_finale, ev.mse_per_head.score_finale]
      } : s.mse_epoch,
      best_pcc: newBestPcc ? ev.pcc_val! : s.best_pcc,
      best_pcc_epoch: newBestPcc ? ev.epoch : s.best_pcc_epoch,
      score_dist_mean: ev.score_distribution ? [...s.score_dist_mean, ev.score_distribution.mean] : s.score_dist_mean,
      score_dist_std: ev.score_distribution ? [...s.score_dist_std, ev.score_distribution.std] : s.score_dist_std
    }
  })
}

export function resetTraining() {
  training.set(initial())
}

function fmtTime(s: number): string {
  if (s < 60) return `${Math.round(s)}s`
  if (s < 3600) return `~${Math.round(s / 60)}m`
  const h = Math.floor(s / 3600)
  const m = Math.round((s % 3600) / 60)
  return `${h}h${m}m`
}
