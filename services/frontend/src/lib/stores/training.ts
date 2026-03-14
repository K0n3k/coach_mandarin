import { writable } from 'svelte/store'
import type { TrainingStore, StepEvent, EpochEndEvent } from '$lib/types'

const HISTORY_LEN = 140
const LOG_MAX = 20

function defaultConfig(): TrainingStore['config'] {
  return {
    model_name: 'mel-cnn-bigru',
    phase: 2,
    device: 'cuda:0',
    amp: true,
    batch_size: 64,
    lr: 5e-4,
    total_epochs: 20,
    warmup_steps: 20653,
    datasets: [
      { name: 'TTS-ph1', active: true, phases: [1, 3] },
      { name: 'AISHELL-1', active: true, phases: [2, 3] },
      { name: 'THCHS-30', active: true, phases: [2] },
      { name: 'CV zh-CN', active: true, phases: [2] },
      { name: 'iCALL', active: false, phases: [3] },
      { name: 'LATIC', active: false, phases: [3] }
    ],
    speakers: 460,
    total_samples: '2.32M',
    val_samples: '118k'
  }
}

function initial(): TrainingStore {
  return {
    status: 'idle',
    run_id: null,
    config: defaultConfig(),
    epoch: 0,
    step: 0,
    steps_per_epoch: 34423,
    loss: 0,
    loss_ma5: 0,
    speed_bps: 0,
    grad_norm: 0,
    lr: 0,
    eta_epoch_s: 0,
    eta_global_s: 0,
    started_at: 0,
    last_update_at: 0,
    gpu: { util_pct: 0, temp_c: 0, vram_used_gb: 0, vram_total_gb: 8, power_w: 0 },
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
    logs: []
  }
}

function pushRolling(arr: number[], val: number): number[] {
  const next = [...arr, val]
  return next.length > HISTORY_LEN ? next.slice(-HISTORY_LEN) : next
}

export const training = writable<TrainingStore>(initial())

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
      best_loss_epoch: newBestLoss ? ev.epoch : s.best_loss_epoch
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
