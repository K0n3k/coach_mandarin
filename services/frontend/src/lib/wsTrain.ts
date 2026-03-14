import { applyStepEvent, applyEpochEndEvent, training } from '$lib/stores/training'
import type { StepEvent, EpochEndEvent } from '$lib/types'

// Simulates a training run producing step events every ~800ms
// and epoch_end events at the right step boundaries.
// Matches the data patterns from training_monitor_v3.html.

class SimTrainWs {
  private timer: ReturnType<typeof setInterval> | null = null
  private tick = 0
  private step = 32000
  private epoch = 4
  private readonly stepsPerEpoch = 34423
  private readonly runId = 'sim-train-001'

  // Seed epoch-level history (epochs 1-3 already done)
  private seeded = false

  connect(): void {
    if (this.timer) return

    if (!this.seeded) {
      this.seedHistory()
      this.seeded = true
    }

    training.update(s => ({
      ...s,
      status: 'running',
      run_id: this.runId,
      started_at: Date.now() - 300_000 // "started 5 min ago"
    }))

    this.timer = setInterval(() => this.emitStep(), 800)
  }

  disconnect(): void {
    if (this.timer) {
      clearInterval(this.timer)
      this.timer = null
    }
  }

  private seedHistory() {
    // Build all seed data in plain arrays first, then apply in ONE store update
    const lossHistory: number[] = []
    const lossMaHistory: number[] = []
    const gpuUseHistory: number[] = []
    const gpuTempHistory: number[] = []
    const gpuVramHistory: number[] = []

    for (let i = 0; i < 100; i++) {
      const base = 1.68 * Math.exp(-i / 100 * 0.22)
      const noise = (Math.random() - 0.5) * 0.12
      const loss = base + noise
      const ma = i < 5 ? loss : loss * 0.2 + (base + 0.02) * 0.8
      lossHistory.push(loss)
      lossMaHistory.push(ma)
    }

    for (let i = 0; i < 60; i++) {
      gpuUseHistory.push(92 + Math.random() * 4)
      gpuTempHistory.push(70 + Math.random() * 3)
      gpuVramHistory.push(6.7 + Math.random() * 0.2)
    }

    // Epoch 1-3 results (pre-seed so charts show evolution on load)
    const epochSeeds = [
      { loss_tr: 1.615, loss_va: 1.602, acc_tr: 33.1, acc_va: 31.8, tones: { T1: 18.2, T2: 12.5, T3: 0.0, T4: 68.1, T5: 0.0 } },
      { loss_tr: 1.542, loss_va: 1.531, acc_tr: 38.6, acc_va: 36.9, tones: { T1: 24.1, T2: 18.3, T3: 1.8, T4: 73.5, T5: 0.6 } },
      { loss_tr: 1.478, loss_va: 1.469, acc_tr: 44.2, acc_va: 42.1, tones: { T1: 29.7, T2: 23.1, T3: 5.2, T4: 77.8, T5: 1.9 } }
    ]

    const lossEpTr: number[] = []
    const lossEpVa: number[] = []
    const accEpTr: number[] = []
    const accEpVa: number[] = []
    const toneEp = { T1: [] as number[], T2: [] as number[], T3: [] as number[], T4: [] as number[], T5: [] as number[] }

    for (const ep of epochSeeds) {
      lossEpTr.push(ep.loss_tr)
      lossEpVa.push(ep.loss_va)
      accEpTr.push(ep.acc_tr)
      accEpVa.push(ep.acc_va)
      toneEp.T1.push(ep.tones.T1)
      toneEp.T2.push(ep.tones.T2)
      toneEp.T3.push(ep.tones.T3)
      toneEp.T4.push(ep.tones.T4)
      toneEp.T5.push(ep.tones.T5)
    }

    const lastEp = epochSeeds[epochSeeds.length - 1]

    // Single store update for ALL seed data
    training.update(s => ({
      ...s,
      loss_history: lossHistory,
      loss_ma_history: lossMaHistory,
      gpu_use_history: gpuUseHistory,
      gpu_temp_history: gpuTempHistory,
      gpu_vram_history: gpuVramHistory,
      loss_epoch_train: lossEpTr,
      loss_epoch_val: lossEpVa,
      acc_epoch_train: accEpTr,
      acc_epoch_val: accEpVa,
      tone_epoch: toneEp,
      best_acc: lastEp.acc_va,
      best_acc_epoch: 3,
      best_loss: lastEp.loss_va,
      best_loss_epoch: 3
    }))
  }

  private emitStep() {
    this.tick++
    this.step += 100

    // Epoch boundary
    if (this.step >= this.stepsPerEpoch) {
      this.emitEpochEnd()
      this.epoch++
      this.step = 100
    }

    const epPct = this.step / this.stepsPerEpoch
    const totalStepsDone = (this.epoch - 1) * this.stepsPerEpoch + this.step
    const totalSteps = 20 * this.stepsPerEpoch
    const etaEpochS = Math.max(0, (1 - epPct) * 3600)
    const etaGlobalS = Math.max(0, ((totalSteps - totalStepsDone) / 9.5))

    const loss = +(1.57 + Math.random() * 0.10).toFixed(4)
    const gpuUtil = Math.min(100, Math.max(85, 92 + Math.sin(this.tick * 0.3) * 4 + (Math.random() - 0.5) * 3))
    const gpuTemp = Math.min(85, Math.max(68, 71 + Math.sin(this.tick * 0.15) * 2 + (Math.random() - 0.5) * 1.5))
    const gpuVram = Math.min(8, Math.max(6.5, 6.8 + (Math.random() - 0.5) * 0.15))
    const power = Math.round(170 + gpuUtil * 1.2)

    const ev: StepEvent = {
      type: 'step',
      run_id: this.runId,
      epoch: this.epoch,
      step: this.step,
      steps_per_epoch: this.stepsPerEpoch,
      loss,
      loss_ma5: +(loss * 0.2 + 1.58 * 0.8).toFixed(4),
      speed_bps: +(9.2 + Math.random() * 0.6).toFixed(1),
      grad_norm: +(0.6 + Math.random() * 0.5).toFixed(3),
      lr: +(5e-4 * (1 - totalStepsDone / totalSteps * 0.3)).toExponential(2),
      eta_epoch_s: Math.round(etaEpochS),
      eta_global_s: Math.round(etaGlobalS),
      gpu: {
        util_pct: Math.round(gpuUtil),
        temp_c: Math.round(gpuTemp),
        vram_used_gb: +gpuVram.toFixed(1),
        vram_total_gb: 8.0,
        power_w: power
      }
    }

    applyStepEvent(ev)
  }

  private emitEpochEnd() {
    const baseLossTr = 1.615 - this.epoch * 0.05 + Math.random() * 0.02
    const baseLossVa = baseLossTr + (Math.random() - 0.3) * 0.03
    const baseAccTr = 33.1 + this.epoch * 5 + Math.random() * 2
    const baseAccVa = baseAccTr - 2 + Math.random() * 1.5

    const ev: EpochEndEvent = {
      type: 'epoch_end',
      run_id: this.runId,
      epoch: this.epoch,
      loss_train: +baseLossTr.toFixed(3),
      loss_val: +baseLossVa.toFixed(3),
      acc_train: +baseAccTr.toFixed(1),
      acc_val: +baseAccVa.toFixed(1),
      tone_accs: {
        T1: +(18 + this.epoch * 3 + Math.random() * 4).toFixed(1),
        T2: +(12 + this.epoch * 3 + Math.random() * 3).toFixed(1),
        T3: +(Math.max(0, -2 + this.epoch * 1.5 + Math.random() * 2)).toFixed(1),
        T4: +(68 + this.epoch * 3 + Math.random() * 3).toFixed(1),
        T5: +(Math.max(0, -1 + this.epoch * 0.8 + Math.random() * 1)).toFixed(1)
      },
      is_best: baseAccVa > 35
    }

    applyEpochEndEvent(ev)
  }
}

export const wsTrainClient = new SimTrainWs()
