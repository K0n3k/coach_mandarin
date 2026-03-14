<script lang="ts">
  import { onMount } from 'svelte'
  import { training } from '$lib/stores/training'
  import type { TrainingStore } from '$lib/types'

  const TONES = [
    { tag: 'T1', name: 'Haut plat', color: '#e05555' },
    { tag: 'T2', name: 'Montant', color: '#f0a030' },
    { tag: 'T3', name: 'Desc-mont', color: '#3ecf7a' },
    { tag: 'T4', name: 'Descendant', color: '#5b8dee' },
    { tag: 'T5', name: 'Neutre', color: '#9b6dff' }
  ] as const

  const MSE_HEADS = [
    { key: 'score_global', label: 'Global', color: '#5b8dee' },
    { key: 'score_ton', label: 'Ton', color: '#f0a030' },
    { key: 'score_initiale', label: 'Initiale', color: '#3ecf7a' },
    { key: 'score_finale', label: 'Finale', color: '#9b6dff' }
  ] as const

  type ToneKey = 'T1' | 'T2' | 'T3' | 'T4' | 'T5'
  type MseKey = 'score_global' | 'score_ton' | 'score_initiale' | 'score_finale'

  // Canvas refs — 4 base + 2 phase 3
  let cLoss: HTMLCanvasElement
  let cLossEp: HTMLCanvasElement
  let cAcc: HTMLCanvasElement
  let cTone: HTMLCanvasElement
  let cPcc: HTMLCanvasElement
  let cMse: HTMLCanvasElement

  // Log auto-scroll ref
  let logEl: HTMLDivElement

  // Store snapshot — updated by manual subscription, not Svelte reactivity
  let s: TrainingStore

  // Derived values computed in the subscription callback, not via $:
  let hasConfig = false
  let epochPct = 0
  let totalSteps = 0
  let globalStepsDone = 0
  let globalPct = 0
  let overfitLabel = 'ok'
  let overfitClass = 'ok'
  let lastToneAccs = TONES.map(t => ({ ...t, val: 0 }))
  let startedAgo = '—'
  let updatedAgo = '—'
  let epochTimeMin = 0
  let gpuVramPct = 0
  let gpuTempPct = 0
  let gpuPowerPct = 0
  let gpuTempColor = '#3ecf7a'
  let logCount = 0
  let samplesSummary = ''
  let samplesDetail = ''
  let isPhase3 = false
  let lastStd = 0
  let stdBadgeColor = '#3ecf7a'
  let stdBadgeLabel = '—'

  function fmtTime(sec: number): string {
    if (sec < 60) return `${Math.round(sec)} s`
    if (sec < 3600) return `${Math.round(sec / 60)} min`
    const h = Math.floor(sec / 3600)
    const m = Math.round((sec % 3600) / 60)
    return `${h} h ${m} min`
  }

  function fmtAgo(sec: number): string {
    if (sec < 60) return `il y a ${Math.round(sec)} s`
    return `il y a ${Math.round(sec / 60)} min`
  }

  function fmtLr(v: number): string {
    if (v === 0) return '0'
    return v.toExponential(2)
  }

  function fmtCount(n: number): string {
    if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`
    if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
    return String(n)
  }

  function recompute() {
    hasConfig = s.config.model_name !== ''
    epochPct = s.steps_per_epoch > 0 ? (s.step / s.steps_per_epoch) * 100 : 0
    totalSteps = s.config.total_epochs * s.steps_per_epoch
    globalStepsDone = (s.epoch - 1) * s.steps_per_epoch + s.step
    globalPct = totalSteps > 0 ? (globalStepsDone / totalSteps) * 100 : 0
    epochTimeMin = s.steps_per_epoch > 0 && s.speed_bps > 0
      ? Math.round(s.steps_per_epoch / s.speed_bps / 60) : 0

    const overfitDelta = s.loss_epoch_train.length > 0
      ? Math.abs(s.loss_epoch_train[s.loss_epoch_train.length - 1] - s.loss_epoch_val[s.loss_epoch_val.length - 1])
      : 0
    overfitLabel = overfitDelta < 0.05 ? 'ok' : overfitDelta < 0.15 ? 'léger' : 'attention'
    overfitClass = overfitDelta < 0.05 ? 'ok' : overfitDelta < 0.15 ? 'warn' : 'err'

    lastToneAccs = s.tone_epoch.T1.length > 0
      ? TONES.map(t => ({ ...t, val: s.tone_epoch[t.tag as ToneKey][s.tone_epoch[t.tag as ToneKey].length - 1] }))
      : TONES.map(t => ({ ...t, val: 0 }))

    startedAgo = s.started_at > 0 ? fmtAgo(Math.round((Date.now() - s.started_at) / 1000)) : '—'
    updatedAgo = s.last_update_at > 0 ? fmtAgo(Math.round((Date.now() - s.last_update_at) / 1000)) : '—'

    gpuVramPct = s.gpu.vram_total_gb > 0 ? (s.gpu.vram_used_gb / s.gpu.vram_total_gb) * 100 : 0
    gpuTempPct = Math.min(100, Math.max(0, ((s.gpu.temp_c - 30) / 70) * 100))
    gpuPowerPct = Math.min(100, (s.gpu.power_w / 290) * 100)
    gpuTempColor = s.gpu.temp_c < 70 ? '#3ecf7a' : s.gpu.temp_c < 80 ? '#f0a030' : '#e05555'

    // Samples summary
    if (hasConfig) {
      samplesSummary = fmtCount(s.config.total_samples)
      samplesDetail = `${fmtCount(s.config.val_samples)} val`
      const activeDs = s.config.datasets.filter(d => d.active && d.samples)
      if (activeDs.length > 0) {
        samplesDetail = activeDs.map(d => `${d.name} ${fmtCount(d.samples!)}`).join(' · ')
      }
    } else {
      samplesSummary = '—'
      samplesDetail = ''
    }

    // Auto-detect phase 3 from PCC data presence
    isPhase3 = s.pcc_epoch_val.length > 0

    // Std badge for phase 3
    if (isPhase3 && s.score_dist_std.length > 0) {
      lastStd = s.score_dist_std[s.score_dist_std.length - 1]
      if (lastStd >= 10) { stdBadgeColor = '#3ecf7a'; stdBadgeLabel = `std ${lastStd.toFixed(1)}` }
      else if (lastStd >= 5) { stdBadgeColor = '#f0a030'; stdBadgeLabel = `std ${lastStd.toFixed(1)}` }
      else { stdBadgeColor = '#e05555'; stdBadgeLabel = `std ${lastStd.toFixed(1)} ⚠` }
    }

    // Auto-scroll logs if new entries — use setTimeout to ensure DOM has updated
    if (s.logs.length !== logCount) {
      logCount = s.logs.length
      scrollLogs()
    }
  }

  // Robust auto-scroll: MutationObserver fires exactly when DOM children change
  let logObserver: MutationObserver | null = null

  function scrollLogs() {
    if (logEl) logEl.scrollTop = logEl.scrollHeight
  }

  function setupLogObserver() {
    if (!logEl || logObserver) return
    logObserver = new MutationObserver(scrollLogs)
    logObserver.observe(logEl, { childList: true })
    scrollLogs()
  }

  // Generic chart draw
  interface ChartDataset { data: number[]; color: string; lw?: number; dash?: boolean; dots?: boolean }
  interface ChartOpts { min?: number; max?: number; pL?: number; pB?: number; xLabels?: string[]; refLine?: number; refColor?: string }

  function drawChart(cv: HTMLCanvasElement | undefined, datasets: ChartDataset[], opts: ChartOpts = {}) {
    if (!cv) return
    const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1
    const W = cv.offsetWidth * dpr
    const H = cv.offsetHeight * dpr
    if (W < 10 || H < 10) return
    cv.width = W; cv.height = H
    const cx = cv.getContext('2d')!
    const pL = (opts.pL ?? 36) * dpr, pR = 8 * dpr, pT = 10 * dpr, pB = (opts.pB ?? 20) * dpr
    const gw = W - pL - pR, gh = H - pT - pB

    cx.fillStyle = '#0d1018'; cx.fillRect(0, 0, W, H)

    const allV = datasets.flatMap(d => d.data).filter(isFinite)
    if (!allV.length) return
    const mn = opts.min ?? Math.min(...allV) * 0.97
    const mx = opts.max ?? Math.max(...allV) * 1.03
    const norm = (v: number) => (v - mn) / ((mx - mn) || 1)

    cx.strokeStyle = 'rgba(255,255,255,.04)'; cx.lineWidth = 1
    for (const v of [0.25, 0.5, 0.75, 1]) {
      const y = pT + gh - v * gh
      cx.beginPath(); cx.moveTo(pL, y); cx.lineTo(W - pR, y); cx.stroke()
      cx.fillStyle = 'rgba(255,255,255,.18)'
      cx.font = `${9 * dpr}px JetBrains Mono,monospace`
      cx.textAlign = 'right'
      const lv = mn + (mx - mn) * v
      cx.fillText(lv > 10 ? lv.toFixed(0) : lv.toFixed(2), pL - 4 * dpr, y + 4 * dpr)
    }

    if (opts.xLabels) {
      opts.xLabels.forEach((l, i) => {
        if (!l) return
        const x = pL + (i / (opts.xLabels!.length - 1 || 1)) * gw
        cx.fillStyle = 'rgba(255,255,255,.16)'
        cx.font = `${9 * dpr}px JetBrains Mono,monospace`
        cx.textAlign = 'center'
        cx.fillText(l, x, H - 3 * dpr)
      })
    }

    // Reference line (e.g. PCC 0.70 target)
    if (opts.refLine !== undefined) {
      const refY = pT + gh - norm(opts.refLine) * gh
      cx.strokeStyle = opts.refColor || 'rgba(255,255,255,.25)'
      cx.lineWidth = 1 * dpr
      cx.setLineDash([6 * dpr, 4 * dpr])
      cx.beginPath(); cx.moveTo(pL, refY); cx.lineTo(W - pR, refY); cx.stroke()
      cx.setLineDash([])
      cx.fillStyle = opts.refColor || 'rgba(255,255,255,.25)'
      cx.font = `${9 * dpr}px JetBrains Mono,monospace`
      cx.textAlign = 'left'
      cx.fillText(String(opts.refLine), pL + 4 * dpr, refY - 4 * dpr)
    }

    datasets.forEach(ds => {
      const n = ds.data.length; if (n < 1) return
      cx.strokeStyle = ds.color; cx.lineWidth = (ds.lw || 1.5) * dpr
      if (ds.dash) cx.setLineDash([4 * dpr, 3 * dpr]); else cx.setLineDash([])
      cx.beginPath()
      ds.data.forEach((v, i) => {
        const x = pL + (n === 1 ? gw / 2 : i / (n - 1) * gw)
        const y = pT + gh - norm(v) * gh
        i === 0 ? cx.moveTo(x, y) : cx.lineTo(x, y)
      })
      cx.stroke(); cx.setLineDash([])

      if (ds.dots) {
        ds.data.forEach((v, i) => {
          const x = pL + (n === 1 ? gw / 2 : i / (n - 1) * gw)
          const y = pT + gh - norm(v) * gh
          cx.fillStyle = ds.color; cx.beginPath(); cx.arc(x, y, 3.5 * dpr, 0, Math.PI * 2); cx.fill()
        })
      }
    })
  }

  function renderCharts() {
    if (!s) return
    const epLabels = s.loss_epoch_train.map((_: number, i: number) => `E${i + 1}`)
    const toneLabels = s.tone_epoch.T1.map((_: number, i: number) => `E${i + 1}`)

    // Batch labels for loss chart — show ~5 labels spread across history
    const lossLen = s.loss_history.length
    const lossLabels: string[] = []
    if (lossLen > 1) {
      const stepBase = Math.max(0, s.step - (lossLen - 1) * 100)
      const stride = Math.max(1, Math.floor(lossLen / 5))
      for (let i = 0; i < lossLen; i++) {
        if (i % stride === 0 || i === lossLen - 1) {
          const batchNum = stepBase + i * 100
          lossLabels.push(batchNum >= 1000 ? `${(batchNum / 1000).toFixed(1)}k` : String(batchNum))
        } else {
          lossLabels.push('')
        }
      }
    }

    drawChart(cLoss, [
      { data: s.loss_history, color: 'rgba(224,85,85,.45)', lw: 1 },
      { data: s.loss_ma_history, color: '#f0a030', lw: 2 }
    ], { xLabels: lossLabels.length > 1 ? lossLabels : undefined })
    drawChart(cLossEp, [
      { data: s.loss_epoch_train, color: '#e05555', lw: 2, dots: true },
      { data: s.loss_epoch_val, color: '#f0a030', lw: 2, dots: true }
    ], { xLabels: epLabels })
    drawChart(cAcc, [
      { data: s.acc_epoch_train, color: '#5b8dee', lw: 2, dots: true },
      { data: s.acc_epoch_val, color: '#3ecf7a', lw: 2, dots: true }
    ], { min: 0, max: 100, xLabels: epLabels })
    drawChart(cTone, TONES.map(t => ({
      data: s.tone_epoch[t.tag as ToneKey],
      color: t.color,
      lw: 1.5,
      dots: true
    })), { min: 0, max: 100, xLabels: toneLabels })

    // Phase 3 charts
    if (isPhase3) {
      const pccLabels = s.pcc_epoch_val.map((_: number, i: number) => `E${i + 1}`)
      drawChart(cPcc, [
        { data: s.pcc_epoch_train, color: '#5b8dee', lw: 1.5, dash: true, dots: false },
        { data: s.pcc_epoch_val, color: '#3ecf7a', lw: 2, dots: true }
      ], { min: 0, max: 1, xLabels: pccLabels, refLine: 0.70, refColor: 'rgba(240,160,48,.4)' })

      const mseLabels = s.mse_epoch.score_global.map((_: number, i: number) => `E${i + 1}`)
      drawChart(cMse, MSE_HEADS.map(h => ({
        data: s.mse_epoch[h.key as MseKey],
        color: h.color,
        lw: 1.5,
        dots: true
      })), { xLabels: mseLabels })
    }
  }

  onMount(() => {
    // Manual subscription — ONE callback per store update, NO Svelte reactive overhead
    const unsub = training.subscribe(val => {
      s = val
      recompute()
    })

    // logEl is inside {#if s}, so it's not available until after the first subscription
    // fires and Svelte renders the block. Poll briefly until it appears.
    const logElCheck = setInterval(() => {
      if (logEl) {
        setupLogObserver()
        clearInterval(logElCheck)
      }
    }, 50)

    // Render charts on a fixed 2s interval — completely decoupled from store updates
    renderCharts()
    const chartTimer = setInterval(renderCharts, 2000)

    const onResize = () => renderCharts()
    window.addEventListener('resize', onResize)

    const agoTimer = setInterval(() => {
      if (s) {
        startedAgo = s.started_at > 0 ? fmtAgo(Math.round((Date.now() - s.started_at) / 1000)) : '—'
        updatedAgo = s.last_update_at > 0 ? fmtAgo(Math.round((Date.now() - s.last_update_at) / 1000)) : '—'
      }
    }, 5000)

    return () => {
      unsub()
      clearInterval(logElCheck)
      clearInterval(chartTimer)
      clearInterval(agoTimer)
      if (logObserver) logObserver.disconnect()
      window.removeEventListener('resize', onResize)
    }
  })
</script>

{#if s}
<div class="card">

  <!-- Header -->
  <div class="card-head">
    <div>
      {#if hasConfig}
        <div class="model-name">{s.config.model_name}</div>
        <div class="model-sub">Phase {s.config.phase} · {s.config.device} · AMP {s.config.amp ? 'ON' : 'OFF'} · batch {s.config.batch_size} · LR {fmtLr(s.config.lr)}</div>
      {:else}
        <div class="model-name wait">En attente du trainer…</div>
        <div class="model-sub">Aucune session active</div>
      {/if}
    </div>
    {#if s.status === 'running'}
      <div class="status"><div class="dot"></div>En cours</div>
    {:else if s.status === 'done'}
      <div class="status" style="color:#5b8dee">Terminé</div>
    {:else if s.status === 'failed'}
      <div class="status" style="color:#e05555">Échoué</div>
    {:else}
      <div class="status" style="color:#2e3a55">Inactif</div>
    {/if}
  </div>

  <!-- KPI strip -->
  <div class="kpi-strip">
    {#if isPhase3}
      <div class="kpi">
        <div class="kl">Best PCC</div>
        <div class="kv" style="color:#3ecf7a">{s.best_pcc.toFixed(4)}</div>
        <div class="ks">Epoch {s.best_pcc_epoch}</div>
      </div>
      <div class="kpi">
        <div class="kl">Std scores</div>
        <div class="kv" style="color:{stdBadgeColor}">{stdBadgeLabel}</div>
        <div class="ks" style="color:{stdBadgeColor}">{lastStd >= 10 ? 'ok' : lastStd >= 5 ? 'faible' : 'collapse'}</div>
      </div>
    {:else}
      <div class="kpi">
        <div class="kl">Best acc.</div>
        <div class="kv" style="color:#3ecf7a">{s.best_acc.toFixed(1)}%</div>
        <div class="ks">Epoch {s.best_acc_epoch}</div>
      </div>
      <div class="kpi">
        <div class="kl">Best loss</div>
        <div class="kv" style="color:#f0a030">{s.best_loss === Infinity ? '—' : s.best_loss.toFixed(3)}</div>
        <div class="ks">{s.best_loss_epoch > 0 ? `Val · ep${s.best_loss_epoch}` : '—'}</div>
      </div>
    {/if}
    <div class="kpi">
      <div class="kl">Epoch</div>
      <div class="kv" style="color:#fff">{s.epoch}<span class="kv-den"> / {s.config.total_epochs}</span></div>
      <div class="ks">~{epochTimeMin} min/ep</div>
    </div>
    <div class="kpi">
      <div class="kl">Loss batch</div>
      <div class="kv" style="color:#f0a030">{s.loss.toFixed(4)}</div>
      <div class="ks">MA5 {s.loss_ma5.toFixed(3)}</div>
    </div>
    <div class="kpi">
      <div class="kl">Vitesse</div>
      <div class="kv" style="color:#5b8dee">{s.speed_bps.toFixed(1)}</div>
      <div class="ks">batch/s</div>
    </div>
    <div class="kpi">
      <div class="kl">Échantillons</div>
      <div class="kv" style="color:#fff">{samplesSummary}</div>
      <div class="ks">{samplesDetail}</div>
    </div>
  </div>

  <!-- 3-col: Config | Logs | Progress + GPU stacked -->
  <div class="mid-row">
    <!-- Config -->
    <div class="mid-col">
      <span class="sec-lbl">Configuration</span>
      {#if hasConfig}
        <div class="cf-row"><span class="cf-k">Modèle</span><span class="cf-v hi">{s.config.model_name}</span></div>
        <div class="cf-row"><span class="cf-k">Locuteurs</span><span class="cf-v">{s.config.speakers}</span></div>
        <div class="cf-row"><span class="cf-k">Warmup</span><span class="cf-v">3% · {s.config.warmup_steps.toLocaleString('fr')} steps</span></div>
        <div class="cf-row"><span class="cf-k">Grad norm</span><span class="cf-v">{s.grad_norm.toFixed(3)}</span></div>
        <div class="cf-row"><span class="cf-k">LR actuel</span><span class="cf-v hi">{fmtLr(s.lr)}</span></div>
        <div class="cf-row"><span class="cf-k">Overfit</span><span class="cf-v {overfitClass}">{overfitLabel}</span></div>
        <div class="cf-row"><span class="cf-k">Démarré</span><span class="cf-v warn">{startedAgo}</span></div>
        <div class="cf-row"><span class="cf-k">Maj.</span><span class="cf-v">{updatedAgo}</span></div>
        <div style="margin-top:6px">
          <span class="sec-lbl">Datasets · Phase {s.config.phase}</span>
          <div class="ds-tags">
            {#each s.config.datasets as ds}
              {@const inPhase = ds.phases.includes(s.config.phase)}
              <span class="ds-tag" class:active={inPhase} class:idle={!inPhase}>
                {ds.name}
                {#if ds.samples}
                  <span class="ds-count">{fmtCount(ds.samples)}</span>
                {/if}
              </span>
            {/each}
          </div>
        </div>
      {:else}
        <div class="wait-msg">En attente de connexion du trainer…</div>
      {/if}
    </div>

    <!-- Logs -->
    <div class="mid-col log-col">
      <span class="sec-lbl">Log</span>
      <div class="log-header">
        <span class="lh-ep">Epoch</span>
        <span class="lh-batch">Batch</span>
        <span class="lh-loss">Loss</span>
        <span class="lh-eta">ETA</span>
      </div>
      <div class="log-lines" bind:this={logEl}>
        {#each s.logs as log, i (log.step)}
          {@const isLast = i === s.logs.length - 1}
          <div class="log-line" class:latest={isLast}>
            <span class="log-ep">{log.epoch}</span>
            <span class="log-batch" class:lv={isLast}>{log.step.toLocaleString('fr')}<span class="log-sep"> / </span>{log.steps_per_epoch.toLocaleString('fr')}</span>
            <span class="log-loss">{log.loss}</span>
            <span class="log-eta">{log.eta}</span>
          </div>
        {/each}
      </div>
    </div>

    <!-- Progress + GPU stacked -->
    <div class="mid-col right-col">
      <!-- Progression -->
      <div class="right-section">
        <span class="sec-lbl">Progression</span>
        <div class="prog-blocks">
          <div class="prog-block">
            <div class="prog-top">
              <span class="prog-name">Epoch <span class="mono">{s.epoch}</span></span>
              <span class="prog-pct" style="color:#5b8dee">{epochPct.toFixed(1)}%</span>
            </div>
            <div class="track"><div class="fill" style="width:{epochPct}%;background:linear-gradient(90deg,#1e3a6e,#5b8dee)"></div></div>
            <div class="prog-detail">
              <span class="prog-eta">Batch <span class="mono">{s.step.toLocaleString('fr')}</span> / <span class="mono">{s.steps_per_epoch.toLocaleString('fr')}</span></span>
              <span class="prog-eta">ETA · {fmtTime(s.eta_epoch_s)}</span>
            </div>
            <div class="prog-detail">
              <span class="prog-eta">~{epochTimeMin} min/ep</span>
            </div>
          </div>
          <div class="prog-block">
            <div class="prog-top">
              <span class="prog-name">Global</span>
              <span class="prog-pct" style="color:#3ecf7a">{globalPct.toFixed(1)}%</span>
            </div>
            <div class="track"><div class="fill" style="width:{globalPct}%;background:linear-gradient(90deg,#0d4a2e,#3ecf7a)"></div></div>
            <div class="prog-detail">
              <span class="prog-eta"><span class="mono">{globalStepsDone.toLocaleString('fr')}</span> / <span class="mono">{totalSteps.toLocaleString('fr')}</span></span>
              <span class="prog-eta">ETA · {fmtTime(s.eta_global_s)}</span>
            </div>
          </div>
        </div>
      </div>
      <!-- GPU -->
      <div class="right-section">
        <span class="sec-lbl">GPU · RTX 3070 Ti</span>
        <div class="gpu-bars">
          <div class="gpu-row">
            <span class="gpu-lbl">Util.</span>
            <div class="gpu-track"><div class="gpu-fill" style="width:{s.gpu.util_pct}%;background:#5b8dee"></div></div>
            <span class="gpu-val" style="color:#5b8dee">{s.gpu.util_pct}%</span>
          </div>
          <div class="gpu-row">
            <span class="gpu-lbl">VRAM</span>
            <div class="gpu-track"><div class="gpu-fill" style="width:{gpuVramPct}%;background:#9b6dff"></div></div>
            <span class="gpu-val" style="color:#9b6dff">{s.gpu.vram_used_gb.toFixed(1)}/{s.gpu.vram_total_gb}G</span>
          </div>
          <div class="gpu-row">
            <span class="gpu-lbl">Temp.</span>
            <div class="gpu-track"><div class="gpu-fill" style="width:{gpuTempPct}%;background:{gpuTempColor}"></div></div>
            <span class="gpu-val" style="color:{gpuTempColor}">{s.gpu.temp_c}°C</span>
          </div>
          <div class="gpu-row">
            <span class="gpu-lbl">Power</span>
            <div class="gpu-track"><div class="gpu-fill" style="width:{gpuPowerPct}%;background:#e05555"></div></div>
            <span class="gpu-val" style="color:#e05555">{s.gpu.power_w}W</span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- Loss real-time — full width -->
  <div class="loss-section">
    <div class="chart-lbl">
      <span class="sec-lbl">Loss temps réel</span>
      <span class="dot-legend" style="color:#e05555">● Loss</span>
      <span class="dot-legend" style="color:#f0a030">● MA(5)</span>
    </div>
    <canvas bind:this={cLoss} style="height:200px"></canvas>
  </div>

  <!-- Bottom 3-col: Loss/epoch | Accuracy or PCC | Tone or MSE -->
  <div class="bot-row">
    <div class="bot-col">
      <div class="chart-lbl">
        <span class="sec-lbl">Loss / epoch</span>
        <span class="dot-legend" style="color:#e05555">● Train</span>
        <span class="dot-legend" style="color:#f0a030">● Val</span>
      </div>
      <canvas bind:this={cLossEp} style="height:150px"></canvas>
    </div>
    {#if isPhase3}
      <div class="bot-col">
        <div class="chart-lbl">
          <span class="sec-lbl">PCC / epoch</span>
          <span class="dot-legend" style="color:#5b8dee">- - Train</span>
          <span class="dot-legend" style="color:#3ecf7a">● Val</span>
          <span class="dot-legend" style="color:rgba(240,160,48,.4)">— 0.70</span>
        </div>
        <canvas bind:this={cPcc} style="height:150px"></canvas>
      </div>
      <div class="bot-col">
        <div class="chart-lbl">
          <span class="sec-lbl">MSE par tête</span>
          <div class="tone-legend">
            {#each MSE_HEADS as h}
              <span class="dot-legend" style="color:{h.color}">● {h.label}</span>
            {/each}
          </div>
        </div>
        <canvas bind:this={cMse} style="height:150px"></canvas>
      </div>
    {:else}
      <div class="bot-col">
        <div class="chart-lbl">
          <span class="sec-lbl">Accuracy / epoch</span>
          <span class="dot-legend" style="color:#5b8dee">● Train</span>
          <span class="dot-legend" style="color:#3ecf7a">● Val</span>
        </div>
        <canvas bind:this={cAcc} style="height:150px"></canvas>
      </div>
      <div class="bot-col">
        <div class="chart-lbl">
          <span class="sec-lbl">Accuracy par ton</span>
          <div class="tone-legend">
            {#each TONES as t}
              <span class="dot-legend" style="color:{t.color}">● {t.tag}</span>
            {/each}
          </div>
        </div>
        <canvas bind:this={cTone} style="height:150px"></canvas>
        <div class="tone-grid">
          {#each lastToneAccs as t}
            <div class="tone-card">
              <div class="tone-tag" style="color:{t.color}">{t.tag}</div>
              <div class="tone-val" style="color:{t.color}">{t.val.toFixed(1)}%</div>
            </div>
          {/each}
        </div>
      </div>
    {/if}
  </div>

</div>
{/if}

<style>
  .card { background: #0d1018; border: 1px solid #1a2035; border-radius: 14px; overflow: hidden; box-shadow: 0 4px 32px rgba(0,0,0,.5) }
  .mono { font-family: 'JetBrains Mono', monospace }
  .sec-lbl { font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: #3d4a60; font-family: 'JetBrains Mono', monospace }

  /* Header */
  .card-head { display: flex; justify-content: space-between; align-items: center; padding: 14px 24px; border-bottom: 1px solid #1a2035 }
  .model-name { font-size: 20px; color: #fff; font-weight: 500 }
  .model-name.wait { color: #3d4a60; font-size: 16px; font-style: italic }
  .model-sub { font-size: 12px; color: #3d4a60; font-family: 'JetBrains Mono', monospace; margin-top: 4px }
  .status { display: flex; align-items: center; gap: 8px; font-size: 13px; font-family: 'JetBrains Mono', monospace; color: #3ecf7a }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: #3ecf7a; box-shadow: 0 0 8px #3ecf7a; animation: pulse 2s infinite }
  @keyframes pulse { 0%,100% { opacity: 1 } 50% { opacity: .35 } }

  /* KPI */
  .kpi-strip { display: grid; grid-template-columns: repeat(6, 1fr); border-bottom: 1px solid #1a2035 }
  .kpi { padding: 10px 16px; border-right: 1px solid #1a2035; text-align: center }
  .kpi:last-child { border-right: none }
  .kl { font-size: 10px; letter-spacing: .1em; text-transform: uppercase; color: #3d4a60; font-family: 'JetBrains Mono', monospace; margin-bottom: 6px }
  .kv { font-family: 'JetBrains Mono', monospace; font-size: 22px; font-weight: 500; line-height: 1 }
  .kv-den { font-size: 13px; color: #3d4a60 }
  .ks { font-size: 10px; color: #3d4a60; margin-top: 4px; font-family: 'JetBrains Mono', monospace }

  /* 3-col row: Config | Logs | Right (Progress + GPU) */
  .mid-row { display: grid; grid-template-columns: 1fr 1fr 1fr; border-bottom: 1px solid #1a2035; overflow: hidden }
  .mid-col { padding: 12px 14px }
  .mid-col:not(:last-child) { border-right: 1px solid #1a2035 }
  .mid-col .sec-lbl { display: block; margin-bottom: 8px }

  /* Config */
  .cf-row { display: flex; justify-content: space-between; align-items: baseline; padding: 3px 0; border-bottom: 1px solid rgba(255,255,255,.03) }
  .cf-row:last-child { border-bottom: none }
  .cf-k { font-size: 11px; color: #4d5a70; font-family: 'JetBrains Mono', monospace; white-space: nowrap }
  .cf-v { font-size: 11px; color: #8892a4; font-family: 'JetBrains Mono', monospace; text-align: right }
  .cf-v.hi { color: #5b8dee }
  .cf-v.ok { color: #3ecf7a }
  .cf-v.warn { color: #f0a030 }
  .cf-v.err { color: #e05555 }

  .ds-tags { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 6px }
  .ds-tag { font-size: 9px; border-radius: 4px; padding: 2px 6px; font-family: 'JetBrains Mono', monospace }
  .ds-tag.active { background: rgba(62,207,122,.12); border: 1px solid rgba(62,207,122,.3); color: #3ecf7a }
  .ds-tag.idle { background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.08); color: #3d4a60 }
  .ds-count { opacity: .6; margin-left: 3px }

  .wait-msg { color: #2e3a55; font-size: 12px; font-family: 'JetBrains Mono', monospace; font-style: italic; padding: 20px 0; text-align: center }

  /* Logs */
  .log-col { display: flex; flex-direction: column }
  .log-header { display: grid; grid-template-columns: 42px 1fr 68px 52px; gap: 8px; font-size: 9px; letter-spacing: .08em; text-transform: uppercase; color: #2e3a50; font-family: 'JetBrains Mono', monospace; padding: 0 0 5px; border-bottom: 1px solid rgba(255,255,255,.04); margin-bottom: 4px }
  .lh-batch { text-align: right }
  .lh-loss { text-align: right }
  .lh-eta { text-align: right }
  .log-lines { flex: 1; overflow-y: auto; overflow-x: hidden; display: flex; flex-direction: column; gap: 1px; min-height: 0; max-height: 240px; padding-right: 14px }
  .log-lines::-webkit-scrollbar { width: 3px }
  .log-lines::-webkit-scrollbar-track { background: transparent }
  .log-lines::-webkit-scrollbar-thumb { background: #1a2035; border-radius: 2px }
  .log-line { display: grid; grid-template-columns: 42px 1fr 68px 52px; gap: 8px; font-size: 12px; font-family: 'JetBrains Mono', monospace; color: #3d4a60; white-space: nowrap; padding: 2px 0 }
  .log-line.latest { color: #8892a4; background: rgba(91,141,238,.04); border-radius: 3px; padding: 2px 4px; margin: 0 -4px }
  .log-line.latest .lv { color: #5b8dee }
  .log-ep { color: #4d5a70; text-align: center }
  .log-batch { text-align: right }
  .log-sep { color: #2e3a50 }
  .log-loss { text-align: right; color: #6b7890 }
  .log-eta { text-align: right; color: #4d5a70 }

  /* GPU bars */
  .right-col { display: flex; flex-direction: column; gap: 0 }
  .right-section { padding-bottom: 10px }
  .right-section:first-child { border-bottom: 1px solid rgba(255,255,255,.04); margin-bottom: 10px }
  .right-section .sec-lbl { display: block; margin-bottom: 8px }
  .gpu-bars { display: flex; flex-direction: column; gap: 8px }
  .gpu-row { display: grid; grid-template-columns: 38px 1fr 56px; gap: 6px; align-items: center }
  .gpu-lbl { font-size: 10px; color: #4d5a70; font-family: 'JetBrains Mono', monospace; text-align: right }
  .gpu-track { height: 6px; background: rgba(255,255,255,.05); border-radius: 3px; overflow: hidden }
  .gpu-fill { height: 100%; border-radius: 3px; transition: width .6s cubic-bezier(.4,0,.2,1) }
  .gpu-val { font-size: 11px; font-family: 'JetBrains Mono', monospace }

  /* Progress */
  .prog-blocks { display: flex; flex-direction: column; gap: 14px }
  .prog-top { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px }
  .prog-name { font-size: 12px; color: #8892a4 }
  .prog-pct { font-size: 13px; font-family: 'JetBrains Mono', monospace }
  .track { height: 5px; background: rgba(255,255,255,.05); border-radius: 3px; overflow: hidden; margin-bottom: 5px }
  .fill { height: 100%; border-radius: 3px; transition: width .5s cubic-bezier(.4,0,.2,1) }
  .prog-detail { display: flex; justify-content: space-between }
  .prog-eta { font-size: 10px; color: #3d4a60; font-family: 'JetBrains Mono', monospace }

  /* Loss full-width */
  .loss-section { padding: 12px 18px; border-bottom: 1px solid #1a2035 }

  /* Bottom 3-col */
  .bot-row { display: grid; grid-template-columns: 1fr 1fr 1fr }
  .bot-col { padding: 12px 18px }
  .bot-col:not(:last-child) { border-right: 1px solid #1a2035 }

  .chart-lbl { margin-bottom: 8px; display: flex; align-items: center; gap: 10px; flex-wrap: wrap }
  canvas { display: block; width: 100% }

  .dot-legend { display: flex; gap: 2px; align-items: center; font-size: 11px; font-family: 'JetBrains Mono', monospace }
  .tone-legend { display: flex; gap: 6px; flex-wrap: wrap }

  /* Tone */
  .tone-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 4px; margin-top: 8px }
  .tone-card { background: #0a0c12; border: 1px solid #1a2035; border-radius: 6px; padding: 4px 2px; text-align: center }
  .tone-tag { font-size: 10px; font-family: 'JetBrains Mono', monospace }
  .tone-val { font-family: 'JetBrains Mono', monospace; font-size: 13px; font-weight: 500 }
</style>
