<script lang="ts">
  import { onMount, tick } from 'svelte'
  import type { ScoringResult, SyllableScore } from '$lib/types'

  export let scoring: ScoringResult
  export let feedbackRaw: string = ''
  export let feedbackDone: boolean = false

  const MARKS = ['¯', '´', 'ˇ', '`', '·']
  const SYL_PTS = 6

  const HINTS: Record<string, string> = {
    ton_plat: 'Descente initiale absente — le ton 3 doit plonger avant de remonter.',
    ton_trop_lent: 'Le ton 4 doit chuter franchement et rapidement.',
    sandhi_manquant: 'T3+T3 → le premier devient ton 2 en pratique.',
    ton_confondu: 'Le ton détecté ne correspond pas au ton attendu.',
    retroflex_palatale: 'Confusion rétroflexe/palatale — la langue doit se recourber vers le palais dur.',
    aspiration_manquante: 'L\'aspiration est absente — un souffle d\'air doit accompagner la consonne.',
    nasale_finale: 'La finale nasale n\'est pas correcte — distinguez -n (langue contre le palais) et -ng (son nasal du fond).',
    voyelle_incorrecte: 'La voyelle produite ne correspond pas à la voyelle attendue.'
  }

  function scoreColor(s: number) {
    if (s >= 80) return { h: '#3ecf7a', bg: 'rgba(62,207,122,.10)', bd: 'rgba(62,207,122,.30)' }
    if (s >= 60) return { h: '#5b8dee', bg: 'rgba(91,141,238,.10)', bd: 'rgba(91,141,238,.28)' }
    if (s >= 40) return { h: '#f0a030', bg: 'rgba(240,160,48,.10)', bd: 'rgba(240,160,48,.28)' }
    return { h: '#e05555', bg: 'rgba(224,85,85,.10)', bd: 'rgba(224,85,85,.30)' }
  }

  let canvas: HTMLCanvasElement
  let activeIdx = -1
  let activeSyl: SyllableScore | null = null

  const PAD_B = 56

  function draw(hi: number) {
    if (!canvas) return
    const syls = scoring.syllabes
    const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1
    const W = canvas.offsetWidth * dpr
    const H = 220 * dpr
    canvas.width = W
    canvas.height = H
    const cx = canvas.getContext('2d')!
    const pL = 42 * dpr, pR = 10 * dpr, pT = 18 * dpr, pB = PAD_B * dpr
    const gw = W - pL - pR, gh = H - pT - pB

    cx.fillStyle = '#0d1018'
    cx.fillRect(0, 0, W, H)

    // Grid lines
    cx.strokeStyle = 'rgba(255,255,255,.03)'
    cx.lineWidth = 1
    for (const v of [0.2, 0.4, 0.6, 0.8]) {
      const y = pT + gh - v * gh
      cx.beginPath(); cx.moveTo(pL, y); cx.lineTo(W - pR, y); cx.stroke()
    }

    // Y-axis labels
    const yLabels = ['grave', 'médium', 'aigu']
    yLabels.forEach((l, i) => {
      cx.fillStyle = 'rgba(255,255,255,.12)'
      cx.font = `${9 * dpr}px JetBrains Mono,monospace`
      cx.textAlign = 'right'
      cx.fillText(l, pL - 4 * dpr, pT + gh - (i + 0.5) / 3 * gh + 3 * dpr)
    })

    const allExp = syls.flatMap(s => s.f0_attendu)
    const allGot = syls.flatMap(s => s.f0_produit)
    const n = allExp.length
    const px = (i: number) => pL + (i / (n - 1)) * gw
    const py = (v: number) => pT + gh - v * gh

    // Per-syllable zones
    syls.forEach((s, si) => {
      const x1 = pL + (si / syls.length) * gw
      const x2 = pL + ((si + 1) / syls.length) * gw
      const mid = (x1 + x2) / 2
      const c = scoreColor(s.scores.global)

      if (si === hi) {
        cx.fillStyle = c.bg.replace(/[\d.]+\)$/, '0.22)')
        cx.fillRect(x1, pT, x2 - x1, gh)
        cx.fillStyle = c.h
        cx.fillRect(x1, pT, x2 - x1, 2 * dpr)
      }

      if (si > 0) {
        cx.strokeStyle = 'rgba(255,255,255,.06)'
        cx.lineWidth = 0.5 * dpr
        cx.setLineDash([3 * dpr, 3 * dpr])
        cx.beginPath(); cx.moveTo(x1, pT); cx.lineTo(x1, pT + gh); cx.stroke()
        cx.setLineDash([])
      }

      // Hanzi label
      const isA = si === hi
      cx.fillStyle = isA ? c.h : 'rgba(255,255,255,.55)'
      cx.font = `${(isA ? 17 : 15) * dpr}px Noto Serif SC,serif`
      cx.textAlign = 'center'
      cx.fillText(s.hanzi, mid, pT + gh + 16 * dpr)

      // Pinyin + tone mark
      cx.fillStyle = isA ? c.h : 'rgba(255,255,255,.22)'
      cx.font = `${10 * dpr}px JetBrains Mono,monospace`
      cx.fillText(s.pinyin + ' ' + MARKS[s.ton_attendu - 1], mid, pT + gh + 27 * dpr)

      // Score badge
      const bw = 26 * dpr, bh = 13 * dpr, bx = mid - bw / 2, by = pT + gh + 33 * dpr
      cx.fillStyle = isA ? c.bd : 'rgba(255,255,255,.05)'
      cx.beginPath(); cx.roundRect(bx, by, bw, bh, 3 * dpr); cx.fill()
      cx.fillStyle = isA ? c.h : 'rgba(255,255,255,.3)'
      cx.font = `500 ${10 * dpr}px JetBrains Mono,monospace`
      cx.fillText(String(s.scores.global), mid, by + 9 * dpr)
    })

    // Expected F0 (dashed)
    cx.strokeStyle = 'rgba(91,141,238,.4)'
    cx.lineWidth = 1.5 * dpr
    cx.setLineDash([5 * dpr, 3 * dpr])
    cx.beginPath()
    allExp.forEach((v, i) => i === 0 ? cx.moveTo(px(i), py(v)) : cx.lineTo(px(i), py(v)))
    cx.stroke()
    cx.setLineDash([])

    // Produced F0 (colored by score)
    for (let i = 0; i < n - 1; i++) {
      const s = Math.min(
        syls[Math.min(Math.floor(i / SYL_PTS), syls.length - 1)].scores.global,
        syls[Math.min(Math.floor((i + 1) / SYL_PTS), syls.length - 1)].scores.global
      )
      cx.strokeStyle = scoreColor(s).h
      cx.lineWidth = 2 * dpr
      cx.beginPath()
      cx.moveTo(px(i), py(allGot[i]))
      cx.lineTo(px(i + 1), py(allGot[i + 1]))
      cx.stroke()
    }

    // Junction dots
    syls.forEach((_, si) => {
      if (si === 0) return
      const i = si * SYL_PTS
      cx.fillStyle = scoreColor(syls[si].scores.global).h
      cx.beginPath()
      cx.arc(px(i), py(allGot[i]), 3 * dpr, 0, Math.PI * 2)
      cx.fill()
    })
  }

  function selectSyllable(i: number) {
    activeIdx = i
    activeSyl = scoring.syllabes[i]
    draw(i)
  }

  function getHint(syl: SyllableScore): { text: string; color: string } {
    if (syl.erreur && HINTS[syl.erreur]) {
      return { text: HINTS[syl.erreur], color: '#f0a030' }
    }
    return { text: 'Prononciation correcte.', color: '#3ecf7a' }
  }

  // Feedback parser — ‹FR›/‹ZH›/‹PY› tags
  interface FbSegment {
    type: 'fr' | 'zh'
    text: string
    pinyin?: string
  }

  function parseFeedback(raw: string): FbSegment[] {
    const segments: FbSegment[] = []
    const re = /‹(FR|ZH|PY)›([\s\S]*?)‹\/\1›/g
    let m: RegExpExecArray | null
    while ((m = re.exec(raw)) !== null) {
      if (m[1] === 'FR') {
        segments.push({ type: 'fr', text: m[2] })
      } else if (m[1] === 'ZH') {
        const inner = m[2]
        const pyM = /‹PY›([\s\S]*?)‹\/PY›/.exec(inner)
        const hz = inner.replace(/‹PY›[\s\S]*?‹\/PY›/, '').trim()
        segments.push({ type: 'zh', text: hz, pinyin: pyM ? pyM[1] : undefined })
      }
    }
    return segments
  }

  $: feedbackSegments = parseFeedback(feedbackRaw)

  let playingHz: string | null = null
  function onHanziClick(hz: string) {
    // In simulation mode — visual flash only (no real TTS)
    playingHz = hz
    setTimeout(() => { playingHz = null }, 700)
  }

  onMount(() => {
    draw(-1)
    const onResize = () => draw(activeIdx)
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  })

  // Score bar helper
  function scoreBars(syl: SyllableScore) {
    return [
      { label: 'global', value: syl.scores.global },
      { label: 'ton', value: syl.scores.ton },
      { label: 'initiale', value: syl.scores.initiale },
      { label: 'finale', value: syl.scores.finale }
    ]
  }
</script>

<div class="card">
  <!-- Header -->
  <div class="card-head">
    <div class="head-left">
      <span class="head-enonce">{scoring.enonce}</span>
      <span class="head-trad">{scoring.traduction}</span>
    </div>
    <div class="head-right">
      <span class="score-val">{scoring.score_global}</span>
      <span class="score-den">/ 100</span>
    </div>
  </div>

  <!-- F0 Graph zone -->
  <div class="graph-zone">
    <canvas bind:this={canvas} height="220"></canvas>
    <div class="hit-strip">
      {#each scoring.syllabes as _, i}
        <button class="hit" on:click={() => selectSyllable(i)} aria-label="Syllabe {i + 1}"></button>
      {/each}
    </div>
  </div>

  <!-- Legend -->
  <div class="legend">
    <div class="leg"><div class="ld" style="border-color:rgba(91,141,238,.45)"></div>attendu</div>
    <div class="leg"><div class="ll" style="background:#3ecf7a"></div>≥ 80</div>
    <div class="leg"><div class="ll" style="background:#5b8dee"></div>60–79</div>
    <div class="leg"><div class="ll" style="background:#f0a030"></div>40–59</div>
    <div class="leg"><div class="ll" style="background:#e05555"></div>&lt; 40</div>
  </div>

  <!-- Detail strip -->
  <div class="detail-strip">
    {#if activeSyl}
      {@const c = scoreColor(activeSyl.scores.global)}
      {@const hint = getHint(activeSyl)}
      <div class="ds-char" style="color:{c.h}">{activeSyl.hanzi}</div>
      <div class="ds-meta">
        <div class="ds-py" style="color:{c.h}">{activeSyl.pinyin} — ton {activeSyl.ton_attendu} {MARKS[activeSyl.ton_attendu - 1]}</div>
        <div class="ds-tr">{activeSyl.traduction}</div>
      </div>
      <div class="ds-bars">
        {#each scoreBars(activeSyl) as bar}
          {@const bc = scoreColor(bar.value)}
          <div class="br">
            <span class="bl">{bar.label}</span>
            <div class="bt"><div class="bf" style="width:{bar.value}%;background:{bc.h}"></div></div>
            <span class="bv" style="color:{bc.h}">{bar.value}</span>
          </div>
        {/each}
      </div>
      <div class="ds-hint" style="color:{hint.color}">{hint.text}</div>
    {:else}
      <div class="ds-char" style="color:#2e3a55">—</div>
      <div class="ds-meta">
        <div class="ds-py" style="color:#2e3a55">sélectionnez une syllabe</div>
        <div class="ds-tr"></div>
      </div>
      <div class="ds-bars"></div>
      <div class="ds-hint" style="border-color:transparent"></div>
    {/if}
  </div>

  <!-- Feedback -->
  <div class="feedback">
    <div class="fb-lbl">Retour pédagogique</div>
    <div class="fb-body">
      {#each feedbackSegments as seg}
        {#if seg.type === 'fr'}
          <span>{seg.text}</span>{' '}
        {:else}
          <button
            class="zh"
            class:playing={playingHz === seg.text}
            on:click={() => onHanziClick(seg.text)}
          >{seg.text}</button>{#if seg.pinyin}<span class="py-ann">{seg.pinyin}</span>{/if}{' '}
        {/if}
      {/each}
      {#if !feedbackDone}
        <span class="cursor">▍</span>
      {/if}
    </div>
  </div>
</div>

<style>
  .card { background: #0d1018; border: 1px solid #1a2035; border-radius: 14px; overflow: hidden; box-shadow: 0 4px 32px rgba(0,0,0,.5) }

  /* Header */
  .card-head { display: flex; justify-content: space-between; align-items: center; padding: 16px 22px 14px; border-bottom: 1px solid #1a2035 }
  .head-left { display: flex; flex-direction: column; gap: 6px }
  .head-enonce { font-family: 'Noto Serif SC', serif; font-size: 22px; color: #fff; letter-spacing: .06em }
  .head-trad { font-size: 13px; color: #4d5a70; font-style: italic }
  .head-right { display: flex; align-items: baseline; gap: 4px }
  .score-val { font-family: 'JetBrains Mono', monospace; font-size: 36px; font-weight: 500; color: #fff; line-height: 1 }
  .score-den { font-size: 13px; color: #3d4a60; font-family: 'JetBrains Mono', monospace }

  /* Graph */
  .graph-zone { position: relative; cursor: default }
  canvas { display: block; width: 100% }
  .hit-strip { display: flex; position: absolute; bottom: 0; left: 0; right: 0; height: 56px }
  .hit { flex: 1; cursor: pointer; background: none; border: none; padding: 0 }

  /* Legend */
  .legend { display: flex; gap: 16px; padding: 9px 22px; border-top: 1px solid #1a2035; border-bottom: 1px solid #1a2035 }
  .leg { display: flex; align-items: center; gap: 6px; font-size: 11px; color: #3d4a60; font-family: 'JetBrains Mono', monospace }
  .ll { width: 22px; height: 2px; border-radius: 1px }
  .ld { width: 22px; height: 0; border-top: 2px dashed }

  /* Detail strip */
  .detail-strip { display: flex; align-items: center; gap: 16px; padding: 12px 22px; border-bottom: 1px solid #1a2035; min-height: 58px }
  .ds-char { font-family: 'Noto Serif SC', serif; font-size: 34px; line-height: 1 }
  .ds-meta { display: flex; flex-direction: column; gap: 3px; flex: 0 0 auto }
  .ds-py { font-family: 'JetBrains Mono', monospace; font-size: 13px }
  .ds-tr { font-size: 12px; color: #4d5a70 }
  .ds-bars { flex: 1; display: flex; flex-direction: column; gap: 4px }
  .br { display: flex; align-items: center; gap: 8px }
  .bl { font-size: 10px; color: #3d4a60; width: 55px; text-align: right; font-family: 'JetBrains Mono', monospace }
  .bt { flex: 1; height: 4px; background: rgba(255,255,255,.05); border-radius: 2px; overflow: hidden }
  .bf { height: 100%; border-radius: 2px; transition: width .3s cubic-bezier(.4,0,.2,1) }
  .bv { font-family: 'JetBrains Mono', monospace; font-size: 10px; width: 24px }
  .ds-hint { font-size: 11px; font-style: italic; flex: 0 0 220px; line-height: 1.5; padding-left: 10px; border-left: 1px solid #1a2035 }

  /* Feedback */
  .feedback { padding: 16px 22px }
  .fb-lbl { font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: #3d4a60; margin-bottom: 10px; font-family: 'JetBrains Mono', monospace }
  .fb-body { font-size: 14px; line-height: 1.9; color: #8892a4 }
  .zh { font-family: 'Noto Serif SC', serif; font-size: 15px; color: #fff; cursor: pointer; border: none; background: none; padding: 0; border-bottom: 1px dotted rgba(255,255,255,.2); transition: color .12s, border-color .12s; display: inline-block; margin: 0 1px }
  .zh:hover { color: #5b8dee; border-color: #5b8dee }
  .zh.playing { color: #5b8dee }
  .py-ann { font-size: 11px; font-family: 'JetBrains Mono', monospace; color: #3d4a60; margin-left: 2px }

  /* Streaming cursor */
  .cursor { color: #5b8dee; animation: blink 1s step-end infinite }
  @keyframes blink { 50% { opacity: 0 } }
</style>
