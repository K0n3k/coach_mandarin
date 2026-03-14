<script lang="ts">
  import { onDestroy } from 'svelte'
  import { session } from '$lib/stores/session'
  import { wsClient } from '$lib/ws'
  import AnalysisCard from '$lib/components/AnalysisCard.svelte'

  let mediaRecorder: MediaRecorder | null = null
  let stream: MediaStream | null = null
  let chunks: Blob[] = []
  let detectedMime = ''

  $: status = $session.status

  async function startRecording() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })

      // PCM priority on Chrome/Edge, Opus fallback (§10)
      const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=pcm')
        ? 'audio/webm;codecs=pcm'
        : 'audio/webm;codecs=opus'
      detectedMime = mimeType

      mediaRecorder = new MediaRecorder(stream, { mimeType })
      chunks = []

      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunks.push(e.data)
      }

      mediaRecorder.onstop = async () => {
        const blob = new Blob(chunks, { type: mimeType })
        // In simulation mode, we don't send the blob anywhere.
        // Just trigger the simulated WS pipeline.
        session.update(s => ({
          ...s,
          status: 'processing',
          scoring: null,
          feedbackRaw: '',
          feedbackDone: false,
          ttsQueue: [],
          error: null
        }))
        wsClient.simulateSession()
      }

      mediaRecorder.start()
      session.update(s => ({ ...s, status: 'recording' }))
    } catch (err) {
      session.update(s => ({
        ...s,
        status: 'error',
        error: { code: 'mic_access_denied', message: 'Impossible d\'accéder au microphone.' }
      }))
    }
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop()
    }
    if (stream) {
      stream.getTracks().forEach(t => t.stop())
      stream = null
    }
  }

  function resetSession() {
    wsClient.disconnect()
    session.set({
      id: null,
      status: 'idle',
      scoring: null,
      feedbackRaw: '',
      feedbackDone: false,
      ttsQueue: [],
      error: null
    })
  }

  onDestroy(() => {
    if (stream) stream.getTracks().forEach(t => t.stop())
    wsClient.disconnect()
  })
</script>

<div class="session-page">
  <header class="page-header">
    <h1>Session de prononciation</h1>
    {#if detectedMime}
      <span class="codec-tag">{detectedMime.includes('pcm') ? 'PCM' : 'Opus'}</span>
    {/if}
  </header>

  <!-- Recording controls -->
  <div class="controls">
    {#if status === 'idle'}
      <button class="btn btn-record" on:click={startRecording}>
        <span class="rec-dot"></span>
        Enregistrer
      </button>
    {:else if status === 'recording'}
      <button class="btn btn-stop" on:click={stopRecording}>
        <span class="stop-icon"></span>
        Arrêter
      </button>
      <span class="recording-label">Enregistrement en cours…</span>
    {:else if status === 'processing'}
      <div class="processing">
        <span class="spinner"></span>
        <span>Analyse en cours…</span>
      </div>
    {:else if status === 'done'}
      <button class="btn btn-reset" on:click={resetSession}>
        Nouvel enregistrement
      </button>
    {:else if status === 'error'}
      <div class="error-box">
        <span class="error-code">{$session.error?.code}</span>
        <span>{$session.error?.message}</span>
      </div>
      <button class="btn btn-reset" on:click={resetSession}>
        Réessayer
      </button>
    {/if}
  </div>

  <!-- Analysis card shown when scoring is available -->
  {#if $session.scoring && (status === 'done' || status === 'processing')}
    <div class="card-wrapper">
      <AnalysisCard
        scoring={$session.scoring}
        feedbackRaw={$session.feedbackRaw}
        feedbackDone={$session.feedbackDone}
      />
    </div>
  {/if}
</div>

<style>
  .session-page { max-width: 720px; margin: 0 auto; padding: 24px 16px }

  .page-header { display: flex; align-items: baseline; gap: 10px; margin-bottom: 24px }
  .page-header h1 { font-size: 16px; color: #fff; font-weight: 500 }
  .codec-tag { font-size: 9px; font-family: 'JetBrains Mono', monospace; color: #5b8dee; background: rgba(91,141,238,.12); border: 1px solid rgba(91,141,238,.25); border-radius: 4px; padding: 1px 6px }

  .controls { display: flex; align-items: center; gap: 12px; margin-bottom: 20px }

  .btn { border: none; border-radius: 8px; font-family: 'IBM Plex Sans', sans-serif; font-size: 13px; padding: 10px 20px; cursor: pointer; display: flex; align-items: center; gap: 8px; transition: background .15s }

  .btn-record { background: rgba(224,85,85,.15); color: #e05555; border: 1px solid rgba(224,85,85,.3) }
  .btn-record:hover { background: rgba(224,85,85,.25) }

  .rec-dot { width: 10px; height: 10px; border-radius: 50%; background: #e05555; animation: pulse-rec 1.5s infinite }
  @keyframes pulse-rec { 0%,100% { opacity: 1 } 50% { opacity: .4 } }

  .btn-stop { background: rgba(240,160,48,.15); color: #f0a030; border: 1px solid rgba(240,160,48,.3) }
  .btn-stop:hover { background: rgba(240,160,48,.25) }
  .stop-icon { width: 10px; height: 10px; border-radius: 2px; background: #f0a030 }

  .recording-label { font-size: 11px; color: #f0a030; font-family: 'JetBrains Mono', monospace }

  .btn-reset { background: rgba(91,141,238,.12); color: #5b8dee; border: 1px solid rgba(91,141,238,.25) }
  .btn-reset:hover { background: rgba(91,141,238,.22) }

  .processing { display: flex; align-items: center; gap: 10px; font-size: 12px; color: #8892a4 }
  .spinner { width: 16px; height: 16px; border: 2px solid rgba(91,141,238,.2); border-top-color: #5b8dee; border-radius: 50%; animation: spin .8s linear infinite }
  @keyframes spin { to { transform: rotate(360deg) } }

  .error-box { background: rgba(224,85,85,.08); border: 1px solid rgba(224,85,85,.2); border-radius: 8px; padding: 10px 14px; font-size: 12px; color: #e05555; display: flex; flex-direction: column; gap: 2px }
  .error-code { font-family: 'JetBrains Mono', monospace; font-size: 9px; opacity: .6 }

  .card-wrapper { animation: fadeIn .3s ease-out }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px) } to { opacity: 1; transform: none } }
</style>
