<script lang="ts">
  import { onMount, onDestroy } from 'svelte'
  import { training, resetTraining } from '$lib/stores/training'
  import { wsTrainClient } from '$lib/wsTrain'
  import TrainingMonitor from '$lib/components/TrainingMonitor.svelte'

  $: status = $training.status

  function start() {
    resetTraining()
    wsTrainClient.connect()
  }

  function stop() {
    wsTrainClient.disconnect()
    training.update(s => ({ ...s, status: 'stopped' }))
  }

  onMount(() => {
    // Auto-start simulation
    start()
  })

  onDestroy(() => {
    wsTrainClient.disconnect()
  })
</script>

<div class="page">
  <header class="page-header">
    <a href="/" class="back">← Accueil</a>
    <h1>Monitoring entraînement</h1>
    <div class="ctrls">
      {#if status === 'running'}
        <button class="btn btn-stop" on:click={stop}>Arrêter</button>
      {:else}
        <button class="btn btn-start" on:click={start}>Démarrer simulation</button>
      {/if}
    </div>
  </header>

  <div class="monitor-wrap">
    <TrainingMonitor />
  </div>
</div>

<style>
  .page { max-width: 1400px; margin: 0 auto; padding: 8px 16px 12px }
  .page-header { display: flex; align-items: center; gap: 14px; margin-bottom: 8px }
  .page-header h1 { font-size: 16px; color: #fff; font-weight: 500; flex: 1 }
  .monitor-wrap { }
  .back { font-size: 11px; color: var(--muted); text-decoration: none; font-family: 'JetBrains Mono', monospace }
  .back:hover { color: var(--accent) }
  .ctrls { display: flex; gap: 8px }
  .btn { border: none; border-radius: 8px; font-family: 'IBM Plex Sans', sans-serif; font-size: 12px; padding: 8px 16px; cursor: pointer }
  .btn-start { background: rgba(62,207,122,.12); color: #3ecf7a; border: 1px solid rgba(62,207,122,.25) }
  .btn-start:hover { background: rgba(62,207,122,.22) }
  .btn-stop { background: rgba(224,85,85,.12); color: #e05555; border: 1px solid rgba(224,85,85,.25) }
  .btn-stop:hover { background: rgba(224,85,85,.22) }
</style>
