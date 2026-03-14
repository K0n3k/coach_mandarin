<script lang="ts">
  import { onMount, onDestroy } from 'svelte'
  import { training, resetTraining } from '$lib/stores/training'
  import { wsTrainClient } from '$lib/wsTrain'
  import TrainingMonitor from '$lib/components/TrainingMonitor.svelte'

  $: status = $training.status

  onMount(() => {
    wsTrainClient.connect()
  })

  onDestroy(() => {
    wsTrainClient.disconnect()
  })
</script>

<div class="page">
  <header class="page-header">
    <a href="/" class="back">← Accueil</a>
    <h1>Monitoring entraînement</h1>
    <div class="status-pill" class:running={status === 'running'} class:idle={status === 'idle' || status === 'connecting'} class:done={status === 'done'} class:failed={status === 'failed'}>
      {#if status === 'running'}● Live{:else if status === 'connecting'}○ Connexion…{:else if status === 'done'}● Terminé{:else if status === 'failed'}● Erreur{:else}○ En attente{/if}
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
  .status-pill {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    padding: 4px 12px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,.08);
    background: rgba(255,255,255,.04);
    color: var(--muted);
  }
  .status-pill.running { color: #3ecf7a; border-color: rgba(62,207,122,.25); background: rgba(62,207,122,.08) }
  .status-pill.done { color: #5b8dee; border-color: rgba(91,141,238,.25); background: rgba(91,141,238,.08) }
  .status-pill.failed { color: #e05555; border-color: rgba(224,85,85,.25); background: rgba(224,85,85,.08) }
</style>
