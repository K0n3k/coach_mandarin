import { applyStepEvent, applyEpochEndEvent, applyConfigEvent, training } from '$lib/stores/training'
import type { StepEvent, EpochEndEvent, ConfigEvent } from '$lib/types'

// Real WebSocket client for training monitoring.
// Connects to /ws/training on the Go API (via nginx proxy).

class TrainWsClient {
  private ws: WebSocket | null = null
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null
  private intentional = false

  connect(): void {
    if (this.ws?.readyState === WebSocket.OPEN) return
    this.intentional = false
    this.open()
  }

  disconnect(): void {
    this.intentional = true
    this.clearReconnect()
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
  }

  private open() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
    const url = `${proto}//${location.host}/ws/training`

    this.ws = new WebSocket(url)

    this.ws.onopen = () => {
      this.clearReconnect()
      training.update(s => ({ ...s, status: 'connecting' }))
    }

    this.ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data)
        this.handleMessage(data)
      } catch {
        // ignore malformed
      }
    }

    this.ws.onclose = () => {
      this.ws = null
      if (!this.intentional) {
        training.update(s => ({
          ...s,
          status: s.status === 'running' ? 'running' : 'idle'
        }))
        this.scheduleReconnect()
      }
    }

    this.ws.onerror = () => {
      // onclose will fire after
    }
  }

  private handleMessage(data: any) {
    switch (data.type) {
      case 'status':
        training.update(s => ({
          ...s,
          status: data.status === 'running' ? 'running' : 'idle'
        }))
        break

      case 'trainer_connected':
        training.update(s => ({ ...s, status: 'running' }))
        break

      case 'trainer_disconnected':
        training.update(s => ({ ...s, status: 'done' }))
        break

      case 'config': {
        const cfg = data as ConfigEvent
        applyConfigEvent(cfg.config, cfg.run_id)
        break
      }

      case 'step':
        applyStepEvent(data as StepEvent)
        break

      case 'epoch_end':
        applyEpochEndEvent(data as EpochEndEvent)
        break

      case 'checkpoint':
        // informational, no store update needed
        break
    }
  }

  private scheduleReconnect() {
    this.clearReconnect()
    this.reconnectTimer = setTimeout(() => this.open(), 3000)
  }

  private clearReconnect() {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer)
      this.reconnectTimer = null
    }
  }
}

export const wsTrainClient = new TrainWsClient()
