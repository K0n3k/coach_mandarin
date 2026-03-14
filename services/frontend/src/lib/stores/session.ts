import { writable } from 'svelte/store'
import type { ScoringResult, TtsSegment } from '$lib/types'

export interface SessionStore {
  id: string | null
  status: 'idle' | 'recording' | 'processing' | 'done' | 'error'
  scoring: ScoringResult | null
  feedbackRaw: string
  feedbackDone: boolean
  ttsQueue: TtsSegment[]
  error: { code: string; message: string } | null
}

export const session = writable<SessionStore>({
  id: null,
  status: 'idle',
  scoring: null,
  feedbackRaw: '',
  feedbackDone: false,
  ttsQueue: [],
  error: null
})
