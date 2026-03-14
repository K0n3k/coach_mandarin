import { session } from '$lib/stores/session'
import type { ScoringResult, TtsSegment } from '$lib/types'

// --- Mock data §4 (你好谢谢你) ---

const MOCK_SCORING: ScoringResult = {
  session_id: 'sim-001',
  mode: 'guidé',
  enonce: '你好，谢谢你。',
  traduction: 'Bonjour, merci à toi.',
  duration_total_s: 2.2,
  score_global: 72,
  score_fluidite: 80,
  score_rythme: 75,
  debit_mots_par_min: 95,
  syllabes: [
    {
      hanzi: '你', pinyin: 'nǐ', traduction: 'tu', ton_attendu: 3, ton_detecte: 3,
      start_s: 0.00, end_s: 0.38,
      scores: { global: 82, ton: 78, initiale: 91, finale: 85 },
      f0_attendu: [0.42, 0.35, 0.22, 0.18, 0.22, 0.30],
      f0_produit: [0.45, 0.38, 0.26, 0.22, 0.25, 0.32],
      erreur: null,
      phonemes: [
        { attendu: 'n', detecte: 'n', score: 91, type_confusion: null, start_ms: 0, end_ms: 80 },
        { attendu: 'i3', detecte: 'i3', score: 78, type_confusion: null, start_ms: 80, end_ms: 380 }
      ]
    },
    {
      hanzi: '好', pinyin: 'hǎo', traduction: 'bonjour', ton_attendu: 3, ton_detecte: 2,
      start_s: 0.38, end_s: 0.80,
      scores: { global: 61, ton: 55, initiale: 72, finale: 68 },
      f0_attendu: [0.42, 0.34, 0.20, 0.16, 0.20, 0.28],
      f0_produit: [0.44, 0.40, 0.37, 0.35, 0.36, 0.38],
      erreur: 'ton_plat',
      phonemes: [
        { attendu: 'h', detecte: 'h', score: 72, type_confusion: null, start_ms: 380, end_ms: 460 },
        { attendu: 'ao3', detecte: 'ao2', score: 55, type_confusion: 'ton_confondu', start_ms: 460, end_ms: 800 }
      ]
    },
    {
      hanzi: '谢', pinyin: 'xiè', traduction: 'remercier', ton_attendu: 4, ton_detecte: 4,
      start_s: 0.85, end_s: 1.20,
      scores: { global: 88, ton: 92, initiale: 82, finale: 89 },
      f0_attendu: [0.72, 0.60, 0.46, 0.32, 0.22, 0.16],
      f0_produit: [0.74, 0.61, 0.47, 0.33, 0.22, 0.17],
      erreur: null,
      phonemes: [
        { attendu: 'x', detecte: 'x', score: 82, type_confusion: null, start_ms: 850, end_ms: 950 },
        { attendu: 'ie4', detecte: 'ie4', score: 92, type_confusion: null, start_ms: 950, end_ms: 1200 }
      ]
    },
    {
      hanzi: '谢', pinyin: 'xiè', traduction: '(redoubl.)', ton_attendu: 4, ton_detecte: 4,
      start_s: 1.20, end_s: 1.55,
      scores: { global: 74, ton: 70, initiale: 80, finale: 76 },
      f0_attendu: [0.72, 0.60, 0.46, 0.32, 0.22, 0.16],
      f0_produit: [0.70, 0.60, 0.50, 0.42, 0.36, 0.32],
      erreur: 'ton_trop_lent',
      phonemes: [
        { attendu: 'x', detecte: 'x', score: 80, type_confusion: null, start_ms: 1200, end_ms: 1300 },
        { attendu: 'ie4', detecte: 'ie4', score: 70, type_confusion: null, start_ms: 1300, end_ms: 1550 }
      ]
    },
    {
      hanzi: '你', pinyin: 'nǐ', traduction: 'toi', ton_attendu: 3, ton_detecte: 1,
      start_s: 1.60, end_s: 2.00,
      scores: { global: 45, ton: 38, initiale: 60, finale: 52 },
      f0_attendu: [0.42, 0.35, 0.22, 0.18, 0.22, 0.30],
      f0_produit: [0.40, 0.38, 0.36, 0.35, 0.35, 0.36],
      erreur: 'ton_plat',
      phonemes: [
        { attendu: 'n', detecte: 'n', score: 60, type_confusion: null, start_ms: 1600, end_ms: 1700 },
        { attendu: 'i3', detecte: 'i1', score: 38, type_confusion: 'ton_confondu', start_ms: 1700, end_ms: 2000 }
      ]
    }
  ]
}

const MOCK_FEEDBACK = `‹FR›Score de 72 sur 100 — bonne base, quelques tons à affiner.‹/FR› ‹FR›Point fort :‹/FR› ‹ZH›谢‹PY›xiè‹/PY›‹/ZH› ‹FR›est très bien prononcé, la chute du ton 4 est nette et franche.‹/FR› ‹FR›À travailler : sur‹/FR› ‹ZH›好‹PY›hǎo‹/PY›‹/ZH› ‹FR›et le dernier‹/FR› ‹ZH›你‹PY›nǐ‹/PY›‹/ZH›‹FR›, le ton 3 reste trop plat — laissez la voix descendre dans un creux, puis remonter légèrement.‹/FR› ‹FR›Exercice : répétez lentement‹/FR› ‹ZH›你好‹PY›níhǎo‹/PY›‹/ZH› ‹FR›en exagérant la descente.‹/FR›`

const MOCK_TTS: TtsSegment = {
  segment_id: 'tts-sim-001',
  lang: 'fr-FR',
  audio_b64: ''
}

// --- Feedback chunking helpers ---

function chunkFeedback(raw: string, chunkSize = 40): string[] {
  const chunks: string[] = []
  let i = 0
  while (i < raw.length) {
    let end = Math.min(i + chunkSize, raw.length)
    // Don't break inside a tag
    if (end < raw.length) {
      const openTag = raw.lastIndexOf('‹', end)
      const closeTag = raw.lastIndexOf('›', end)
      if (openTag > closeTag && openTag >= i) {
        end = openTag
      }
    }
    if (end <= i) end = i + chunkSize
    chunks.push(raw.slice(i, end))
    i = end
  }
  return chunks
}

// --- Simulated WS client ---

class SimWsClient {
  private running = false

  connect(): void {
    // No real connection in simulation mode
  }

  disconnect(): void {
    this.running = false
  }

  send(_event: object): void {
    // No-op in simulation
  }

  /**
   * Simulates the backend pipeline after audio is "sent":
   * 1. scoring_result arrives after ~1s
   * 2. feedback_chunk arrives in streaming chunks (~200ms each)
   * 3. tts_ready arrives after feedback is done
   * 4. session_complete
   */
  async simulateSession(): Promise<void> {
    this.running = true

    const sessionId = `sim-${Date.now()}`

    session.update(s => ({ ...s, id: sessionId, status: 'processing' }))

    // 1. Scoring result (~1s delay)
    await this.delay(1000)
    if (!this.running) return
    session.update(s => ({ ...s, scoring: MOCK_SCORING }))

    // 2. Stream feedback chunks (~200ms per chunk)
    const chunks = chunkFeedback(MOCK_FEEDBACK)
    for (const chunk of chunks) {
      await this.delay(200)
      if (!this.running) return
      session.update(s => ({ ...s, feedbackRaw: s.feedbackRaw + chunk }))
    }

    // 3. TTS ready
    await this.delay(300)
    if (!this.running) return
    session.update(s => ({
      ...s,
      ttsQueue: [...s.ttsQueue, MOCK_TTS]
    }))

    // 4. Session complete
    await this.delay(200)
    if (!this.running) return
    session.update(s => ({
      ...s,
      status: 'done',
      feedbackDone: true
    }))
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms))
  }
}

export const wsClient = new SimWsClient()
