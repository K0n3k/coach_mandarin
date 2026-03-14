// Types canoniques — basés sur le schéma §4 de copilot-instructions.md

export interface PhonemeScore {
  attendu: string
  detecte: string
  score: number
  type_confusion: string | null
  start_ms: number
  end_ms: number
}

export interface SyllableScore {
  hanzi: string
  pinyin: string
  traduction: string
  ton_attendu: number
  ton_detecte: number
  start_s: number
  end_s: number
  scores: {
    global: number
    ton: number
    initiale: number
    finale: number
  }
  f0_attendu: number[]
  f0_produit: number[]
  erreur: string | null
  phonemes: PhonemeScore[]
}

export interface ScoringResult {
  session_id: string
  mode: 'guidé' | 'libre'
  enonce: string
  traduction: string
  duration_total_s: number
  score_global: number
  score_fluidite: number
  score_rythme: number
  debit_mots_par_min: number
  syllabes: SyllableScore[]
}

export interface TtsSegment {
  segment_id: string
  lang: string
  audio_b64: string
}
