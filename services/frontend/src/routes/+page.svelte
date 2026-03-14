<div class="home">
  <h1 class="title">Mandarin Coach</h1>
  <p class="sub">Prononciation du mandarin avec scoring temps réel</p>

  <a href="/session" class="cta">
    Démarrer une session
    <span class="arrow">→</span>
  </a>
  <a href="/training" class="cta cta-sec">
    Monitoring entraînement
    <span class="arrow">→</span>
  </a>
</div>

<style>
  .home { display: flex; flex-direction: column; align-items: center; justify-content: center; min-height: 100vh; gap: 16px }
  .title { font-family: 'Noto Serif SC', serif; font-size: 28px; color: #fff; font-weight: 600; letter-spacing: .04em }
  .sub { font-size: 13px; color: var(--muted) }
  .cta {
    margin-top: 20px;
    display: inline-flex; align-items: center; gap: 8px;
    padding: 12px 28px;
    background: rgba(91,141,238,.12);
    border: 1px solid rgba(91,141,238,.25);
    color: var(--accent);
    font-size: 14px; font-weight: 500;
    border-radius: 10px;
    text-decoration: none;
    transition: background .15s;
  }
  .cta:hover { background: rgba(91,141,238,.22) }
  .cta-sec { background: rgba(255,255,255,.04); border-color: rgba(255,255,255,.10); color: var(--muted); font-size: 12px; margin-top: 0 }
  .cta-sec:hover { background: rgba(255,255,255,.08) }
  .arrow { font-size: 16px }
</style>
