#!/usr/bin/env bash
set -euo pipefail

# ── Couleurs ──────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
BOLD='\033[1m'
RST='\033[0m'

ok()   { printf "${GREEN}  ✓ %s${RST}\n" "$*"; }
fail() { printf "${RED}  ✗ %s${RST}\n" "$*"; exit 1; }
step() { printf "${BLUE}  → %s${RST}\n" "$*"; }

# ── Config par machine ────────────────────────────────────
declare -A REMOTE_DIR=(
  [proliant]="~/container/coach_mandarin"
  [ai]="~/containers/coach_mandarin"
)

declare -A COMPOSE_FILE=(
  [proliant]="docker-compose.yml"
  [ai]="docker-compose.ai.yml"
)

# ── Fichiers à déployer par machine ──────────────────────
# Seuls ces fichiers/dossiers sont envoyés — tout le reste est ignoré.
# Utilise les filter rules rsync : "+ " = include, "- " = exclude.

proliant_filters() {
  cat <<'RULES'
+ /docker-compose.yml
+ /.env.example
+ /services/
+ /services/frontend/
+ /services/frontend/**
+ /services/api/
+ /services/api/**
+ /infra/
+ /infra/**
- /services/*
- *
RULES
}

ai_filters() {
  cat <<'RULES'
+ /docker-compose.ai.yml
+ /.env.example
+ /services/
+ /services/scorer/
+ /services/scorer/**
+ /services/whisper/
+ /services/whisper/**
+ /services/tts/
+ /services/tts/**
+ /shared/
+ /shared/**
- /services/*
- *
RULES
}

# Exclusions communes (artefacts à ne jamais copier, même dans les dossiers inclus)
COMMON_EXCLUDES=(
  "node_modules/"
  ".svelte-kit/"
  "build/"
  "__pycache__/"
  "*.pyc"
  "*.pt"
  "*.ckpt"
  ".env"
)

# ── Usage ─────────────────────────────────────────────────
usage() {
  cat <<EOF
${BOLD}Usage:${RST} ./deploy.sh <target> [options]

  target:
    proliant    déploie frontend + api sur proliant
    ai          déploie scorer + whisper + tts + ollama sur ai
    all         déploie sur les deux machines

  options:
    --dry-run   affiche ce qui serait transféré sans rien envoyer
    --backup    après déploiement, commit + push sur GitHub
EOF
  exit 1
}

# ── Parse arguments ───────────────────────────────────────
[[ $# -lt 1 ]] && usage

TARGET="$1"
BACKUP=false
DRY_RUN=false

shift
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backup)  BACKUP=true ;;
    --dry-run) DRY_RUN=true ;;
    *) fail "Option inconnue : $1" ;;
  esac
  shift
done

case "$TARGET" in
  proliant) MACHINES=(proliant) ;;
  ai)       MACHINES=(ai) ;;
  all)      MACHINES=(proliant ai) ;;
  *)        fail "Cible inconnue : $TARGET (proliant | ai | all)" ;;
esac

# ── Vérifie qu'on est à la racine du monorepo ────────────
[[ -f ".github/copilot-instructions.md" ]] || fail "Lancer depuis la racine du monorepo"

# ── Déploiement ──────────────────────────────────────────
deploy_machine() {
  local machine="$1"
  local remote_dir="${REMOTE_DIR[$machine]}"
  local compose="${COMPOSE_FILE[$machine]}"

  printf "\n${BOLD}── %s ──${RST}\n" "$machine"

  # Vérifier que le compose file existe localement
  [[ -f "$compose" ]] || fail "$compose introuvable — déploiement $machine impossible"

  # Générer les filter rules dans un fichier temporaire
  local filter_file
  filter_file=$(mktemp)
  trap "rm -f $filter_file" RETURN

  case "$machine" in
    proliant) proliant_filters > "$filter_file" ;;
    ai)       ai_filters > "$filter_file" ;;
  esac

  # Construire les flags --exclude communs
  local exclude_flags=()
  for ex in "${COMMON_EXCLUDES[@]}"; do
    exclude_flags+=(--exclude "$ex")
  done

  local rsync_flags=(-azP --delete --filter="merge $filter_file" "${exclude_flags[@]}")

  if [[ "$DRY_RUN" == true ]]; then
    rsync_flags+=(--dry-run)
    step "dry-run rsync vers ${machine}:${remote_dir}"
  else
    step "rsync vers ${machine}:${remote_dir}"
  fi

  rsync "${rsync_flags[@]}" \
    ./ "${machine}:${remote_dir}/" \
    || fail "rsync vers $machine a échoué"
  ok "fichiers synchronisés"

  rm -f "$filter_file"

  if [[ "$DRY_RUN" == true ]]; then
    ok "dry-run terminé — aucun fichier transféré"
    return
  fi

  step "docker compose up -d --build sur $machine"
  ssh "$machine" "cd ${remote_dir} && docker compose -f ${compose} up -d --build" \
    || fail "docker compose sur $machine a échoué"
  ok "containers démarrés"
}

for m in "${MACHINES[@]}"; do
  deploy_machine "$m"
done

# ── Backup GitHub ────────────────────────────────────────
if [[ "$BACKUP" == true ]]; then
  printf "\n${BOLD}── backup GitHub ──${RST}\n"

  step "git add + commit"
  git add -A
  git diff --cached --quiet && {
    ok "rien à committer"
  } || {
    git commit -m "backup: $(date '+%Y-%m-%d %H:%M')" \
      || fail "git commit a échoué"
    ok "commit créé"
  }

  step "git push origin master"
  git push origin master || fail "git push a échoué"
  ok "poussé sur GitHub"
fi

printf "\n${GREEN}${BOLD}  Déploiement terminé.${RST}\n\n"
