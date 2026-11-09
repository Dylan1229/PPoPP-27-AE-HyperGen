#!/usr/bin/env bash
# Record the exact package versions of every environment into env/requirements/<env>.txt.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/scripts/config.sh"
for e in cogvideo hunyuan wan vbench metrics cinescale freeswim; do
  [ -x "$ENV_ROOT/$e/bin/python" ] || continue
  { echo "# $e: $("$ENV_ROOT/$e/bin/python" --version 2>&1), frozen $(date -u +%Y-%m-%d) on $(hostname)"
    "${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}" pip freeze --python "$ENV_ROOT/$e/bin/python" 2>/dev/null \
      || "$ENV_ROOT/$e/bin/python" -m pip freeze; } \
    | sed "s#file://$ROOT/#./#" > "$ROOT/env/requirements/$e.txt" && echo "wrote env/requirements/$e.txt"
done
for r in Wan2.1 VBench CineScale FreeSwim; do
  [ -d "$THIRD_PARTY/$r/.git" ] && echo "$r $(git -C "$THIRD_PARTY/$r" rev-parse HEAD)"
done > "$ROOT/env/requirements/third_party_commits.txt"
