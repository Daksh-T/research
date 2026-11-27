#!/usr/bin/env bash
# Compile all 3 economics manuscripts using either a project-local .latex/bin/tectonic
# binary or a system-installed `tectonic` on PATH.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -x "$ROOT/.latex/bin/tectonic" ]]; then
  export HOME="$ROOT/.latex/home"
  export XDG_CACHE_HOME="$ROOT/.latex/cache"
  mkdir -p "$HOME" "$XDG_CACHE_HOME"
  TECTONIC="$ROOT/.latex/bin/tectonic"
elif command -v tectonic >/dev/null 2>&1; then
  TECTONIC="$(command -v tectonic)"
else
  echo "ERROR: tectonic not found on PATH or at $ROOT/.latex/bin/tectonic" >&2
  echo "Install tectonic (https://tectonic-typesetting.github.io/) or compile with pdflatex + bibtex." >&2
  exit 1
fi

compile_one() {
  local dir="$1"
  local tex="$2"
  echo "=== Compiling $dir/$tex ==="
  (cd "$ROOT/$dir" && "$TECTONIC" "$tex")
}

compile_one "papers/paper1_genai_immigrants" "paper1_manuscript.tex"
compile_one "papers/paper2_intermarriage_comment" "paper2_manuscript.tex"
compile_one "papers/paper3_enclaves_wfh" "paper3_manuscript.tex"

echo "=== All 3 manuscripts compiled successfully ==="
ls -lh \
  "$ROOT/papers/paper1_genai_immigrants/paper1_manuscript.pdf" \
  "$ROOT/papers/paper2_intermarriage_comment/paper2_manuscript.pdf" \
  "$ROOT/papers/paper3_enclaves_wfh/paper3_manuscript.pdf"
