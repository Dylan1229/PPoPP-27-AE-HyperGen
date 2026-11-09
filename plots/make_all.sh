#!/usr/bin/env bash
# Regenerate every main-paper table and figure from a results directory. CPU only, ~1 minute.
#
#   bash plots/make_all.sh                         results/paper -> out/paper  (the paper's data)
#   bash plots/make_all.sh --results results/ae    your runs     -> out/ae
#
# Writes out/<name>/tables/*.tex (+ tables.tex, a compilable wrapper), out/<name>/figures/*.pdf,
# out/<name>/previews/*.png, and out/<name>/summary.md (key numbers next to the paper's).
# A generator whose inputs are absent (e.g. an experiment you have not run) is reported and skipped.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=../scripts/config.sh
source "$ROOT/scripts/config.sh"

RESULTS_DIR="$ROOT/results/paper"
while [ "$#" -gt 0 ]; do
  case "$1" in
    --results) RESULTS_DIR="$(cd "$2" && pwd)" || exit 1; shift ;;
    -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument $1" >&2; exit 1 ;;
  esac
  shift
done
export RESULTS_DIR
export OUT_DIR="${OUT_DIR:-$OUT_ROOT/$(basename "$RESULTS_DIR")}"
PAPER=0; [ "$RESULTS_DIR" = "$ROOT/results/paper" ] && PAPER=1
# A --quick run scores fewer prompts per Table 3 cell than the paper's 5 / 3; accept them.
[ "$PAPER" = 0 ] && export MIN_PROMPTS="${MIN_PROMPTS:-1}"

PY="${PY:-$METRICS_ENV/bin/python}"
"$PY" -c 'import matplotlib, numpy' 2>/dev/null || PY=python3
"$PY" -c 'import matplotlib, numpy' 2>/dev/null || { echo "need a Python with matplotlib and numpy (env/setup_envs.sh metrics)" >&2; exit 1; }

echo "results: $RESULTS_DIR"
echo "output:  $OUT_DIR"
ok=(); skipped=()
gen() {   # label script [args...]
  local label="$1"; shift
  echo; echo "---- $label"
  if "$PY" "$ROOT/plots/$1" "${@:2}"; then ok+=("$label"); else skipped+=("$label"); fi
}
gen "Table 3"   table3_quality.py
gen "Table 4"   table4_cache_speedup.py
gen "Table 5"   table5_cache_comparison.py
gen "Figure 11" fig11_e2e.py
gen "Figure 12" fig12_breakdown.py
if [ "$PAPER" = 1 ]; then
  gen "Figure 13" fig13_scalability.py --paper
  gen "Figure 14" fig14_rebalance.py --paper
else
  gen "Figure 13" fig13_scalability.py
  gen "Figure 14" fig14_rebalance.py
fi

# A standalone LaTeX file that typesets every generated table (pdflatex tables.tex).
T="$OUT_DIR/tables"
if ls "$T"/*.tex >/dev/null 2>&1; then
  {
    echo '\documentclass{article}'
    echo '\usepackage[margin=1.5cm]{geometry}\usepackage{booktabs,multirow,graphicx}\usepackage[table]{xcolor}'
    echo '\newcommand{\Mname}{\textsc{HyperGen}}'
    echo '\begin{document}'
    for f in "$T"/table*.tex; do echo "\\input{$(basename "$f")}"; echo '\clearpage'; done
    echo '\end{document}'
  } > "$T/tables.tex"
fi

"$PY" "$ROOT/plots/compare.py" --results "$RESULTS_DIR" --paper "$ROOT/results/paper" > "$OUT_DIR/summary.md" 2>/dev/null \
  && echo && echo "---- summary: $OUT_DIR/summary.md" && cat "$OUT_DIR/summary.md"

echo
echo "generated: ${ok[*]:-none}"
[ "${#skipped[@]}" -gt 0 ] && echo "skipped (inputs missing or incomplete, see messages above): ${skipped[*]}"
echo "tables:  $OUT_DIR/tables    figures: $OUT_DIR/figures    previews: $OUT_DIR/previews"
exit 0
