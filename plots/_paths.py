"""Where the generators read data from and write tables / figures to.

    RESULTS_DIR   a results directory with the measurement layout (T1_quality/, T7_figures/, ...).
                  Default: results/paper, the paper's data.
    OUT_DIR       where tables/, figures/ and previews/ are written.
                  Default: out/<basename of RESULTS_DIR>, e.g. out/paper or out/ae.

plots/make_all.sh sets both; the generators can also be run one by one with the variables exported.
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = Path(os.environ.get("RESULTS_DIR", ROOT / "results" / "paper")).resolve()
OUT = Path(os.environ.get("OUT_DIR", ROOT / "out" / RES.name)).resolve()
TABLES = OUT / "tables"
FIGS = OUT / "figures"
PREV = OUT / "previews"
for _d in (TABLES, FIGS, PREV):
    _d.mkdir(parents=True, exist_ok=True)
