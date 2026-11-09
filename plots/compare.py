"""Print the key numbers of a results directory next to the paper's, as Markdown.

    python plots/compare.py --results results/ae --paper results/paper

Only cells present in the evaluated directory are listed, so a --quick run gives a short report.
Absolute latencies depend on the machine (GPU clocks, interconnect, storage); compare speedups and
retention rather than seconds. VBench of a single prompt is noisy; Table 3 averages 5 (2K) or 3 (4K)
prompts per cell.
"""
import argparse
import csv
import os
import statistics


def load(path, keep=lambda r: True):
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [r for r in csv.DictReader(f, delimiter="\t") if keep(r)]


def f(x, spec=".1f"):
    try:
        return format(float(x), spec)
    except (TypeError, ValueError):
        return "--"


def ratio(a, b):
    try:
        return f"{float(a) / float(b):.2f}x"
    except (TypeError, ValueError, ZeroDivisionError):
        return "--"


def t7(res_dir, block):
    out = {}
    for r in load(os.path.join(res_dir, "T7_figures", "timings.tsv"),
                  lambda r: r["block"] == block and r["rc"] == "0" and r.get("stage2_s")):
        out[(r["config"], r["res"], r["variant"], r["ngpu"])] = r      # last wins
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--paper", required=True)
    a = ap.parse_args()
    R, P = a.results, a.paper
    same = os.path.realpath(R) == os.path.realpath(P)
    print(f"# Key numbers: `{os.path.basename(os.path.normpath(R))}`" + ("" if same else " vs. the paper"))
    print()
    print("Latencies are Stage 2 unless stated; absolute seconds depend on the machine, so compare")
    print("speedups and retention. Single-prompt VBench is noisy.")

    # ---- Table 3
    q = {(r["config"], r["res"], r["variant"]): r for r in load(os.path.join(R, "T1_quality", "vbench.tsv"))}
    qp = {(r["config"], r["res"], r["variant"]): r for r in load(os.path.join(P, "T1_quality", "vbench.tsv"))}
    if q:
        print("\n## Table 3 - VBench average (%)\n")
        print("| backbone | setting | prompts | this run | paper |\n|---|---|---|---|---|")
        for k, r in sorted(q.items()):
            print(f"| {k[0]} | {k[1]} {k[2]} | {r['n_videos']} | {r['avg']} | {qp.get(k, {}).get('avg', '--')} |")

    # ---- Table 4
    t4, t4p = t7(R, "fig11"), t7(P, "fig11")
    qual = {(r["config"], r["res"], r["ngpu"]): r for r in load(os.path.join(R, "T7_figures", "fig11_quality.tsv"))}
    qualp = {(r["config"], r["res"], r["ngpu"]): r for r in load(os.path.join(P, "T7_figures", "fig11_quality.tsv"))}
    cells = sorted({(c, res, n) for (c, res, v, n) in t4})
    if cells:
        print("\n## Table 4 - cache: Stage-2 latency w/ / w/o, speedup, retention\n")
        print("| backbone | setting | this run: s (w/ / w/o) | speedup | PSNR | LPIPS | paper: speedup | PSNR | LPIPS |")
        print("|---|---|---|---|---|---|---|---|---|")
        for c, res, n in cells:
            on, off = t4.get((c, res, "cache", n), {}), t4.get((c, res, "nocache", n), {})
            pon, poff = t4p.get((c, res, "cache", n), {}), t4p.get((c, res, "nocache", n), {})
            m, mp = qual.get((c, res, n), {}), qualp.get((c, res, n), {})
            print(f"| {c} | {res} {n} GPU | {f(on.get('stage2_s'), '.0f')} / {f(off.get('stage2_s'), '.0f')} "
                  f"| {ratio(off.get('stage2_s'), on.get('stage2_s'))} | {f(m.get('psnr'), '.2f')} | {f(m.get('lpips'), '.3f')} "
                  f"| {ratio(poff.get('stage2_s'), pon.get('stage2_s'))} | {f(mp.get('psnr'), '.2f')} | {f(mp.get('lpips'), '.3f')} |")

    # ---- Table 5
    t5 = load(os.path.join(R, "T2_baselines", "quality.tsv"))
    t5p = {(r["method"], r["thresh"]): r for r in load(os.path.join(P, "T2_baselines", "quality.tsv")) if r.get("stage2_s")}
    if t5:
        print("\n## Table 5 - HyperGen cache vs TeaCache (CogVideoX 2K, 1 GPU)\n")
        print("| method | threshold | this run: speedup | LPIPS | skip | paper: speedup | LPIPS | skip |\n|---|---|---|---|---|---|---|---|")
        for r in t5:
            p = t5p.get((r["method"], r["thresh"]), {})
            print(f"| {r['method']} | {r['thresh']} | {f(r['speedup'], '.2f')}x | {f(r['lpips'], '.3f')} | {f(r['skip_rate'], '.2f')} "
                  f"| {f(p.get('speedup'), '.2f')}x | {f(p.get('lpips'), '.3f')} | {f(p.get('skip_rate'), '.2f')} |")

    # ---- Figure 11
    e, ep = t7(R, "fig10"), t7(P, "fig10")
    s1 = {r["config"]: r["stage1_s"] for r in load(os.path.join(R, "T8_stage1", "timings.tsv")) if r.get("stage1_s")}
    s1p = {r["config"]: r["stage1_s"] for r in load(os.path.join(P, "T8_stage1", "timings.tsv")) if r.get("stage1_s")}
    groups = sorted({(c, res) for (c, res, v, n) in e})
    if groups:
        print("\n## Figure 11 - end-to-end latency (Stage 1 + upsampling + Stage 2)\n")
        print("| backbone | res | this run: origin / +par / +par+cache (s) | speedup | paper speedup |\n|---|---|---|---|---|")

        def e2e(d, s, c, res, v):
            r = next((x for (cc, rr, vv, n), x in d.items() if (cc, rr, vv) == (c, res, v)), None)
            if not r or c not in s:
                return None
            return float(s[c]) + float(r.get("upscale_s") or 0) + float(r["stage2_s"])
        for c, res in groups:
            v = [e2e(e, s1, c, res, k) for k in ("origin", "parallel", "parcache")]
            vp = [e2e(ep, s1p, c, res, k) for k in ("origin", "parallel", "parcache")]
            print(f"| {c} | {res} | {' / '.join(f(x, '.0f') for x in v)} | {ratio(v[0], v[2]) if None not in (v[0], v[2]) else '--'} "
                  f"| {ratio(vp[0], vp[2]) if None not in (vp[0], vp[2]) else '--'} |")

    # ---- Figure 12
    b, bp = t7(R, "fig12"), t7(P, "fig12")
    if b:
        print("\n## Figure 12 - CogVideoX 4K, Stage-2 latency as mechanisms are added\n")
        print("| step | GPUs | this run (s) | skip | paper (s) |\n|---|---|---|---|---|")
        for step in ("baseline", "tilepar", "cache", "regionaware", "rebalance"):
            r = next((x for (c, res, v, n), x in b.items() if v == step), None)
            p = next((x for (c, res, v, n), x in bp.items() if v == step), {})
            if r:
                print(f"| {step} | {r['ngpu']} | {f(r['stage2_s'], '.0f')} | {r.get('skip_rate') or '-'} | {f(p.get('stage2_s'), '.0f')} |")

    # ---- Figure 13 / 14
    s13 = t7(R, "fig13")
    if s13:
        print("\n## Figure 13 - tile-parallel scalability (no cache), speedup over 1 GPU\n")
        print("| backbone | res | GPUs | this run (s) | speedup |\n|---|---|---|---|---|")
        for (c, res, v, n), r in sorted(s13.items(), key=lambda kv: (kv[0][0], kv[0][1], int(kv[0][3]))):
            one = s13.get((c, res, v, "1"), {}).get("stage2_s")
            print(f"| {c} | {res} | {n} | {f(r['stage2_s'], '.0f')} | {ratio(one, r['stage2_s']) if one else '--'} |")
    s14 = t7(R, "fig14")
    if s14:
        print("\n## Figure 14 - workload rebalance (4K, cache on)\n")
        print("| backbone | GPUs | this run: w/o / w/ (s) | speedup |\n|---|---|---|---|")
        for c, n in sorted({(c, n) for (c, res, v, n) in s14}, key=lambda x: (x[0], int(x[1]))):
            a0 = s14.get((c, "4k", "cache_reb0", n), {}).get("stage2_s")
            a1 = s14.get((c, "4k", "cache_reb1", n), {}).get("stage2_s")
            print(f"| {c} | {n} | {f(a0, '.0f')} / {f(a1, '.0f')} | {ratio(a0, a1) if a0 and a1 else '--'} |")


if __name__ == "__main__":
    main()
