"""LaTeX tables of the paper from the outputs of evaluation.evaluate and the run directories.

    python -m evaluation.tables main  --results results
    python -m evaluation.tables dense --results results
    python -m evaluation.tables tau   --runs 0.5=<run dir> 0.8=<run dir> ... --timing loss_time.json

Values are means over the units of a case set, then mean +- sd over models (seeds, and
folds for MMWHS). A star marks a competitor that sparse BM beats significantly (one-sided
paired t-test over models, p < 0.05); bold is the best arm. The training time is the median
training time per epoch (without validation), relative to Dice+CE.
"""
import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

ARM_TEX = {"dice_ce": "Dice+CE", "cldice": "clDice", "skeleton_recall": "Skel.\\ Recall",
           "warping": "warping$^{\\dagger}$", "dmt": "DMT$^{\\dagger}$", "topograph": "Topograph",
           "dense_bm": "dense BM", "sparse_bm": "\\textbf{sparse BM (ours)}"}
DATASETS = [("atm", "ATM'26", "airway lumen", ["dice_ce", "cldice", "skeleton_recall", "warping", "sparse_bm"]),
            ("nisb_b", "NISB-B", "cell interfaces", ["dice_ce", "cldice", "skeleton_recall", "warping", "sparse_bm"]),
            ("bratsmets", "BraTS-METS", "NETC $\\cup$ ET", ["dice_ce", "cldice", "skeleton_recall", "warping", "sparse_bm"]),
            ("mmwhs_ct", "MMWHS", "LV myocardium", ["dice_ce", "cldice", "skeleton_recall", "warping", "sparse_bm"]),
            ("fives", "FIVES", "retinal vessels", ["dice_ce", "cldice", "skeleton_recall", "warping", "dmt", "topograph", "sparse_bm"]),
            ("acdc2d", "ACDC", "LV myocardium", ["dice_ce", "cldice", "skeleton_recall", "warping", "dmt", "topograph", "sparse_bm"])]
COLUMNS = [("dice_hard", "Dice$\\uparrow$", "higher"), ("bm_error", "BM err.$\\downarrow$", "lower"),
           ("cldice", "clDice$\\uparrow$", "higher"), ("voi", "VOI$\\downarrow$", "lower"),
           ("nsd", "NSD$\\uparrow$", "higher")]
VOI_KEY = {"nisb_b": "voi_bg"}   # the objects of NISB-B are the regions between the boundaries


# --------------------------------------------------------------------------- loading

def load_models(results, dataset, split="test"):
    """{(arm, seed, fold): json} of a dataset."""
    out = {}
    for path in sorted(glob.glob(os.path.join(results, dataset, split, "*.json"))):
        m = json.load(open(path))
        out[(m["arm"], m["seed"], m.get("fold"))] = m
    return out


def model_value(m, key):
    """Mean over units of the per-case values; Dice ignores cases with an empty label."""
    per_unit = defaultdict(list)
    for r in m["rows"]:
        v = r.get(key)
        if v is None or (isinstance(v, float) and np.isnan(v)) or (key == "dice_hard" and r["gt_empty"]):
            continue
        per_unit[r["unit"]].append(float(v))
    return float(np.mean([np.mean(v) for v in per_unit.values()]))


def train_time(m):
    """Median training time per epoch of the run the model checkpoint belongs to."""
    history = Path(m["model"]["checkpoint"]).parent / "history.jsonl"
    times = [json.loads(line).get("time/train") for line in open(history)]
    return float(np.median([t for t in times if t is not None]))


# ------------------------------------------------------------------------ statistics

def analyse(models, arms, voi_key):
    res = {}
    for arm in arms:
        keys = sorted(k for k in models if k[0] == arm)
        if not keys:
            continue
        res[arm] = {"keys": keys}
        for col, *_ in COLUMNS:
            key = voi_key if col == "voi" else col
            res[arm][col] = {k[1:]: model_value(models[k], key) for k in keys}
        try:
            res[arm]["time"] = float(np.mean([train_time(models[k]) for k in keys]))
        except (FileNotFoundError, KeyError, ValueError):
            res[arm]["time"] = None
    return res


def better_than(res, arm, col, ref="sparse_bm", alpha=0.05):
    """sparse BM significantly better than `arm` (one-sided paired t over models)."""
    if arm == ref or ref not in res:
        return False
    better = dict((c, b) for c, _, b in COLUMNS)[col]
    a, b = res[ref][col], res[arm][col]
    common = sorted(set(a) & set(b))
    if len(common) < 2:
        return False
    x, y = np.array([a[k] for k in common]), np.array([b[k] for k in common])
    if np.allclose(x, y):
        return False
    p = stats.ttest_rel(x, y, alternative="less" if better == "lower" else "greater").pvalue
    return bool(p < alpha)


# ------------------------------------------------------------------------ formatting

def fmt(m, s, col):
    if col == "dice_hard":
        out = f"{m:.4f}", f"{s:.4f}"
    elif col in ("cldice", "nsd"):
        out = f"{m:.3f}", f"{s:.3f}"
    elif col == "voi":
        nd = 4 if abs(m) < 0.1 else 3
        out = f"{m:.{nd}f}", f"{s:.{nd}f}"
    elif abs(m) >= 10000:
        out = f"{m / 1000:.1f}k", (f"{s / 1000:.1f}k" if s >= 1000 else f"{s:.0f}")
    elif abs(m) >= 100:
        out = f"{m:.0f}", f"{s:.0f}"
    elif abs(m) >= 10:
        out = f"{m:.1f}", f"{s:.1f}"
    elif abs(m) >= 1:
        out = f"{m:.2f}", f"{s:.2f}"
    else:
        out = f"{m:.3f}", f"{s:.3f}"
    return tuple(x[1:] if x.startswith("0.") else x for x in out)


def rows_tex(res, arms, first_cell):
    lines = []
    present = [a for a in arms if a in res]
    if not present:
        return []
    best = {}
    for col, _, better in COLUMNS:
        means = {a: np.mean(list(res[a][col].values())) for a in present}
        target = (min if better == "lower" else max)(means.values())
        best[col] = {a for a, v in means.items() if fmt(v, 0, col)[0] == fmt(target, 0, col)[0]}
    base_time = res.get("dice_ce", {}).get("time")
    for i, arm in enumerate(present):
        cells = [first_cell(len(present)) if i == 0 else "", ARM_TEX[arm]]
        for col, *_ in COLUMNS:
            v = np.array(list(res[arm][col].values()))
            m, s = fmt(v.mean(), v.std(ddof=1) if v.size > 1 else 0.0, col)
            txt = f"\\textbf{{{m}}}" if arm in best[col] else m
            if v.size > 1:
                txt += f"{{\\tiny$\\pm${s}}}"
            if better_than(res, arm, col):
                txt += "$^{*}$"
            cells.append(txt)
        t = res[arm]["time"]
        cells.append(f"{t / base_time:.2f}$\\times$" if t and base_time else "--")
        lines.append(" & ".join(cells) + " \\\\")
    return lines


def table(caption, label, first_header, blocks):
    head = ["\\begin{table*}[t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{4.5pt}",
            f"\\caption{{{caption}}}", f"\\label{{{label}}}",
            "\\begin{tabular}{ll" + "r" * (len(COLUMNS) + 1) + "}", "\\toprule",
            f"{first_header} & loss & " + " & ".join(h for _, h, _ in COLUMNS) + " & Train.\\ time$\\downarrow$ \\\\"]
    body = []
    for block in blocks:
        body += ["\\midrule"] + block
    return "\n".join(head + body + ["\\bottomrule", "\\end{tabular}", "\\end{table*}", ""])


def multirow(n, lines):
    inner = "\\\\ ".join(lines)
    return f"\\multirow{{{n}}}{{*}}{{\\begin{{tabular}}[c]{{@{{}}l@{{}}}}{inner}\\end{{tabular}}}}"


# --------------------------------------------------------------------------- tables

def main_table(args):
    blocks = []
    for key, name, target, arms in DATASETS:
        models = load_models(args.results, key)
        if not models:
            continue
        res = analyse(models, arms, VOI_KEY.get(key, "voi_fg"))
        patch = "{\\times}".join(map(str, next(iter(models.values()))["model"]["patch_size"]))
        lines = [f"\\textit{{{name}}}", f"{{\\tiny {target}}}", f"{{\\tiny ${patch}$}}"]
        blocks.append(rows_tex(res, arms, lambda n, lines=lines: multirow(n, lines)))
    tex = table("Test-set results, mean$\\pm$sd over three seeds (MMWHS: three seeds on each of four folds). "
                "Bold: best loss of the dataset. "
                "$^{*}$: sparse BM is significantly better (paired $t$-test over seeds, $p<0.05$). "
                "$^{\\dagger}$: runtime-optimised version of the authors' code.",
                "tab:main", "dataset", blocks)
    Path(args.out, "main_table.tex").write_text(tex)
    print(tex)


def dense_table(args):
    models = load_models(args.results, "atm")
    blocks = []
    for patch, prefix, arms in (("$128^3$", "", ["dice_ce", "sparse_bm"]),
                                ("$64^3$", "p64_", ["dice_ce", "dmt", "dense_bm", "sparse_bm"])):
        sub = {(k[0][len(prefix):],) + k[1:]: m for k, m in models.items()
               if (k[0].startswith(prefix) if prefix else not k[0].startswith("p64_"))}
        res = analyse(sub, arms, "voi_fg")
        blocks.append(rows_tex(res, arms, lambda n, patch=patch: f"\\multirow{{{n}}}{{*}}{{{patch}}}"))
    tex = table("Comparison to PH-based losses with reduced patch size on ATM'26.", "tab:dense", "patch", blocks)
    Path(args.out, "atm_dense_comparison.tex").write_text(tex)
    print(tex)


def tau_table(args):
    from evaluation.select import tail_mean

    timing = json.load(open(args.timing))["summary"]
    rows = []
    for item in args.runs:
        tau, run = item.split("=", 1)
        t = timing[str(float(tau))]
        rows.append((float(tau), 100 * t["kept"], t["ms_per_sample"], t["sem"],
                     tail_mean(run, "val/bm_error"), tail_mean(run, "val/dice_hard")))
    best_bm, best_dice = min(r[4] for r in rows), max(r[5] for r in rows)
    lines = ["\\begin{table}[t]", "\\centering", "\\scriptsize", "\\setlength{\\tabcolsep}{4pt}",
             "\\caption{Kept fraction, runtime, validation BM error and Dice with varying $\\tau$ on ATM'26 (validation).}",
             "\\label{tab:tau}", "\\begin{tabular}{lrrrr}", "\\toprule",
             "$\\tau$ & kept (\\%) & runtime (ms)$\\downarrow$ & BM err.$\\downarrow$ & Dice$\\uparrow$ \\\\", "\\midrule"]
    for tau, kept, ms, sem, bm, dice in sorted(rows):
        bm_txt = f"{bm:.2f}" if f"{bm:.2f}" != f"{best_bm:.2f}" else f"\\textbf{{{bm:.2f}}}"
        d = f"{dice:.4f}"[1:]
        d_txt = d if f"{dice:.4f}" != f"{best_dice:.4f}" else f"\\textbf{{{d}}}"
        lines.append(f"{tau:g} & {kept:.2f} & {ms:.1f}{{\\tiny$\\pm${sem:.1f}}} & {bm_txt} & {d_txt} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    tex = "\n".join(lines)
    Path(args.out, "atm_tau_ablation.tex").write_text(tex)
    print(tex)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("table", choices=("main", "dense", "tau"))
    ap.add_argument("--results", default="results")
    ap.add_argument("--runs", nargs="*", default=[], metavar="TAU=RUN_DIR")
    ap.add_argument("--timing", default=None)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    {"main": main_table, "dense": dense_table, "tau": tau_table}[args.table](args)


if __name__ == "__main__":
    main()
