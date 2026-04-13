"""Produce a compact markdown digest of a W&B run for LLM consumption.

Reads scalar metric history via wandb.Api, aggregates server-side where
possible, and writes a digest (< 200 lines) suitable for feeding to the
`analyst` subagent. Never dumps raw 10k-step loss curves.

Usage:
    python scripts/summarize_run.py <run_id>                 # final digest
    python scripts/summarize_run.py <run_id> --partial       # mid-run digest
    python scripts/summarize_run.py <run_id> --baseline <id> # side-by-side diff
    python scripts/summarize_run.py <run_id> --output <path> # write to file

Run ID can be just the W&B id (e.g. `pirxiw5h`) or `entity/project/id`.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import wandb
except ImportError:
    print("wandb is not installed in this environment. Activate `nnunet`.", file=sys.stderr)
    sys.exit(2)


DEFAULT_PROJECT = "3D-Segmentation"

PRIMARY_METRICS = ["val/dice_hard", "val/loss", "train/loss_iter"]
LOSS_COMPONENT_PREFIX = "train/loss_"
PERF_METRICS = ["train/epoch_time", "train/iter_time", "train/learning_rate"]
SYSTEM_METRICS = ["gpu.process.0.gpu", "gpu.process.0.memoryAllocatedBytes"]


N_BINS = 10


@dataclass
class MetricSummary:
    name: str
    first: float
    last: float
    vmin: float
    vmax: float
    step_min: int
    step_max: int
    slope_tail: float  # slope over last 20 % of samples
    snr_tail: float    # mean/std over last 20 %
    n_samples: int
    n_nonfinite: int
    still_decreasing: bool
    shape: str                     # trajectory classification
    bins: list[tuple[float, float, float]]  # (step_lo, step_hi, mean); mean=nan if bin empty/all nonfinite
    nonfinite_segments: list[tuple[float, float]]  # inclusive (start_step, end_step) ranges where value was nan/inf
    jumps: list[tuple[float, float]]  # (step, signed_magnitude) for step-to-step changes > 4σ of local diff


def _fit_slope(xs: np.ndarray, ys: np.ndarray) -> float:
    if len(xs) < 2:
        return 0.0
    xs = xs.astype(float)
    ys = ys.astype(float)
    xm, ym = xs.mean(), ys.mean()
    num = ((xs - xm) * (ys - ym)).sum()
    den = ((xs - xm) ** 2).sum()
    return float(num / den) if den > 0 else 0.0


def _bin_series(steps_all: np.ndarray, values_all: np.ndarray, n_bins: int = N_BINS) -> list[tuple[float, float, float]]:
    """Split steps into n_bins equal-width intervals; report the mean of finite values in each.
    Emits NaN for bins with no finite values so the caller can visualize 'dead zones'.
    """
    if len(values_all) == 0:
        return []
    s_min, s_max = float(steps_all.min()), float(steps_all.max())
    if s_max <= s_min:
        return [(s_min, s_max, float(np.nanmean(values_all)))]
    edges = np.linspace(s_min, s_max, n_bins + 1)
    bins: list[tuple[float, float, float]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (steps_all >= lo) & (steps_all <= hi)
        else:
            mask = (steps_all >= lo) & (steps_all < hi)
        vals = values_all[mask]
        finite = vals[np.isfinite(vals)]
        mean_val = float(np.mean(finite)) if len(finite) > 0 else float("nan")
        bins.append((float(lo), float(hi), mean_val))
    return bins


def _nonfinite_segments(steps_all: np.ndarray, values_all: np.ndarray) -> list[tuple[float, float]]:
    """Contiguous ranges of steps where value was nan/inf."""
    bad = ~np.isfinite(values_all)
    segments: list[tuple[float, float]] = []
    start: Optional[float] = None
    for i, is_bad in enumerate(bad):
        if is_bad and start is None:
            start = float(steps_all[i])
        elif not is_bad and start is not None:
            segments.append((start, float(steps_all[i - 1])))
            start = None
    if start is not None:
        segments.append((start, float(steps_all[-1])))
    return segments


def _detect_jumps(steps: np.ndarray, values: np.ndarray, k_sigma: float = 4.0, max_jumps: int = 5) -> list[tuple[float, float]]:
    """Step-to-step changes > k_sigma * std(all_diffs). Returns up to max_jumps largest."""
    if len(values) < 4:
        return []
    diffs = np.diff(values)
    finite_diffs = diffs[np.isfinite(diffs)]
    if len(finite_diffs) < 2:
        return []
    sigma = float(np.std(finite_diffs))
    if sigma == 0:
        return []
    outliers = []
    for i, d in enumerate(diffs):
        if np.isfinite(d) and abs(d) > k_sigma * sigma:
            outliers.append((float(steps[i + 1]), float(d)))
    outliers.sort(key=lambda x: abs(x[1]), reverse=True)
    return outliers[:max_jumps]


def _classify_shape(values_finite: np.ndarray) -> str:
    """Text classification of trajectory shape from finite values (NaN stripped)."""
    n = len(values_finite)
    if n < 3:
        return "too-short"
    peak_idx = int(np.argmax(values_finite))
    trough_idx = int(np.argmin(values_finite))
    start, end = float(values_finite[0]), float(values_finite[-1])
    peak, trough = float(values_finite[peak_idx]), float(values_finite[trough_idx])

    # Rise-then-collapse: interior peak, final ≤ 50 % of peak (signature of divergence after learning).
    if 0.2 * n <= peak_idx <= 0.8 * n and end <= peak * 0.5 and peak > start * 1.2:
        return f"rise-then-collapse (peak {peak:.3g} @ bin {int(10 * peak_idx / n) + 1}/10; end {end:.3g} = {100 * end / max(peak, 1e-12):.0f}% of peak)"

    # Fall-then-recovery: interior trough, final ≥ 2× trough.
    if 0.2 * n <= trough_idx <= 0.8 * n and end >= trough * 2 and start > trough * 1.2:
        return f"fall-then-recovery (trough {trough:.3g} @ bin {int(10 * trough_idx / n) + 1}/10; end {end:.3g})"

    # Overall linear fit + noise check.
    slope = _fit_slope(np.arange(n, dtype=float), values_finite.astype(float))
    tail_n = max(2, n // 5)
    tail = values_finite[-tail_n:]
    tail_mean = abs(float(np.mean(tail)))
    tail_std = float(np.std(tail))
    noisy = tail_std / max(tail_mean, 1e-12) > 0.2

    if slope < 0 and not noisy:
        return "monotonic-decreasing"
    if slope > 0 and not noisy:
        return "monotonic-increasing"
    if slope < -1e-12:
        return "noisy-decreasing"
    if slope > 1e-12:
        return "noisy-increasing"
    return "oscillating / flat"


def _summarize_series(name: str, steps: np.ndarray, values: np.ndarray) -> Optional[MetricSummary]:
    # W&B can return object-dtype arrays (e.g., mixed numeric/None). Coerce.
    try:
        values = np.asarray(values, dtype=float)
    except (TypeError, ValueError):
        return None
    try:
        steps = np.asarray(steps, dtype=float)
    except (TypeError, ValueError):
        steps = np.arange(len(values), dtype=float)

    n_total = len(values)
    if n_total == 0:
        return None

    # Detect non-finite segments on the raw series (before filtering, so we can report them).
    nonfinite_segs = _nonfinite_segments(steps, values)
    n_nonfinite = int(np.sum(~np.isfinite(values)))

    # Bins computed on the raw series so NaN-dominated bins are reported as NaN.
    bins = _bin_series(steps, values, n_bins=N_BINS)

    # Jumps computed on the raw series, ignoring NaN diffs.
    jumps = _detect_jumps(steps, values)

    # Remaining stats computed on the finite subset.
    mask = np.isfinite(values) & np.isfinite(steps)
    steps_f = steps[mask]
    values_f = values[mask]
    if len(values_f) == 0:
        return MetricSummary(
            name=name, first=float("nan"), last=float("nan"),
            vmin=float("nan"), vmax=float("nan"),
            step_min=-1, step_max=-1,
            slope_tail=0.0, snr_tail=0.0,
            n_samples=n_total, n_nonfinite=n_nonfinite,
            still_decreasing=False,
            shape="all-nonfinite",
            bins=bins, nonfinite_segments=nonfinite_segs, jumps=jumps,
        )

    tail_n = max(2, len(values_f) // 5)
    tail_vals = values_f[-tail_n:]
    tail_steps = steps_f[-tail_n:]
    tail_mean = float(np.mean(tail_vals))
    tail_std = float(np.std(tail_vals))
    snr = tail_mean / tail_std if tail_std > 0 else math.inf
    slope = _fit_slope(tail_steps, tail_vals)
    still_dec = slope < 0
    shape = _classify_shape(values_f)

    # If the tail of the raw series is all nonfinite, override shape to surface that.
    if len(values) > 0 and not np.isfinite(values[-1]):
        trailing_bad = 0
        for v in values[::-1]:
            if np.isfinite(v):
                break
            trailing_bad += 1
        shape = f"nan-hit (last {trailing_bad}/{n_total} samples nonfinite; prior shape: {shape})"

    return MetricSummary(
        name=name,
        first=float(values_f[0]),
        last=float(values_f[-1]),
        vmin=float(np.min(values_f)),
        vmax=float(np.max(values_f)),
        step_min=int(steps_f[np.argmin(values_f)]),
        step_max=int(steps_f[np.argmax(values_f)]),
        slope_tail=slope,
        snr_tail=snr,
        n_samples=n_total,
        n_nonfinite=n_nonfinite,
        still_decreasing=still_dec,
        shape=shape,
        bins=bins,
        nonfinite_segments=nonfinite_segs,
        jumps=jumps,
    )


def _classify_convergence(train_loss: Optional[MetricSummary]) -> str:
    if train_loss is None:
        return "no-data"
    if train_loss.shape.startswith("nan-hit") or train_loss.shape == "all-nonfinite":
        return "NaN-hit"
    if train_loss.shape.startswith("rise-then-collapse"):
        return "diverged-after-learning"
    if not math.isfinite(train_loss.last):
        return "NaN-hit"
    if train_loss.last > train_loss.first * 1.5:
        return "diverged"
    if train_loss.still_decreasing and abs(train_loss.slope_tail) > 1e-6:
        return "still-decreasing"
    return "flat"


def _resolve_run(run_id: str) -> "wandb.apis.public.Run":
    api = wandb.Api()
    if run_id.count("/") == 2:
        return api.run(run_id)
    # assume default entity (set via env) + default project
    try:
        return api.run(f"{DEFAULT_PROJECT}/{run_id}")
    except Exception:
        # try with env-default entity
        import os
        entity = os.environ.get("WANDB_ENTITY", "")
        if entity:
            return api.run(f"{entity}/{DEFAULT_PROJECT}/{run_id}")
        raise


def _fetch_scalars(run, keys: list[str], samples: int = 500) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {key: (steps, values)} using server-side sampling."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    try:
        hist = run.history(keys=keys, samples=samples, pandas=True, x_axis="_step")
    except Exception as e:
        print(f"# warning: run.history failed ({e}); falling back to summary only", file=sys.stderr)
        return out
    for k in keys:
        if k not in hist.columns:
            continue
        sub = hist[["_step", k]].dropna()
        if sub.empty:
            continue
        out[k] = (sub["_step"].to_numpy(), sub[k].to_numpy())
    return out


def _fetch_loss_components(run, samples: int = 500) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Discover any `train/loss_*` components logged by this run."""
    keys = [k for k in run.summary.keys() if k.startswith(LOSS_COMPONENT_PREFIX) and k != "train/loss_iter"]
    return _fetch_scalars(run, keys, samples=samples) if keys else {}


def _fetch_system(run) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    try:
        sys_hist = run.history(stream="system", pandas=True, samples=300)
    except Exception:
        return out
    if sys_hist is None or sys_hist.empty:
        return out
    step_col = "_step" if "_step" in sys_hist.columns else sys_hist.columns[0]
    steps = sys_hist[step_col].to_numpy() if step_col in sys_hist.columns else np.arange(len(sys_hist))
    for k in SYSTEM_METRICS:
        if k in sys_hist.columns:
            sub = sys_hist[[step_col, k]].dropna() if step_col in sys_hist.columns else sys_hist[[k]].dropna()
            vals = sub[k].to_numpy()
            xs = sub[step_col].to_numpy() if step_col in sub.columns else np.arange(len(vals))
            out[k] = (xs, vals)
    return out


def _fmt_bin_value(v: float) -> str:
    if not math.isfinite(v):
        return "nan"
    return f"{v:.3g}"


def _fmt_metric(m: Optional[MetricSummary]) -> str:
    if m is None:
        return "  (no data)"
    # Headline line
    head = (
        f"  - `{m.name}`: first={m.first:.4g} last={m.last:.4g} "
        f"min={m.vmin:.4g}@{m.step_min} max={m.vmax:.4g}@{m.step_max} "
        f"tail_slope={m.slope_tail:+.3g} snr={m.snr_tail:.2f} "
        f"n={m.n_samples} nonfinite={m.n_nonfinite} "
        f"{'↓' if m.still_decreasing else '→/↑'}"
    )
    # Shape line
    shape_line = f"      shape: {m.shape}"
    # Bins line: "b0 b1 b2 ..." with arrows where direction flips
    bin_vals = [b[2] for b in m.bins]
    bin_strs = [_fmt_bin_value(v) for v in bin_vals]
    # Mark direction changes to make the trajectory visible at a glance
    marks: list[str] = []
    for i, v in enumerate(bin_vals):
        if i == 0 or not math.isfinite(v) or not math.isfinite(bin_vals[i - 1]):
            marks.append("")
        else:
            d = v - bin_vals[i - 1]
            if abs(d) < 1e-12:
                marks.append("")
            elif d > 0:
                marks.append("↑" if d > abs(bin_vals[i - 1]) * 0.5 else "↗")
            else:
                marks.append("↓" if abs(d) > abs(bin_vals[i - 1]) * 0.5 else "↘")
    pairs = [f"{s}{mk}" for s, mk in zip(bin_strs, marks)]
    bins_line = "      bins[10]: " + "  ".join(pairs)
    # Step range the bins span
    if m.bins:
        step_lo = int(m.bins[0][0])
        step_hi = int(m.bins[-1][1])
        bins_line += f"    (steps {step_lo}→{step_hi})"
    lines = [head, shape_line, bins_line]
    # NaN / Inf segments
    if m.nonfinite_segments:
        segs = ", ".join(f"[{int(s)}..{int(e)}]" for s, e in m.nonfinite_segments[:5])
        extra = "" if len(m.nonfinite_segments) <= 5 else f" (+{len(m.nonfinite_segments) - 5} more)"
        lines.append(f"      nan/inf segments: {segs}{extra}")
    # Jumps
    if m.jumps:
        js = ", ".join(f"{v:+.3g}@{int(s)}" for s, v in m.jumps[:5])
        lines.append(f"      jumps (>4σ): {js}")
    return "\n".join(lines)


def _fmt_system(sys_metrics: dict[str, MetricSummary]) -> list[str]:
    out = []
    gpu_util = sys_metrics.get("gpu.process.0.gpu")
    gpu_mem = sys_metrics.get("gpu.process.0.memoryAllocatedBytes")
    if gpu_util is not None:
        util_mean = (gpu_util.first + gpu_util.last) / 2  # crude, but we already summarized
        util_tail = gpu_util.last
        flag = " ⚠ LOW UTIL" if util_tail < 70 else ""
        out.append(f"  - GPU util: mean≈{util_mean:.0f}% tail={util_tail:.0f}%{flag}")
    if gpu_mem is not None:
        mem_gb_last = gpu_mem.last / (1024**3)
        mem_gb_max = gpu_mem.vmax / (1024**3)
        out.append(f"  - GPU mem: last={mem_gb_last:.1f} GiB max={mem_gb_max:.1f} GiB")
    return out


def build_digest(run_id: str, partial: bool = False, baseline_id: Optional[str] = None) -> str:
    run = _resolve_run(run_id)
    cfg = dict(run.config)
    summary = dict(run.summary)

    primary = _fetch_scalars(run, PRIMARY_METRICS, samples=500)
    perf = _fetch_scalars(run, PERF_METRICS, samples=200)
    components = _fetch_loss_components(run, samples=300)
    system = _fetch_system(run)

    def summarize_all(d):
        return {k: _summarize_series(k, s, v) for k, (s, v) in d.items()}

    primary_s = summarize_all(primary)
    perf_s = summarize_all(perf)
    components_s = summarize_all(components)
    system_s = summarize_all(system)

    train_loss = primary_s.get("train/loss_iter")
    convergence = _classify_convergence(train_loss)

    # NaN/inf detection across primary metrics
    nan_hit = any(
        (s is not None and (math.isnan(s.last) or not math.isfinite(s.last)))
        for s in primary_s.values()
    )

    lines: list[str] = []
    lines.append(f"# Run digest — {run.id}{' (partial)' if partial else ''}")
    lines.append("")
    lines.append("## Metadata")
    lines.append(f"- name: {run.name}")
    lines.append(f"- state: {run.state}")
    lines.append(f"- url: {run.url}")
    lines.append(f"- created: {run.created_at}")
    lines.append(f"- runtime: {summary.get('_runtime', 'n/a')} s")
    lines.append(f"- git_sha: {cfg.get('git_sha', 'n/a')}")
    lines.append(f"- partition: {cfg.get('slurm_partition', 'n/a')}")
    lines.append(f"- sanity_ok_sha: {cfg.get('sanity_ok_sha', 'n/a')}")
    lines.append(f"- model_size: {cfg.get('model_size', 'n/a')} | fold: {cfg.get('fold', 'n/a')} | batch_size: {cfg.get('batch_size', 'n/a')}")
    lines.append(f"- text_prompted: {cfg.get('text_prompted', cfg.get('text_prompted.enabled', 'n/a'))}")
    lines.append("")

    lines.append("## Convergence verdict")
    lines.append(f"- verdict: **{convergence}**{' (NaN detected in a primary metric!)' if nan_hit else ''}")
    lines.append("")

    lines.append("## Primary metrics")
    for k in PRIMARY_METRICS:
        lines.append(_fmt_metric(primary_s.get(k)))
    lines.append("")

    if components_s:
        lines.append("## Loss components")
        for k, m in sorted(components_s.items()):
            lines.append(_fmt_metric(m))
        lines.append("")

    lines.append("## Perf / debug")
    for k in PERF_METRICS:
        lines.append(_fmt_metric(perf_s.get(k)))
    lines.append("")

    lines.append("## System")
    sys_lines = _fmt_system(system_s)
    lines.extend(sys_lines if sys_lines else ["  (no system metrics)"])
    lines.append("")

    lines.append("## Artifacts")
    run_dir = cfg.get("run_dir") or cfg.get("output_dir")
    if run_dir:
        lines.append(f"- last_checkpoint: {run_dir}/checkpoint_last.pth")
    lines.append(f"- wandb: {run.url}")
    lines.append("")

    if baseline_id:
        try:
            baseline_digest = build_digest(baseline_id, partial=False, baseline_id=None)
            lines.append("## Baseline comparison")
            lines.append(f"- baseline run: {baseline_id}")
            # Extract baseline's headline val/dice_hard
            lines.append("")
            lines.append("<details><summary>baseline digest</summary>")
            lines.append("")
            lines.append(baseline_digest)
            lines.append("</details>")
        except Exception as e:
            lines.append(f"## Baseline comparison: FAILED to fetch baseline ({e})")

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_id", help="W&B run id, or entity/project/id")
    p.add_argument("--partial", action="store_true", help="mid-run digest")
    p.add_argument("--baseline", default=None, help="baseline run id for side-by-side")
    p.add_argument("--output", default=None, help="write digest to path instead of stdout")
    args = p.parse_args()

    digest = build_digest(args.run_id, partial=args.partial, baseline_id=args.baseline)
    if args.output:
        Path(args.output).write_text(digest)
        print(f"digest written to {args.output}")
    else:
        print(digest)


if __name__ == "__main__":
    main()
