# Run digest — wnvtqob0 (partial)

## Metadata
- name: fold_0
- state: running
- url: https://wandb.ai/aimt/3D-Segmentation-sanity/runs/wnvtqob0
- created: 2026-04-14T18:30:12Z
- runtime: 14396.919695281 s
- git_sha: n/a
- partition: n/a
- sanity_ok_sha: n/a
- model_size: n/a | fold: 0 | batch_size: n/a
- text_prompted: n/a

## Convergence verdict
- verdict: **no-data**

## Primary metrics
  (no data)
  (no data)
  (no data)

## Loss components
  - `train/loss_ce`: first=0.01459 last=5.883e-06 min=1.499e-13@7425 max=0.02189@54 tail_slope=-4.08e-10 snr=1.21 n=300 nonfinite=0 ↓
      shape: fall-then-recovery (trough 1.5e-13 @ bin 8/10; end 5.88e-06)
      bins[10]: 0.00284  4.65e-05↓  3.04e-05↘  2.07e-05↘  1.34e-05↘  8.98e-06↘  7.98e-06↘  6.6e-06↘  5.77e-06↘  5.23e-06↘    (steps 24→10047)
      jumps (>4σ): -0.0113@158, +0.01@418, -0.00895@432, +0.0073@54, +0.00624@313
  - `train/loss_dice`: first=0.9993 last=0.000233 min=0@1134 max=0.9993@24 tail_slope=-6.92e-09 snr=1.71 n=300 nonfinite=0 ↓
      shape: noisy-decreasing
      bins[10]: 0.223  0.0015↓  0.000669↓  0.000482↘  0.000326↘  0.000226↘  0.000251↗  0.000181↘  0.000192↗  0.000185↘    (steps 24→10047)
      jumps (>4σ): +0.925@313, -0.467@335, -0.467@304, -0.451@376, -0.407@158

## Perf / debug
  - `train/epoch_time`: first=350.4 last=352.5 min=337.2@4999 max=367.1@1999 tail_slope=-0.000815 snr=54.33 n=40 nonfinite=0 ↓
      shape: monotonic-decreasing
      bins[10]: 352  357↗  354↘  355↗  350↘  355↗  349↘  357↗  354↘  352↘    (steps 249→9999)
  - `train/iter_time`: first=1.409 last=1.613 min=1.042@7749 max=2.09@499 tail_slope=+0.000293 snr=6.07 n=40 nonfinite=0 →/↑
      shape: monotonic-decreasing
      bins[10]: 1.64  1.77↗  1.57↘  1.5↘  1.22↘  1.47↗  1.33↘  1.33↘  1.22↘  1.52↗    (steps 249→9999)
  - `train/learning_rate`: first=0.001 last=0.0006409 min=0.0006409@9999 max=0.001@249 tail_slope=-3.76e-08 snr=31.27 n=40 nonfinite=0 ↓
      shape: monotonic-decreasing
      bins[10]: 0.000986  0.00095↘  0.000914↘  0.000878↘  0.000841↘  0.000804↘  0.000767↘  0.00073↘  0.000693↘  0.000655↘    (steps 249→9999)
      jumps (>4σ): -9.45e-06@9999, -9.43e-06@9749, -9.42e-06@9499, -9.4e-06@9249, -9.39e-06@8999

## System
  (no system metrics)

## Artifacts
- wandb: https://wandb.ai/aimt/3D-Segmentation-sanity/runs/wnvtqob0
