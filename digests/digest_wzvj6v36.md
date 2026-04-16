# Run digest — wzvj6v36

## Metadata
- name: fold_0
- state: finished
- url: https://wandb.ai/aimt/3D-Segmentation-sanity/runs/wzvj6v36
- created: 2026-04-14T03:57:55Z
- runtime: 5106 s
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
  - `train/loss_ce`: first=0.01258 last=2.198e-07 min=2.197e-07@24750 max=0.01649@335 tail_slope=-1.67e-12 snr=85.83 n=300 nonfinite=0 ↓
      shape: monotonic-decreasing
      bins[10]: 0.00377  1.55e-05↓  6.14e-06↓  2.5e-06↓  1.18e-06↓  5.25e-07↓  2.49e-07↓  2.33e-07↘  2.25e-07↘  2.21e-07↘    (steps 158→24797)
      jumps (>4σ): -0.00887@696, -0.00314@418, -0.00242@473
  - `train/loss_dice`: first=0.9931 last=5.776e-05 min=5.776e-05@24243 max=0.9931@158 tail_slope=-4.46e-10 snr=84.56 n=300 nonfinite=0 ↓
      shape: monotonic-decreasing
      bins[10]: 0.259  0.000715↓  0.000313↓  0.00018↘  0.000116↘  8.74e-05↘  6.54e-05↘  6.12e-05↘  5.9e-05↘  5.8e-05↘    (steps 158→24797)
      jumps (>4σ): -0.653@696

## Perf / debug
  - `train/epoch_time`: first=51.7 last=50.08 min=49.91@3249 max=51.84@20499 tail_slope=-0.000133 snr=131.32 n=100 nonfinite=0 ↓
      shape: monotonic-increasing
      bins[10]: 50.5  50.3↘  50.5↗  50.5↗  50.5↘  50.5↗  50.7↗  50.7↘  50.9↗  50.7↘    (steps 249→24999)
  - `train/iter_time`: first=0.1749 last=0.1743 min=0.1741@5749 max=0.3922@22999 tail_slope=+1.33e-06 snr=3.18 n=100 nonfinite=0 →/↑
      shape: noisy-increasing
      bins[10]: 0.195  0.175↘  0.216↗  0.175↘  0.195↗  0.195↗  0.216↗  0.195↘  0.195↘  0.198↗    (steps 249→24999)
  - `train/learning_rate`: first=0.001 last=1.585e-05 min=1.585e-05@24999 max=0.001@249 tail_slope=-4.57e-08 snr=1.96 n=100 nonfinite=0 ↓
      shape: noisy-decreasing
      bins[10]: 0.000959  0.000868↘  0.000776↘  0.000683↘  0.000589↘  0.000492↘  0.000394↘  0.000292↘  0.000186↘  7.25e-05↓    (steps 249→24999)
      jumps (>4σ): -1.37e-05@24999, -1.3e-05@24749, -1.26e-05@24499, -1.23e-05@24249, -1.2e-05@23999

## System
  (no system metrics)

## Artifacts
- wandb: https://wandb.ai/aimt/3D-Segmentation-sanity/runs/wzvj6v36
