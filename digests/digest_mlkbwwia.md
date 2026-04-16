# Run digest — mlkbwwia

## Metadata
- name: fold_0
- state: finished
- url: https://wandb.ai/aimt/3D-Segmentation-sanity/runs/mlkbwwia
- created: 2026-04-14T18:36:24Z
- runtime: 24403 s
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
  - `train/loss_ce`: first=0.0008084 last=1.803e-05 min=8.583e-16@42904 max=0.003017@1150 tail_slope=+3.81e-10 snr=1.21 n=300 nonfinite=0 →/↑
      shape: noisy-decreasing
      bins[10]: 0.000984  0.000224↓  0.000131↘  0.000122↘  0.000146↗  0.000109↘  6.76e-05↘  5.2e-05↘  4.8e-05↘  4.73e-05↘    (steps 158→49941)
      jumps (>4σ): +0.00253@1150, -0.00206@768, +0.00165@2088, +0.00162@4156, -0.0016@4225
  - `train/loss_dice`: first=0.369 last=0.0241 min=0@10008 max=1@335 tail_slope=-4.34e-06 snr=0.44 n=300 nonfinite=0 ↓
      shape: fall-then-recovery (trough 0 @ bin 3/10; end 0.0241)
      bins[10]: 0.227  0.23↗  0.138↘  0.105↘  0.232↑  0.0399↓  0.0277↘  0.0188↘  0.061↑  0.035↘    (steps 158→49941)

## Perf / debug
  - `train/epoch_time`: first=125.4 last=114.1 min=112.4@49249 max=125.4@249 tail_slope=-1.95e-05 snr=75.47 n=200 nonfinite=0 ↓
      shape: monotonic-decreasing
      bins[10]: 120  119↘  118↘  118↘  117↘  117↗  116↘  117↗  116↘  116↘    (steps 249→49999)
  - `train/iter_time`: first=0.3663 last=0.3648 min=0.3638@32249 max=2.313@29499 tail_slope=-8.6e-06 snr=1.16 n=200 nonfinite=0 ↓
      shape: rise-then-collapse (peak 2.31 @ bin 6/10; end 0.365 = 16% of peak)
      bins[10]: 0.375  0.462↗  0.369↘  0.521↗  0.462↘  0.466↗  0.443↘  0.46↗  0.432↘  0.546↗    (steps 249→49999)
  - `train/learning_rate`: first=0.001 last=8.493e-06 min=8.493e-06@49999 max=0.001@249 tail_slope=-2.29e-08 snr=1.91 n=200 nonfinite=0 ↓
      shape: noisy-decreasing
      bins[10]: 0.000957  0.000866↘  0.000774↘  0.000681↘  0.000586↘  0.00049↘  0.000391↘  0.00029↘  0.000184↘  6.94e-05↓    (steps 249→49999)
      jumps (>4σ): -7.36e-06@49999, -6.98e-06@49749, -6.75e-06@49499, -6.58e-06@49249, -6.45e-06@48999

## System
  (no system metrics)

## Artifacts
- wandb: https://wandb.ai/aimt/3D-Segmentation-sanity/runs/mlkbwwia
