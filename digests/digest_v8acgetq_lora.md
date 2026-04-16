# Run digest — v8acgetq

## Metadata
- name: fold_0
- state: finished
- url: https://wandb.ai/aimt/3D-Segmentation-sanity/runs/v8acgetq
- created: 2026-04-15T22:41:29Z
- runtime: 18184 s
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
  - `train/loss_ce`: first=0.003264 last=1.56e-05 min=2.705e-13@33394 max=0.01409@1188 tail_slope=+1.19e-08 snr=0.72 n=300 nonfinite=0 →/↑
      shape: fall-then-recovery (trough 2.71e-13 @ bin 7/10; end 1.56e-05)
      bins[10]: 0.00196  0.000654↓  0.000641↘  0.000413↘  0.000254↘  0.000359↗  0.00021↘  0.000239↗  0.000184↘  0.000249↗    (steps 158→49941)
      jumps (>4σ): +0.0138@1188, -0.0137@1320, -0.00796@768, +0.00739@696
  - `train/loss_dice`: first=1 last=0.001059 min=0@33394 max=1@158 tail_slope=+3.36e-06 snr=0.38 n=300 nonfinite=0 →/↑
      shape: fall-then-recovery (trough 0 @ bin 7/10; end 0.00106)
      bins[10]: 0.488  0.282↘  0.461↑  0.275↘  0.164↘  0.134↘  0.0211↓  0.0176↘  0.0954↑  0.0879↘    (steps 158→49941)

## Perf / debug
  - `train/epoch_time`: first=91.45 last=84.44 min=83.32@39999 max=91.45@249 tail_slope=-2.57e-05 snr=144.49 n=200 nonfinite=0 ↓
      shape: monotonic-decreasing
      bins[10]: 84.9  84.6↘  84.4↘  84.3↘  84.7↗  84.5↘  84.3↘  84.4↗  84.5↗  84.4↘    (steps 249→49999)
      jumps (>4σ): -7.2@499
  - `train/iter_time`: first=0.3058 last=0.3053 min=0.3038@49749 max=1.899@20999 tail_slope=-1.04e-05 snr=1.94 n=200 nonfinite=0 ↓
      shape: rise-then-collapse (peak 1.9 @ bin 5/10; end 0.305 = 16% of peak)
      bins[10]: 0.306  0.422↗  0.376↘  0.306↘  0.386↗  0.306↘  0.369↗  0.363↘  0.362↘  0.306↘    (steps 249→49999)
      jumps (>4σ): -1.59@21249, +1.59@20999, +1.4@7749, -1.4@7999, +1.39@12749
  - `train/learning_rate`: first=0.001 last=8.493e-06 min=8.493e-06@49999 max=0.001@249 tail_slope=-2.29e-08 snr=1.91 n=200 nonfinite=0 ↓
      shape: noisy-decreasing
      bins[10]: 0.000957  0.000866↘  0.000774↘  0.000681↘  0.000586↘  0.00049↘  0.000391↘  0.00029↘  0.000184↘  6.94e-05↓    (steps 249→49999)
      jumps (>4σ): -7.36e-06@49999, -6.98e-06@49749, -6.75e-06@49499, -6.58e-06@49249, -6.45e-06@48999

## System
  (no system metrics)

## Artifacts
- wandb: https://wandb.ai/aimt/3D-Segmentation-sanity/runs/v8acgetq
