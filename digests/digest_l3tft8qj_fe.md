# Run digest — l3tft8qj

## Metadata
- name: fold_0
- state: finished
- url: https://wandb.ai/aimt/3D-Segmentation-sanity/runs/l3tft8qj
- created: 2026-04-15T22:41:28Z
- runtime: 19462 s
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
  - `train/loss_ce`: first=0.02103 last=1.406e-05 min=1.381e-11@22713 max=0.02103@158 tail_slope=-1.43e-08 snr=0.81 n=300 nonfinite=0 ↓
      shape: fall-then-recovery (trough 1.38e-11 @ bin 5/10; end 1.41e-05)
      bins[10]: 0.00224  0.000311↓  0.000698↑  0.000351↘  0.000225↘  0.000286↗  0.000405↗  0.000249↘  0.000297↗  0.000174↘    (steps 158→49941)
      jumps (>4σ): -0.00894@335, -0.0049@489, -0.00464@1082
  - `train/loss_dice`: first=1 last=0.001278 min=1.788e-07@30795 max=1@158 tail_slope=-3.07e-05 snr=0.44 n=300 nonfinite=0 ↓
      shape: fall-then-recovery (trough 1.79e-07 @ bin 7/10; end 0.00128)
      bins[10]: 0.484  0.281↘  0.255↘  0.103↓  0.089↘  0.135↑  0.133↘  0.135↗  0.262↑  0.0689↓    (steps 158→49941)

## Perf / debug
  - `train/epoch_time`: first=94.44 last=88.53 min=87.38@12749 max=94.44@249 tail_slope=-2.51e-05 snr=126.55 n=200 nonfinite=0 ↓
      shape: monotonic-decreasing
      bins[10]: 89.3  89.2↘  88.9↘  89.1↗  89.1↘  89↘  89↘  89.1↗  88.8↘  88.8↘    (steps 249→49999)
      jumps (>4σ): -3.92@499
  - `train/iter_time`: first=0.3229 last=0.3237 min=0.3225@42249 max=1.269@29499 tail_slope=-1.07e-07 snr=309.46 n=200 nonfinite=0 ↓
      shape: rise-then-collapse (peak 1.27 @ bin 6/10; end 0.324 = 26% of peak)
      bins[10]: 0.371  0.325↘  0.325↘  0.325↘  0.325↗  0.372↗  0.325↘  0.324↘  0.325↗  0.324↘    (steps 249→49999)
      jumps (>4σ): -0.945@29749, +0.944@29499, +0.931@3499, -0.93@3749
  - `train/learning_rate`: first=0.001 last=8.493e-06 min=8.493e-06@49999 max=0.001@249 tail_slope=-2.29e-08 snr=1.91 n=200 nonfinite=0 ↓
      shape: noisy-decreasing
      bins[10]: 0.000957  0.000866↘  0.000774↘  0.000681↘  0.000586↘  0.00049↘  0.000391↘  0.00029↘  0.000184↘  6.94e-05↓    (steps 249→49999)
      jumps (>4σ): -7.36e-06@49999, -6.98e-06@49749, -6.75e-06@49499, -6.58e-06@49249, -6.45e-06@48999

## System
  (no system metrics)

## Artifacts
- wandb: https://wandb.ai/aimt/3D-Segmentation-sanity/runs/l3tft8qj
