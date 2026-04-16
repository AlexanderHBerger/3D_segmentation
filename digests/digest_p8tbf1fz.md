# Run digest — p8tbf1fz

## Metadata
- name: fold_0
- state: finished
- url: https://wandb.ai/aimt/3D-Segmentation-sanity/runs/p8tbf1fz
- created: 2026-04-14T04:28:57Z
- runtime: 9438 s
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
  - `train/loss_ce`: first=0.01554 last=5.586e-05 min=7.371e-06@47520 max=0.07324@335 tail_slope=-2.05e-09 snr=1.14 n=300 nonfinite=0 ↓
      shape: noisy-decreasing
      bins[10]: 0.00597  0.000355↓  0.000232↘  0.000163↘  0.00011↘  0.000118↗  0.000111↘  0.000101↘  6.23e-05↘  6.5e-05↗    (steps 158→49941)
      jumps (>4σ): +0.0577@335, -0.0438@489, -0.026@696
  - `train/loss_dice`: first=0.9925 last=0.0005733 min=0.0002362@44059 max=1@768 tail_slope=-8.31e-08 snr=0.70 n=300 nonfinite=0 ↓
      shape: noisy-decreasing
      bins[10]: 0.189  0.0105↓  0.00442↓  0.00428↘  0.00336↘  0.00278↘  0.00372↗  0.00371↘  0.00162↓  0.00188↗    (steps 158→49941)
      jumps (>4σ): -0.866@989, +0.809@768, -0.808@696, +0.629@1671, -0.623@1924

## Perf / debug
  - `train/epoch_time`: first=47.51 last=46.39 min=45.55@11999 max=66.13@12249 tail_slope=-3.31e-05 snr=148.37 n=200 nonfinite=0 ↓
      shape: monotonic-increasing
      bins[10]: 46.5  46.2↘  47.3↗  46.5↘  47↗  46.6↘  46.6↘  46.9↗  46.7↘  46.6↘    (steps 249→49999)
      jumps (>4σ): +20.6@12249, -19.7@12499, +9.81@24999, -9.62@25249
  - `train/iter_time`: first=0.1753 last=0.1752 min=0.1742@40749 max=0.6062@43249 tail_slope=-2.58e-06 snr=2.78 n=200 nonfinite=0 ↓
      shape: noisy-increasing
      bins[10]: 0.176  0.175↘  0.176↗  0.176↗  0.176↘  0.176↗  0.177↗  0.177↗  0.198↗  0.176↘    (steps 249→49999)
      jumps (>4σ): +0.431@43249, -0.431@43499
  - `train/learning_rate`: first=0.001 last=8.493e-06 min=8.493e-06@49999 max=0.001@249 tail_slope=-2.29e-08 snr=1.91 n=200 nonfinite=0 ↓
      shape: noisy-decreasing
      bins[10]: 0.000957  0.000866↘  0.000774↘  0.000681↘  0.000586↘  0.00049↘  0.000391↘  0.00029↘  0.000184↘  6.94e-05↓    (steps 249→49999)
      jumps (>4σ): -7.36e-06@49999, -6.98e-06@49749, -6.75e-06@49499, -6.58e-06@49249, -6.45e-06@48999

## System
  (no system metrics)

## Artifacts
- wandb: https://wandb.ai/aimt/3D-Segmentation-sanity/runs/p8tbf1fz
