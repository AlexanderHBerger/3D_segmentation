# Run digest — d842zptt

## Metadata
- name: fold_0
- state: finished
- url: https://wandb.ai/aimt/3D-Segmentation-sanity/runs/d842zptt
- created: 2026-04-15T22:41:27Z
- runtime: 32848 s
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
  - `train/loss_ce`: first=0.01922 last=0.0004814 min=1.881e-11@30818 max=0.02643@335 tail_slope=-6.89e-09 snr=0.88 n=300 nonfinite=0 ↓
      shape: fall-then-recovery (trough 1.88e-11 @ bin 7/10; end 0.000481)
      bins[10]: 0.00383  0.00102↓  0.00145↗  0.000854↘  0.000296↓  0.000319↗  0.000221↘  0.0002↘  0.000253↗  0.000223↘    (steps 158→49941)
      jumps (>4σ): -0.0167@489, -0.00945@8338, +0.00938@8325
  - `train/loss_dice`: first=1 last=0.00627 min=5.364e-07@30818 max=1@158 tail_slope=-7.92e-06 snr=0.44 n=300 nonfinite=0 ↓
      shape: fall-then-recovery (trough 5.36e-07 @ bin 7/10; end 0.00627)
      bins[10]: 0.637  0.459↘  0.231↘  0.106↓  0.206↑  0.233↗  0.172↘  0.133↘  0.227↑  0.0986↓    (steps 158→49941)

## Perf / debug
  - `train/epoch_time`: first=159.8 last=154.4 min=153.5@42749 max=162.4@27749 tail_slope=+4.19e-05 snr=143.36 n=200 nonfinite=0 →/↑
      shape: monotonic-decreasing
      bins[10]: 156  156↘  156↗  156↘  156↗  156↘  155↘  155↗  155↘  155↗    (steps 249→49999)
      jumps (>4σ): +6.88@27749, -6.5@27999
  - `train/iter_time`: first=0.5927 last=0.5919 min=0.5882@49499 max=1.685@44249 tail_slope=+2.39e-06 snr=2.73 n=200 nonfinite=0 →/↑
      shape: noisy-increasing
      bins[10]: 0.637  0.646↗  0.593↘  0.593↘  0.593↘  0.593↘  0.592↘  0.593↗  0.647↗  0.646↘    (steps 249→49999)
      jumps (>4σ): +1.09@44249, -1.09@44499, -1.08@46999, +1.08@46749, -1.04@9499
  - `train/learning_rate`: first=0.001 last=8.493e-06 min=8.493e-06@49999 max=0.001@249 tail_slope=-2.29e-08 snr=1.91 n=200 nonfinite=0 ↓
      shape: noisy-decreasing
      bins[10]: 0.000957  0.000866↘  0.000774↘  0.000681↘  0.000586↘  0.00049↘  0.000391↘  0.00029↘  0.000184↘  6.94e-05↓    (steps 249→49999)
      jumps (>4σ): -7.36e-06@49999, -6.98e-06@49749, -6.75e-06@49499, -6.58e-06@49249, -6.45e-06@48999

## System
  (no system metrics)

## Artifacts
- wandb: https://wandb.ai/aimt/3D-Segmentation-sanity/runs/d842zptt
