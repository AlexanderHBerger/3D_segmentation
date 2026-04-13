# Run digest — 0ltd2fet

## Metadata
- name: fold_0
- state: crashed
- url: https://wandb.ai/aimt/3D-Segmentation/runs/0ltd2fet
- created: 2026-04-03T19:32:14Z
- runtime: 242995.073202008 s
- git_sha: n/a
- partition: n/a
- sanity_ok_sha: n/a
- model_size: n/a | fold: 0 | batch_size: n/a
- text_prompted: n/a

## Convergence verdict
- verdict: **NaN-hit**

## Primary metrics
  - `val/dice_hard`: first=0.1346 last=0.007622 min=0.007622@9999 max=0.3317@14999 tail_slope=+0 snr=inf n=42 nonfinite=0 →/↑
      shape: monotonic-decreasing
      bins[10]: 0.21  0.242↗  0.0592↓  0.00762↓  0.00762  0.00762  0.00762  0.00762  0.00762  0.00762    (steps 0→102499)
      jumps (>4σ): -0.321@9999
  - `val/loss`: first=0.8733 last=0.9662 min=0.682@7499 max=1.033@9999 tail_slope=-9.11e-08 snr=8488.10 n=42 nonfinite=29 ↓
      shape: nan-hit (last 29/42 samples nonfinite; prior shape: monotonic-increasing)
      bins[10]: 0.805  0.777↘  0.967↗  nan  nan  nan  nan  nan  nan  nan    (steps 0→102499)
      nan/inf segments: [32499..102499]
  - `train/loss_iter`: first=0.6849 last=1.008 min=0.3941@12499 max=1.034@9999 tail_slope=+5.39e-06 snr=148.72 n=42 nonfinite=29 →/↑
      shape: nan-hit (last 29/42 samples nonfinite; prior shape: fall-then-recovery (trough 0.394 @ bin 4/10; end 1.01))
      bins[10]: 0.761  0.664↘  1.01↑  nan  nan  nan  nan  nan  nan  nan    (steps 0→102499)
      nan/inf segments: [32499..102499]

## Loss components
  - `train/loss_ce`: first=0.0008158 last=0.05647 min=0.0001511@1671 max=0.05647@30108 tail_slope=+3.24e-06 snr=1.60 n=300 nonfinite=217 →/↑
      shape: nan-hit (last 216/300 samples nonfinite; prior shape: noisy-increasing)
      bins[10]: 0.0131  0.00919↘  0.0236↑  nan  nan  nan  nan  nan  nan  nan    (steps 489→102747)
      nan/inf segments: [20422..20422], [30916..102747]
  - `train/loss_dice`: first=0.8646 last=0.9097 min=0.2388@15127 max=1@7946 tail_slope=-1.88e-06 snr=26.13 n=300 nonfinite=217 ↓
      shape: nan-hit (last 216/300 samples nonfinite; prior shape: fall-then-recovery (trough 0.239 @ bin 5/10; end 0.91))
      bins[10]: 0.82  0.742↘  0.96↗  nan  nan  nan  nan  nan  nan  nan    (steps 489→102747)
      nan/inf segments: [20422..20422], [30916..102747]

## Perf / debug
  - `train/epoch_time`: first=556.6 last=537.9 min=473.7@60249 max=946.3@49999 tail_slope=-0.00141 snr=4.95 n=200 nonfinite=0 ↓
      shape: noisy-increasing
      bins[10]: 561  557↘  577↗  536↘  595↗  553↘  536↘  561↗  589↗  578↘    (steps 1249→102249)
  - `train/iter_time`: first=1.147 last=0.9933 min=0.9876@31999 max=14.83@78749 tail_slope=+2.08e-08 snr=189.31 n=200 nonfinite=0 →/↑
      shape: rise-then-collapse (peak 14.8 @ bin 8/10; end 0.993 = 7% of peak)
      bins[10]: 1.29  1.89↗  2.11↗  1.19↘  0.996↘  0.995↘  0.995↘  1.81↑  0.996↘  0.996↗    (steps 1249→102249)
      jumps (>4σ): -13.8@78999, +13.8@78749, +13.4@16249, -13.4@16749
  - `train/learning_rate`: first=9.964e-05 last=6.239e-05 min=6.239e-05@102249 max=9.964e-05@1249 tail_slope=-3.77e-10 snr=32.28 n=200 nonfinite=0 ↓
      shape: monotonic-decreasing
      bins[10]: 9.78e-05  9.43e-05↘  9.05e-05↘  8.65e-05↘  8.31e-05↘  7.92e-05↘  7.57e-05↘  7.16e-05↘  6.81e-05↘  6.4e-05↘    (steps 1249→102249)
      jumps (>4σ): -7.31e-07@35499, -7.26e-07@21249, -7.22e-07@8249, -5.59e-07@74999, -5.59e-07@73499

## System
  (no system metrics)

## Artifacts
- wandb: https://wandb.ai/aimt/3D-Segmentation/runs/0ltd2fet


