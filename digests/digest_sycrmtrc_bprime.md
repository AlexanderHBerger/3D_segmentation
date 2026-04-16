# Run digest — sycrmtrc

## Metadata
- name: fold_0
- state: finished
- url: https://wandb.ai/aimt/3D-Segmentation-sanity/runs/sycrmtrc
- created: 2026-04-15T22:15:38Z
- runtime: 32908 s
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
  - `train/loss_ce`: first=0.02168 last=0.0001949 min=4.295e-14@19240 max=0.05457@1320 tail_slope=+1.97e-09 snr=0.79 n=300 nonfinite=0 →/↑
      shape: fall-then-recovery (trough 4.3e-14 @ bin 5/10; end 0.000195)
      bins[10]: 0.00627  0.00131↓  0.000238↓  0.000212↘  0.000132↘  0.000104↘  5.16e-05↓  6.01e-05↗  6.26e-05↗  6.13e-05↘    (steps 158→49941)
      jumps (>4σ): -0.0498@1554, +0.0487@1320
  - `train/loss_dice`: first=1 last=1 min=0@10757 max=1@158 tail_slope=+2.67e-05 snr=0.37 n=300 nonfinite=0 →/↑
      shape: fall-then-recovery (trough 0 @ bin 3/10; end 1)
      bins[10]: 0.459  0.359↘  0.124↓  0.0612↓  0.123↑  0.194↑  0.00129↓  0.13↑  0.0382↓  0.189↑    (steps 158→49941)

## Perf / debug
  - `train/epoch_time`: first=158.1 last=155.1 min=154@21749 max=159.8@48749 tail_slope=+6.94e-05 snr=171.96 n=200 nonfinite=0 →/↑
      shape: monotonic-decreasing
      bins[10]: 156  155↘  155↘  155↗  155↘  155↘  155↘  155↗  155↘  155↗    (steps 249→49999)
  - `train/iter_time`: first=0.5911 last=0.5935 min=0.589@9999 max=0.5955@19249 tail_slope=+1.02e-08 snr=450.68 n=200 nonfinite=0 →/↑
      shape: monotonic-increasing
      bins[10]: 0.591  0.591↗  0.591↘  0.591↗  0.591↘  0.591↘  0.592↗  0.592↗  0.591↘  0.592↗    (steps 249→49999)
  - `train/learning_rate`: first=0.001 last=8.493e-06 min=8.493e-06@49999 max=0.001@249 tail_slope=-2.29e-08 snr=1.91 n=200 nonfinite=0 ↓
      shape: noisy-decreasing
      bins[10]: 0.000957  0.000866↘  0.000774↘  0.000681↘  0.000586↘  0.00049↘  0.000391↘  0.00029↘  0.000184↘  6.94e-05↓    (steps 249→49999)
      jumps (>4σ): -7.36e-06@49999, -6.98e-06@49749, -6.75e-06@49499, -6.58e-06@49249, -6.45e-06@48999

## System
  (no system metrics)

## Artifacts
- wandb: https://wandb.ai/aimt/3D-Segmentation-sanity/runs/sycrmtrc
