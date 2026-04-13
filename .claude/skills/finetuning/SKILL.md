---
name: finetuning
description: Finetuning modes — freeze encoder, differential LR, LoRA on transformer decoder, two-phase unfreeze. Invoke when the user mentions transfer learning, VoxTell init, or parameter-efficient adaptation.
---

# Finetuning Modes

All modes work with `--init_checkpoint` to load pretrained weights (tries strict, falls back to non-strict for architecture mismatches). Fresh optimizer, epoch 0.

## Freeze encoder

Train only decoder / transformer / projections. Memory-efficient.

```bash
python main.py --text_prompted --init_checkpoint /path/to/voxtell.pth \
    --freeze_encoder --fold 0
```

Sets `requires_grad=False` on encoder params and excludes them from the optimizer.

## Differential learning rate

Encoder trains with a smaller LR than the rest (slower adaptation of pretrained features).

```bash
python main.py --text_prompted --init_checkpoint /path/to/voxtell.pth \
    --encoder_lr_factor 0.1 --fold 0
```

Creates separate optimizer param groups: encoder LR = base LR × `encoder_lr_factor`.

## LoRA on transformer decoder

Minimal trainable parameters. Combine with `--freeze_encoder` for the leanest config.

```bash
# Freeze encoder + LoRA (minimum trainable params)
python main.py --text_prompted --init_checkpoint /path/to/voxtell.pth \
    --freeze_encoder --lora --fold 0

# Custom rank
python main.py --text_prompted --init_checkpoint /path/to/voxtell.pth \
    --lora --lora_rank 32 --fold 0
```

**Custom implementation** in `lora.py` (no `peft` dependency):
- `LoRALinear` wraps frozen `nn.Linear` with trainable low-rank A/B matrices.
- `LoRAMultiheadAttention` decomposes the fused Q/K/V projection into separate LoRA-wrapped projections.
- `apply_lora_to_transformer()` applies LoRA to all attention layers and freezes all non-LoRA transformer parameters.
- For `--init_checkpoint + --lora`: weights load into the original MHA first, then LoRA decomposes and copies them.

## Two-phase finetuning

Phase 1: freeze encoder, train the rest.

```bash
python main.py --text_prompted --init_checkpoint /path/to/voxtell.pth \
    --freeze_encoder --fold 0
```

Phase 2: unfreeze with a small encoder LR, warm-restart from Phase 1.

```bash
python main.py --resume <phase1_run_id> --use_new_config --warm_restart \
    --encoder_lr_factor 0.1 --epochs 200
```

## VoxTell converted weights

Starting point for most text-prompted finetuning: `experiments/voxtell_converted/checkpoint.pth`. Conversion script: `convert_voxtell_checkpoint.py`.
