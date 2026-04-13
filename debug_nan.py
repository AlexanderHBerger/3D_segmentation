"""Debug script: track activation magnitudes over training to find NaN root cause.
Usage: python debug_nan.py [checkpoint_path]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import sys, os

sys.path.insert(0, os.path.dirname(__file__))


def main():
    from model import create_model

    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else \
        '/midtier/paetzollab/scratch/ahb4007/3D_segmentation/experiments/fold_0_8lu19b3y/best_model.pth'

    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = ckpt['config']
    print(f"Checkpoint epoch: {ckpt.get('epoch', '?')}")

    model = create_model(config)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model = model.cuda()
    model.train()

    patch_size = config.data.patch_size
    B = 2
    N = 1

    # Load real embedding
    data_dir = '/midtier/paetzollab/scratch/ahb4007/data/nnUNet_preprocessed/Dataset018_TextPrompted'
    embeddings_dict = torch.load(os.path.join(data_dir, 'embeddings.pt'), map_location='cpu', weights_only=False)
    real_embed = embeddings_dict[list(embeddings_dict.keys())[0]]
    text_embed = real_embed.unsqueeze(0).expand(B, -1).unsqueeze(1).cuda()

    # Track specific activations
    tracked = {}
    def make_hook(name):
        def hook(module, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(t):
                tracked[name] = (t.min().item(), t.max().item(), t.abs().mean().item())
        return hook

    # Hook the critical layers
    handles = []
    for name, module in model.named_modules():
        if name in [
            'project_text_embed', 'project_text_embed.2',
            'project_bottleneck_embed', 'project_bottleneck_embed.2',
            'transformer_decoder', 'transformer_decoder.norm',
            'project_to_decoder_channels.0', 'project_to_decoder_channels.0.2',
            'project_to_decoder_channels.1', 'project_to_decoder_channels.2',
            'decoder.stages.4', 'decoder.stages.3', 'decoder.stages.2',
        ]:
            handles.append(module.register_forward_hook(make_hook(name)))

    # Also hook the einsum by patching the decoder forward
    original_decoder_forward = model.decoder.forward.__func__
    einsum_stats = {}

    def patched_forward(self, skips, mask_embeddings):
        lres_input = skips[-1]
        seg_outputs = []
        mask_embeddings_rev = mask_embeddings[::-1]

        for stage_idx in range(len(self.stages)):
            x = self.transpconvs[stage_idx](lres_input)
            x = torch.cat((x, skips[-(stage_idx + 2)]), dim=1)
            x = self.stages[stage_idx](x)

            if stage_idx == (len(self.stages) - 1):
                me = mask_embeddings_rev[-1]
                einsum_stats['final_x'] = (x.min().item(), x.max().item(), x.abs().mean().item())
                einsum_stats['final_mask_emb'] = (me.min().item(), me.max().item(), me.abs().mean().item())
                seg_pred = torch.einsum('b c h w d, b n c -> b n h w d', x, me)
                einsum_stats['final_output'] = (seg_pred.min().item(), seg_pred.max().item(), seg_pred.abs().mean().item())
                seg_outputs.append(seg_pred)
            elif stage_idx >= len(self.stages) - len(mask_embeddings_rev):
                mask_embedding = mask_embeddings_rev.pop(0)
                batch_size, _, channels = mask_embedding.shape
                mask_embedding_reshaped = mask_embedding.view(batch_size, self.num_heads, -1)
                fusion_features = torch.einsum('b c h w d, b n c -> b n h w d', x, mask_embedding_reshaped)

                if stage_idx == 1:  # Track intermediate stage
                    einsum_stats[f'stage{stage_idx}_x'] = (x.min().item(), x.max().item(), x.abs().mean().item())
                    einsum_stats[f'stage{stage_idx}_mask'] = (mask_embedding_reshaped.min().item(), mask_embedding_reshaped.max().item(), mask_embedding_reshaped.abs().mean().item())
                    einsum_stats[f'stage{stage_idx}_fusion'] = (fusion_features.min().item(), fusion_features.max().item(), fusion_features.abs().mean().item())

                x = torch.cat((x, fusion_features), dim=1)
                seg_outputs.append(self.seg_layers[stage_idx](x))
            lres_input = x

        seg_outputs = seg_outputs[::-1]
        if not self.deep_supervision:
            return seg_outputs[:1]
        return seg_outputs

    import types
    model.decoder.forward = types.MethodType(patched_forward, model.decoder)

    # Training simulation with SGD (matching actual training)
    print(f"\n{'='*80}")
    print("Training simulation: SGD lr=0.001, momentum=0.99, fp16")
    print(f"{'='*80}")

    scaler = GradScaler('cuda')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.99,
                                 weight_decay=5e-5, nesterov=True)

    header = f"{'Step':>5} | {'Loss':>8} | {'GradNorm':>9} | {'Scale':>8} | {'Out min':>9} {'Out max':>9} | {'x_abs':>7} {'me_abs':>7} | {'text_proj':>9} {'bot_proj':>9} {'trans_out':>9}"
    print(header)
    print("-" * len(header))

    for step in range(300):
        optimizer.zero_grad()
        tracked.clear()
        einsum_stats.clear()

        image = torch.randn(B, 1, *patch_size, device='cuda')
        target = (torch.rand(B, N, *patch_size, device='cuda') > 0.95).float()

        with autocast(device_type='cuda', enabled=True):
            output = model(image, text_embed)
            loss_bce = F.binary_cross_entropy_with_logits(output, target)
            probs = torch.sigmoid(output)
            p = probs.view(B, N, -1)
            t = target.view(B, N, -1)
            tp = (p * t).sum(-1)
            fp = (p * (1 - t)).sum(-1)
            fn = ((1 - p) * t).sum(-1)
            dice = 1.0 - (2*tp + 1e-5) / (2*tp + fp + fn + 1e-5)
            loss = loss_bce + dice.mean()

        if torch.isnan(loss):
            print(f"\n*** NaN LOSS at step {step} ***")
            print(f"  Output: [{output.min().item():.4g}, {output.max().item():.4g}]" if not torch.isnan(output).any() else "  Output: NaN")
            for k, v in sorted(einsum_stats.items()):
                print(f"  {k}: [{v[0]:.4g}, {v[1]:.4g}], abs_mean={v[2]:.4g}")
            for k, v in sorted(tracked.items()):
                print(f"  {k}: [{v[0]:.4g}, {v[1]:.4g}], abs_mean={v[2]:.4g}")
            break

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
        old_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        new_scale = scaler.get_scale()

        if step % 5 == 0 or new_scale < old_scale or step < 10:
            fx = einsum_stats.get('final_x', (0,0,0))
            fme = einsum_stats.get('final_mask_emb', (0,0,0))
            fo = einsum_stats.get('final_output', (0,0,0))
            tp_ = tracked.get('project_text_embed.2', (0,0,0))
            bp_ = tracked.get('project_bottleneck_embed.2', (0,0,0))
            tn_ = tracked.get('transformer_decoder.norm', (0,0,0))

            scale_str = f"{old_scale:.0f}"
            if new_scale < old_scale:
                scale_str += f"→{new_scale:.0f}"

            gn = f"{grad_norm:.4g}" if not (isinstance(grad_norm, float) and grad_norm != grad_norm) else "NaN"

            print(f"{step:5d} | {loss.item():8.4f} | {gn:>9} | {scale_str:>8} | "
                  f"{fo[0]:9.2f} {fo[1]:9.2f} | {fx[2]:7.3f} {fme[2]:7.3f} | "
                  f"{tp_[2]:9.4f} {bp_[2]:9.4f} {tn_[2]:9.4f}")
    else:
        print(f"\nNo NaN after 300 steps.")

    # Repeat without mixed precision
    print(f"\n{'='*80}")
    print("Training simulation: SGD lr=0.001, momentum=0.99, fp32")
    print(f"{'='*80}")

    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.99,
                                 weight_decay=5e-5, nesterov=True)

    print(header)
    print("-" * len(header))

    for step in range(300):
        optimizer.zero_grad()
        tracked.clear()
        einsum_stats.clear()

        image = torch.randn(B, 1, *patch_size, device='cuda')
        target = (torch.rand(B, N, *patch_size, device='cuda') > 0.95).float()

        output = model(image, text_embed)
        loss_bce = F.binary_cross_entropy_with_logits(output, target)
        probs = torch.sigmoid(output)
        p = probs.view(B, N, -1)
        t = target.view(B, N, -1)
        tp = (p * t).sum(-1)
        fp = (p * (1 - t)).sum(-1)
        fn = ((1 - p) * t).sum(-1)
        dice = 1.0 - (2*tp + 1e-5) / (2*tp + fp + fn + 1e-5)
        loss = loss_bce + dice.mean()

        if torch.isnan(loss):
            print(f"\n*** NaN LOSS at step {step} (fp32) ***")
            break

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
        optimizer.step()

        if step % 5 == 0 or step < 10:
            fx = einsum_stats.get('final_x', (0,0,0))
            fme = einsum_stats.get('final_mask_emb', (0,0,0))
            fo = einsum_stats.get('final_output', (0,0,0))
            tp_ = tracked.get('project_text_embed.2', (0,0,0))
            bp_ = tracked.get('project_bottleneck_embed.2', (0,0,0))
            tn_ = tracked.get('transformer_decoder.norm', (0,0,0))

            gn = f"{grad_norm:.4g}" if not (isinstance(grad_norm, float) and grad_norm != grad_norm) else "NaN"

            print(f"{step:5d} | {loss.item():8.4f} | {gn:>9} | {'fp32':>8} | "
                  f"{fo[0]:9.2f} {fo[1]:9.2f} | {fx[2]:7.3f} {fme[2]:7.3f} | "
                  f"{tp_[2]:9.4f} {bp_[2]:9.4f} {tn_[2]:9.4f}")
    else:
        print(f"\nNo NaN after 300 steps (fp32).")

    for h in handles:
        h.remove()
    print("\nDone.")


if __name__ == '__main__':
    main()
