"""Training loop: SGD with Nesterov momentum and polynomial learning-rate decay, mixed
precision, a fixed number of iterations per epoch and periodic validation (nnU-Net)."""
import json
import math
import os
import shutil
import time
from pathlib import Path

import torch
import wandb
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import PolynomialLR
from tqdm import tqdm

from data_loading import create_data_loaders
from losses import get_loss_function
from metrics import MetricsCalculator
from model import count_parameters, create_model
from utils import AverageMeter, load_checkpoint, save_checkpoint, set_seed


class Trainer:
    """`init_from`: continue model, optimizer, scheduler and epoch from a checkpoint in a
    new run with the current config (used to fork runs from a common parent).
    `init_weights`: initialise only the weights. `resume_dir`: continue an interrupted run
    with its own config."""

    def __init__(self, config, fold, init_from=None, init_weights=None, resume_dir=None):
        ckpt = None
        if resume_dir is not None:
            ckpt = load_checkpoint(Path(resume_dir) / "checkpoint.pth")
            config, fold = ckpt["config"], ckpt["fold"]
        elif init_from is not None:
            ckpt = load_checkpoint(init_from)
        self.config, self.fold = config, fold
        t = config.training

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        set_seed(config.seed + fold)
        self.amp = config.mixed_precision
        self.scaler = GradScaler(self.device.type, enabled=self.amp)
        self.model = create_model(config).to(self.device)
        print(f"{config.model.architecture}-{config.model.model_size}: "
              f"{count_parameters(self.model):,} parameters", flush=True)
        if init_weights is not None:
            self.model.load_state_dict(load_checkpoint(init_weights)["model_state_dict"])
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=t.initial_lr, momentum=t.momentum,
                                         weight_decay=t.weight_decay, nesterov=True)
        self.criterion = get_loss_function(config)
        if t.interleaved and not hasattr(self.criterion, "stage_forward"):
            raise ValueError(f"loss {t.loss_function!r} does not support interleaved training")
        self.metrics = MetricsCalculator(
            ignore_index=t.ignore_index, compute_betti_matching=t.compute_betti_matching_metric,
            compute_calibration=t.compute_calibration_metrics,
            betti_num_workers=t.betti_metric_num_workers, betti_max_samples=t.betti_metric_max_samples,
            betti_sample_stride=t.betti_metric_sample_stride)
        self.train_loader, self.val_loader = create_data_loaders(config, fold)

        self.epoch = 0
        self.scheduler = PolynomialLR(self.optimizer, total_iters=t.max_epochs, power=t.poly_lr_pow)
        if ckpt is not None:
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if ckpt.get("scaler_state_dict"):
                self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            self.epoch = ckpt["epoch"] + 1
            print(f"continuing at epoch {self.epoch} from {resume_dir or init_from}", flush=True)

        run_id = Path(resume_dir).name.split("_", 2)[-1] if resume_dir else None
        if os.environ.get("WANDB_MODE", "").lower() != "disabled" and wandb.run is None:
            wandb.init(project=config.wandb.project, entity=config.wandb.entity,
                       name=config.wandb.run_name or f"fold_{fold}", tags=list(config.wandb.tags),
                       config=dict(config.to_dict(), fold=fold), id=run_id,
                       resume="must" if run_id else None)
        if run_id is None:
            run_id = wandb.run.id if wandb.run is not None else time.strftime("%Y%m%d-%H%M%S")
        self.output_dir = Path(resume_dir) if resume_dir else Path(config.output_dir) / f"fold_{fold}_{run_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if resume_dir is None:
            (self.output_dir / "run.json").write_text(json.dumps(
                {"config": config.to_dict(), "fold": fold, "init_from": init_from,
                 "init_weights": init_weights}, indent=1))

    # ------------------------------------------------------------------ helpers

    def _next_batch(self):
        try:
            batch = next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            batch = next(self._train_iter)
        out = {k: v.to(self.device, non_blocking=True) for k, v in batch.items() if torch.is_tensor(v)}
        if out["label"].dim() == len(self.config.data.patch_size) + 2:
            out["label"] = out["label"].squeeze(1)
        return out

    def _step(self):
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 12)
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _log(self, loss, components, step):
        if wandb.run is None:
            return
        log = {f"train/loss_{k}": (v.item() if torch.is_tensor(v) else v) for k, v in components.items()}
        log.update({"train/loss_iter": loss, "train/learning_rate": self.optimizer.param_groups[0]["lr"]})
        wandb.log(log, step=step)

    # ----------------------------------------------------------------- training

    def train_epoch(self):
        """One epoch of num_iterations_per_epoch batches with grad_accum batches per step."""
        t = self.config.training
        self.model.train()
        self.train_loader.dataset.set_epoch(self.epoch)
        self._train_iter = iter(self.train_loader)
        meter, consecutive_nan, k = AverageMeter(), 0, 0
        for it in tqdm(range(t.num_iterations_per_epoch), desc=f"epoch {self.epoch}", ncols=100):
            batch = self._next_batch()
            if k == 0:
                self.optimizer.zero_grad()
            with autocast(device_type=self.device.type, enabled=self.amp):
                loss, components = self.criterion(self.model(batch["image"]), batch["label"],
                                                  skel=batch.get("skel"))
            value = loss.item()
            if not math.isfinite(value):
                # Skip the whole accumulation group; stop after ten in a row.
                self.optimizer.zero_grad()
                k, consecutive_nan = 0, consecutive_nan + 1
                if consecutive_nan >= 10:
                    raise RuntimeError("10 consecutive non-finite losses")
                continue
            consecutive_nan = 0
            step_now = k + 1 >= t.grad_accum or it + 1 >= t.num_iterations_per_epoch
            self.scaler.scale(loss / t.grad_accum).backward()
            if step_now:
                self._step()
            k = 0 if step_now else k + 1
            meter.update(value, batch["image"].shape[0])
            self._log(value, components, self.epoch * t.num_iterations_per_epoch + it)
        return meter.avg

    def train_epoch_interleaved(self):
        """Two micro-batches per optimizer step, both forwarded with the same weights, so
        that this is a step on a batch of 2 * batch_size. The CPU part of the loss of one
        micro-batch runs while the GPU computes the forward or backward pass of the other."""
        t = self.config.training
        self.model.train()
        self.train_loader.dataset.set_epoch(self.epoch)
        self._train_iter = iter(self.train_loader)
        self.optimizer.zero_grad(set_to_none=True)
        crit, meter = self.criterion, AverageMeter()

        def forward(b):
            with autocast(device_type=self.device.type, enabled=self.amp):
                return crit.stage_forward(self.model(b["image"]), b["label"])

        def finish(s):
            with autocast(device_type=self.device.type, enabled=self.amp):
                return crit.stage_finish(s)

        n_micro = t.num_iterations_per_epoch
        for g in tqdm(range(n_micro // 2), desc=f"epoch {self.epoch}", ncols=100):
            b0, b1 = self._next_batch(), self._next_batch()
            s0 = forward(b0)
            crit.stage_masks(s0)
            s1 = forward(b1)                       # GPU: forward 1
            loss0, _ = finish(s0)                  # CPU: matching 0
            crit.stage_masks(s1)
            self.scaler.scale(loss0 / 2).backward()   # GPU: backward 0
            loss1, components = finish(s1)            # CPU: matching 1
            self.scaler.scale(loss1 / 2).backward()
            self._step()
            self.optimizer.zero_grad(set_to_none=True)
            for loss, b in ((loss0, b0), (loss1, b1)):
                if math.isfinite(loss.item()):
                    meter.update(loss.item(), b["image"].shape[0])
            self._log(loss1.item(), components, self.epoch * n_micro + 2 * g)
        if n_micro % 2:
            b = self._next_batch()
            s = forward(b)
            crit.stage_masks(s)
            loss, _ = finish(s)
            self.scaler.scale(loss).backward()
            self._step()
            self.optimizer.zero_grad(set_to_none=True)
            if math.isfinite(loss.item()):
                meter.update(loss.item(), b["image"].shape[0])
        return meter.avg

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        self.metrics.reset()
        for batch in tqdm(self.val_loader, desc="validation", ncols=100, leave=False):
            label = batch["label"].to(self.device)
            if label.dim() == len(self.config.data.patch_size) + 2:
                label = label.squeeze(1)
            self.metrics.update(self.model(batch["image"].to(self.device)), label)
        return self.metrics.compute()

    def fit(self):
        t = self.config.training
        last = t.max_epochs if t.stop_at_epoch is None else min(t.stop_at_epoch, t.max_epochs)
        for epoch in range(self.epoch, last):
            self.epoch = epoch
            start = time.time()
            train_loss = self.train_epoch_interleaved() if t.interleaved else self.train_epoch()
            lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()
            log = {"epoch": epoch, "train/loss": train_loss, "train/learning_rate": lr,
                   "time/train": time.time() - start}
            if (epoch + 1) % t.val_check_interval == 0:
                start = time.time()
                log.update({f"val/{k}": v for k, v in self.validate().items()})
                log["time/val"] = time.time() - start
            if wandb.run is not None:
                wandb.log(log)
            with open(self.output_dir / "history.jsonl", "a") as f:
                f.write(json.dumps(log) + "\n")
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in log.items()),
                  flush=True)

            save_checkpoint({
                "epoch": epoch, "fold": self.fold, "config": self.config.to_dict(),
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
            }, self.output_dir / "checkpoint.pth")
            if epoch + 1 in tuple(t.milestone_epochs) or epoch + 1 == last:
                shutil.copy2(self.output_dir / "checkpoint.pth", self.output_dir / f"epoch{epoch + 1}.pth")
        return self.output_dir


def train(config, fold, **kwargs):
    trainer = Trainer(config, fold, **kwargs)
    out = trainer.fit()
    if wandb.run is not None:
        wandb.finish()
    return out
