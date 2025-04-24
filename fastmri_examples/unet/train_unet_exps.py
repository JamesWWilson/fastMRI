#!/usr/bin/env python
# fastmri_examples/unet/train_unet_exps.py
"""
A unified, MPS-enabled training script for systematic U-Net experiments:
- No LoRA / bottleneck+final LoRA / full LoRA
- Freezing encoder and/or decoder
- Manual MPS placement with Lightning running on CPU backend
- Lightning fixes (to-override, checkpoint hotfixes)
- TensorBoard image logging of reconstructions via ReconLogger callback
- Configurable optimizer (RMSprop or AdamW) and adapter learning rate
"""

import multiprocessing as mp
mp.set_start_method("fork", force=True)

import os
import pathlib
import argparse
import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, Callback
from torchvision.utils import make_grid

from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import UnetDataTransform, UnetSample
from fastmri.pl_modules import FastMriDataModule, UnetModule
from fastmri.models.unet_lora2 import UnetLoRA

# ── Override nn.Module.to to ignore unwanted CPU moves ───────────────────────
_orig_to = nn.Module.to
def _to_override(self, device, *args, **kwargs):
    if device == torch.device("cpu") or str(device) == "cpu":
        return self
    return _orig_to(self, device, *args, **kwargs)
nn.Module.to = _to_override

# ── Ensure np.Inf exists ────────────────────────────────────────────────────
if not hasattr(np, "Inf"):
    np.Inf = np.inf

# ── Raw sample filter for Lightning hooks ──────────────────────────────────
def always_true(x):
    return True

# ── Callback to optionally log reconstructions ────────────────────────────
class ReconLogger(Callback):
    """
    Logs a grid of input → prediction → ground truth each validation epoch.
    Works with UnetSample from UnetDataTransform.
    """
    def on_validation_epoch_end(self, trainer, pl_module):
        # Grab first val dataloader
        val_loaders = trainer.datamodule.val_dataloader()
        loader = val_loaders[0] if isinstance(val_loaders, (list, tuple)) else val_loaders

        batch = next(iter(loader))
        if not isinstance(batch, UnetSample):
            return  # silently skip if format unexpected

        inp = batch.image  # [B,C,H,W]
        gt  = batch.target

        # to device
        device = pl_module.device
        inp = inp.to(device)
        gt  = gt.to(device)

        # possible dims swap fix
        B, C, H, W = inp.shape
        expC = pl_module.unet.in_chans
        if C != expC and B == expC:
            inp = inp.permute(1,0,2,3)
            gt  = gt.permute(1,0,2,3)

        # inference
        with torch.no_grad():
            pred = pl_module.unet(inp)

        # grid: [inp;pred;gt]
        imgs = torch.cat([inp, pred, gt], dim=0)
        grid = make_grid(imgs, nrow=inp.size(0), normalize=True)

        trainer.logger.experiment.add_image(
            "Reconstructions",
            grid,
            global_step=trainer.current_epoch
        )

def main():
    args = build_args()
    pl.seed_everything(args.seed)

    # ── Dynamically override configure_optimizers ────────────────────────────
    from torch.optim import RMSprop, AdamW
    from torch.optim.lr_scheduler import StepLR

    def configure_optimizers(self):
        if args.optimizer == "adamw":
            # two param‐groups in one optimizer
            backbone = [p for n,p in self.named_parameters() if "lora_" not in n]
            adapters = [p for n,p in self.named_parameters() if "lora_" in n]
            optimizer = AdamW([
                {"params": backbone,
                 "lr": self.hparams.lr,
                 "weight_decay": self.hparams.weight_decay},
                {"params": adapters,
                 "lr": args.adapter_lr or self.hparams.lr,
                 "weight_decay": 0.0},
            ])
        else:
            optimizer = RMSprop(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        scheduler = StepLR(
            optimizer,
            self.hparams.lr_step_size,
            self.hparams.lr_gamma
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

    UnetModule.configure_optimizers = configure_optimizers

    # ── Prepare paths & transforms ────────────────────────────────────────────
    args.data_path = pathlib.Path(args.data_path)
    args.default_root_dir = pathlib.Path(args.default_root_dir)
    args.raw_sample_filter = always_true

    mask = create_mask_for_mask_type(
        args.mask_type, [args.center_fractions], [args.accelerations]
    )
    train_tf = UnetDataTransform(args.challenge, mask_func=mask, use_seed=False)
    val_tf   = UnetDataTransform(args.challenge, mask_func=mask)
    test_tf  = UnetDataTransform(args.challenge)

    # ── Manual MPS placement ─────────────────────────────────────────────────
    mps_avail = getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    mps_device = torch.device("mps") if mps_avail else torch.device("cpu")
    print(f"→ Forcing model + data onto: {mps_device}")

    def transfer_batch_to_device(self, batch, device, dataloader_idx=None):
        def _move(t: torch.Tensor):
            if t.dtype == torch.float64:
                return t.float().to(mps_device)
            return t.to(mps_device)
        return apply_to_collection(batch, torch.Tensor, _move)
    UnetModule.transfer_batch_to_device = transfer_batch_to_device

    def on_train_start(self):
        print(f"🏷️  All model parameters are on: {next(self.parameters()).device}")
    UnetModule.on_train_start = on_train_start

    # ── DataModule ────────────────────────────────────────────────────────────
    dm = FastMriDataModule(
        data_path           = args.data_path,
        challenge           = args.challenge,
        train_transform     = train_tf,
        val_transform       = val_tf,
        test_transform      = test_tf,
        test_split          = "val",
        sample_rate         = 1.0,
        batch_size          = args.batch_size,
        num_workers         = args.num_workers,
        distributed_sampler = False,
    )

    # ── Model instantiation ───────────────────────────────────────────────────
    model = UnetModule(
        in_chans        = args.in_chans,
        out_chans       = args.out_chans,
        chans           = args.chans,
        num_pool_layers = args.num_pool_layers,
        drop_prob       = args.drop_prob,
        lr              = args.lr,
        lr_step_size    = args.lr_step_size,
        lr_gamma        = args.lr_gamma,
        weight_decay    = args.weight_decay,
    )

    # inject LoRA if requested
    if args.lora_mode in ("bottleneck_final", "all"):
        lora_net = UnetLoRA(
            args.in_chans, args.out_chans,
            args.chans, args.num_pool_layers,
            args.drop_prob,
            r       = args.lora_rank,
            alpha   = args.lora_alpha,
            dropout = args.lora_dropout,
        )
        lora_net.freeze_base()
        model.unet = lora_net

    # freeze encoder/decoder if requested
    if args.freeze_encoder:
        for blk in model.unet.down_sample_layers:
            for p in blk.parameters(): p.requires_grad = False
        for p in model.unet.conv.parameters(): p.requires_grad = False
    if args.freeze_decoder:
        for blk in model.unet.up_transpose_conv:
            for p in blk.parameters(): p.requires_grad = False
        for blk in model.unet.up_conv:
            for p in blk.parameters(): p.requires_grad = False

    # optionally resume from a raw state_dict
    if args.resume_from_checkpoint:
        sd = torch.load(args.resume_from_checkpoint, map_location="cpu")
        model.unet.load_state_dict(sd)

    # ── Move the whole model to MPS ──────────────────────────────────────────
    model = model.to(mps_device)
    print(f"🖥️  Model parameters now on: {next(model.parameters()).device}")


    # ── Checkpoint & logging callbacks ───────────────────────────────────────
    ckpt_dir = args.default_root_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = ModelCheckpoint(
        dirpath    = ckpt_dir,
        monitor    = "validation_loss",
        mode       = "min",
        save_top_k = 1,
    )

    tb_logger = TensorBoardLogger(
        save_dir        = args.default_root_dir,
        name            = "tb_logs",
        default_hp_metric = False,
    )
    lr_monitor   = LearningRateMonitor(logging_interval="epoch")

    cbs = [checkpoint_cb, lr_monitor]
    if args.recon_log:
        cbs.append(ReconLogger())

    # ── Trainer ────────────────────────────────────────────────────────────────
    trainer = Trainer.from_argparse_args(
        args,
        callbacks    = cbs,
        logger       = tb_logger,
        deterministic = True,
        num_sanity_val_steps=0
    )

    if args.mode == "train":
        trainer.fit(model, datamodule=dm)
    else:
        trainer.test(model, datamodule=dm)


def build_args():
    core = argparse.ArgumentParser(add_help=False)
    core.add_argument("--mode", choices=("train","test"), default="train")
    core.add_argument("--challenge",    choices=("singlecoil","multicoil"), required=True)
    core.add_argument("--data_path",    required=True)
    core.add_argument("--mask_type",    choices=("random","equispaced_fraction"), default="random")
    core.add_argument("--center_fractions", type=float, default=0.08)
    core.add_argument("--accelerations",    type=int,   default=4)
    core.add_argument("--batch_size",        type=int,   default=8)
    core.add_argument("--num_workers",       type=int,   default=12)
    core.add_argument("--in_chans",          type=int,   default=1)
    core.add_argument("--out_chans",         type=int,   default=1)
    core.add_argument("--chans",             type=int,   default=32)
    core.add_argument("--num_pool_layers",   type=int,   default=4)
    core.add_argument("--drop_prob",         type=float, default=0.0)
    core.add_argument("--lr",                type=float, default=1e-4)
    core.add_argument("--adapter_lr",        type=float, default=None,
                      help="Learning rate for LoRA adapters (if using AdamW)")
    core.add_argument("--optimizer",
                      choices=("rmsprop","adamw"),
                      default="rmsprop",
                      help="Optimizer to use")
    core.add_argument("--lr_step_size",      type=int,   default=40)
    core.add_argument("--lr_gamma",          type=float, default=0.1)
    core.add_argument("--weight_decay",      type=float, default=0.0)
    core.add_argument("--max_epochs",        type=int,   default=5)
    core.add_argument("--resume_from_checkpoint", type=str, default=None)
    core.add_argument("--freeze_encoder", action="store_true")
    core.add_argument("--freeze_decoder", action="store_true")
    core.add_argument("--lora_mode",
                      choices=("none","bottleneck_final","all"),
                      default="none")
    core.add_argument("--lora_rank",    type=int,   default=4)
    core.add_argument("--lora_alpha",   type=float, default=16)
    core.add_argument("--lora_dropout", type=float, default=0.1)
    core.add_argument("--seed",         type=int,   default=42)
    core.add_argument("--recon_log",    action="store_true",
                      help="Log sample reconstructions each val epoch")

    parser = argparse.ArgumentParser(parents=[core], conflict_handler="resolve")
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(
        accelerator      = "cpu",
        devices          = 1,
        default_root_dir = str(
            fetch_dir("log_path", pathlib.Path("../../fastmri_dirs.yaml"))
            / "unet" / "exps"
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
