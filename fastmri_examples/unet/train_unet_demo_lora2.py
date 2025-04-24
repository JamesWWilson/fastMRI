# fastmri_examples/unet/train_unet_demo.py

# TRY FORCING MP 
import multiprocessing as mp
mp.set_start_method("fork", force=True)

import os
import pathlib
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.apply_func import apply_to_collection

from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from pytorch_lightning import Trainer

from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import UnetDataTransform
from fastmri.pl_modules import FastMriDataModule, UnetModule
from fastmri.models.unet_lora2 import UnetLoRA


# hot fix learning rate step scheduler for lighting versions per 
import types
from torch.optim import RMSprop
from torch.optim.lr_scheduler import StepLR

def _configure_optimizers(self):
    optim = RMSprop(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    sched = StepLR(optim, self.lr_step_size, self.lr_gamma)
    return [optim], [{"scheduler": sched, "interval": "epoch"}]

# Directly overwrite the method on the class
UnetModule.configure_optimizers = _configure_optimizers
# You define a function with the correct signature (self is the module instance).
# You assign it on the class itself, so when Lightning does model.configure_optimizers(), it invokes your function with self=model.
# You return the optimizer list and scheduler‐dict in the new Lightning‑2.x format.


# ── 1) patch nn.Module.to so cpu moves do nothing ──
import torch.nn as nn
_orig_to = nn.Module.to
def _to_override(self, device, *args, **kwargs):
    # if they try to move you back to CPU, just ignore it
    if device == torch.device("cpu") or str(device) == "cpu":
        return self
    return _orig_to(self, device, *args, **kwargs)
nn.Module.to = _to_override



import numpy as np
# restore the old alias so Lightning’s checkpoint callback doesn’t break
if not hasattr(np, "Inf"):
    np.Inf = np.inf


# YAN HOT FIX 
def always_true(x):
    return True

def cli_main(args):
    pl.seed_everything(args.seed)

    args.data_path = pathlib.Path(args.data_path)
    args.default_root_dir = pathlib.Path(args.default_root_dir)

    # build the k-space mask
    mask = create_mask_for_mask_type(
        args.mask_type, [args.center_fractions], [args.accelerations]
    )
    train_transform = UnetDataTransform(args.challenge, mask_func=mask, use_seed=False)
    val_transform   = UnetDataTransform(args.challenge, mask_func=mask)
    test_transform  = UnetDataTransform(args.challenge)

    # # data module
    # data_module = FastMriDataModule(
    #     data_path=args.data_path,
    #     challenge=args.challenge,
    #     train_transform=train_transform,
    #     val_transform=val_transform,
    #     test_transform=test_transform,
    #     test_split="val",
    #     test_path=None,
    #     sample_rate=1.0,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     # persistent_workers=True, # try this 
    #     # prefetch_factor=2, # try this 
    #     # pin_memory=True,         # optional but can help
    #     distributed_sampler=False,
    # )

    # # model
    # model = UnetModule(
    #     in_chans=args.in_chans,
    #     out_chans=args.out_chans,
    #     chans=args.chans,
    #     num_pool_layers=args.num_pool_layers,
    #     drop_prob=args.drop_prob,
    #     lr=args.lr,
    #     lr_step_size=args.lr_step_size,
    #     lr_gamma=args.lr_gamma,
    #     weight_decay=args.weight_decay,
    # )

    # # if you passed a .pt (state_dict), load it
    # if args.resume_from_checkpoint and args.resume_from_checkpoint.endswith(".pt"):
    #     sd = torch.load(args.resume_from_checkpoint, map_location="cpu")
    #     model.unet.load_state_dict(sd)

    # # optionally freeze the encoder
    # # ── freeze encoder layers if requested ──
    # if args.freeze_encoder:
    #     # freeze all the "down" conv blocks
    #     for block in model.unet.down_sample_layers:  
    #         for p in block.parameters():
    #             p.requires_grad = False
    #     # also freeze the center ConvBlock if you want
    #     for p in model.unet.conv.parameters():
    #         p.requires_grad = False

    # # prepare checkpoint callback
    # ckpt_dir = pathlib.Path(args.default_root_dir) / "checkpoints"
    # ckpt_dir.mkdir(parents=True, exist_ok=True)
    # checkpoint_cb = pl.callbacks.ModelCheckpoint(
    #     dirpath=ckpt_dir,
    #     monitor="validation_loss",
    #     mode="min",
    #     save_top_k=1,
    # )

    # DETERMINE WHAT WE ARE USING 
    print(f"torch.cuda.is_available() = {torch.cuda.is_available()}")
    print(f"torch.backends.mps.is_available() = {getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()}")


    # **** “force” everything onto MPS **** 
    # pick the MPS device (or CPU fallback)
    mps_device = torch.device("mps") if \
                    getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() \
                 else torch.device("cpu")
    print(f"→ Forcing both model + data onto: {mps_device}")

    # 1) patch your LightningModule so Lightning never tries to move things itself
    def transfer_batch_to_device(self, batch, device, dataloader_idx=None):
        # move every torch.Tensor in your batch to MPS,
        # and if it's float64, cast it to float32 (MPS doesn’t support float64)
        def move(t: torch.Tensor):
            if t.dtype == torch.float64:
                t = t.float()
            return t.to(mps_device)
        return apply_to_collection(batch, torch.Tensor, move)

    UnetModule.transfer_batch_to_device = transfer_batch_to_device

    # 2) patch an on_train_start hook so you see the device
    def on_train_start(self):
        dev = next(self.parameters()).device
        print(f"🏷️  All model parameters are on: {dev}")
    UnetModule.on_train_start = on_train_start

    # build your datamodule & model as before…
    # data module
    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split="val",
        test_path=None,
        sample_rate=1.0,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        # persistent_workers=True, # try this 
        # prefetch_factor=2, # try this 
        # pin_memory=True,         # optional but can help
        distributed_sampler=False,
    )

    # model
    model = UnetModule(
        in_chans=args.in_chans,
        out_chans=args.out_chans,
        chans=args.chans,
        num_pool_layers=args.num_pool_layers,
        drop_prob=args.drop_prob,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
    )

    # ── swap in our LoRA-augmented U-Net ──
    lora_net = UnetLoRA(
         args.in_chans,
         args.out_chans,
         args.chans,
         args.num_pool_layers,
         args.drop_prob,
         r=args.lora_rank,
         alpha=args.lora_alpha,
         dropout=args.lora_dropout,
    )
    lora_net.freeze_base()
    model.unet = lora_net

    # if you passed a .pt (state_dict), load it
    if args.resume_from_checkpoint and args.resume_from_checkpoint.endswith(".pt"):
        sd = torch.load(args.resume_from_checkpoint, map_location="cpu")
        model.unet.load_state_dict(sd)

    # # optionally freeze the encoder
    # # ── freeze encoder layers if requested ──
    # if args.freeze_encoder:
    #     # freeze all the "down" conv blocks
    #     for block in model.unet.down_sample_layers:  
    #         for p in block.parameters():
    #             p.requires_grad = False
    #     # also freeze the center ConvBlock if you want
    #     for p in model.unet.conv.parameters():
    #         p.requires_grad = False

    # prepare checkpoint callback
    ckpt_dir = pathlib.Path(args.default_root_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=ckpt_dir,
        monitor="validation_loss",
        mode="min",
        save_top_k=1,
    )

    # now move the model itself onto MPS
    model = model.to(mps_device)

# ── 1) define your loggers ──
    # this will write TensorBoard event files under:
    #   <default_root_dir>/tb_logs/version_x/
    tb_logger = TensorBoardLogger(
        save_dir=args.default_root_dir,
        name="tb_logs",
        default_hp_metric=False,
        log_graph=False,            # set True if you want to visualize the graph
        # version=f"finetune_ch{args.challenge}" # no version to self increment  /   version=f"lr{args.lr}_bs{args.batch_size}_{int(time.time())}",
    )

    # this will write CSVs under:
    #   <default_root_dir>/csv_logs/version_x/metrics.csv
    csv_logger = CSVLogger(
        save_dir=args.default_root_dir,
        name="csv_logs"
    )

    # ── 2) (optional) track your LR schedule in TB ──
    lr_monitor = LearningRateMonitor(logging_interval="epoch")


    # trainer = pl.Trainer(
    trainer = pl.Trainer.from_argparse_args(
        args,
        # accelerator="cpu",
        # devices=1,
        # precision=args.precision,
        # gradient_clip_val=args.gradient_clip_val,        
        # max_epochs=args.max_epochs,
        # default_root_dir=args.default_root_dir,
        callbacks=[checkpoint_cb, lr_monitor],
        # logger=[tb_logger, csv_logger],    # attach both loggers
        logger=tb_logger, # only one logger
        log_every_n_steps=50,              # flush metrics every 50 steps
        deterministic=True,
    )


    # # normal working 
    # trainer = pl.Trainer(
    #     accelerator= "auto", #"mps",            # force PyTorch-MPS
    #     devices=1,                    # single MPS device
    #     precision=32,           # or "16-mps" if you want mixed precision
    #     max_epochs=args.max_epochs,
    #     default_root_dir=args.default_root_dir,
    #     callbacks=[checkpoint_cb],
    #     deterministic=True,
    #     # no strategy=… here
    # )


    # YAN HOT FIX 
    args.raw_sample_filter = always_true


    # run
    if args.mode == "train":
        trainer.fit(model, datamodule=data_module)
    else:
        trainer.test(model, datamodule=data_module)


def build_args():
    parser = ArgumentParser(description="Fine‑tune the fastMRI U‑Net")

    # mode + data
    parser.add_argument("--mode", choices=("train","test"), default="train")
    parser.add_argument("--challenge", choices=("singlecoil","multicoil"), required=True)
    parser.add_argument("--data_path", required=True, help="Path to singlecoil_train folder")

    # mask
    parser.add_argument("--mask_type", choices=("random","equispaced_fraction"), default="random")
    parser.add_argument("--center_fractions", type=float, default=0.08)
    parser.add_argument("--accelerations", type=int, default=4)

    # loaders
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    # model hyperparams
    parser.add_argument("--in_chans", type=int, default=1)
    parser.add_argument("--out_chans", type=int, default=1)
    parser.add_argument("--chans", type=int, default=256)
    parser.add_argument("--num_pool_layers", type=int, default=4)
    parser.add_argument("--drop_prob", type=float, default=0.0)

    # optimization
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_step_size", type=int, default=40)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # training
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to .pt or .ckpt to start from")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze the down_blocks of the U‑Net")

    parser.add_argument("--lora_rank",    type=int,   default=4,
                          help="LoRA rank (low-rank adapter dimension)")
    parser.add_argument("--lora_alpha",   type=float, default=16,
                          help="LoRA alpha (scaling factor)")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                          help="Dropout for the LoRA adapters")

    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        help="Floating‑point precision: 32, 16, or 16-mps on Apple Silicon",
    )
    parser.add_argument(
        "--gradient_clip_val",
        type=float,
        default=0.0,
        help="Max‐norm for gradient clipping",
    )


    # logging
    default_root = fetch_dir("log_path", pathlib.Path("../../fastmri_dirs.yaml")) / "unet" / "finetune"
    parser.add_argument("--default_root_dir", type=str, default=str(default_root),
                        help="Where to write logs and checkpoints")

    # Now pull in **all** Trainer flags _except_ the ones you’ve already declared.
    # Use conflict_handler="resolve" so that if Trainer defines --default_root_dir,
    # we can re‑declare it ourselves after.
    #
    parser = ArgumentParser(conflict_handler="resolve",
                            parents=[parser],
                            add_help=False)

    # this will add flags like --accelerator, --devices, --precision, --gradient_clip_val, --default_root_dir, etc.
    parser = pl.Trainer.add_argparse_args(parser)

    # now override default_root_dir if you want your own default
    default_root = fetch_dir("log_path", pathlib.Path("../../fastmri_dirs.yaml")) / "unet" / "finetune"
    parser.add_argument("--default_root_dir", type=str, default=str(default_root),
                        help="Where to write logs and checkpoints")
 

    return parser.parse_args()


def run_cli():
    args = build_args()
    cli_main(args)


if __name__ == "__main__":
    run_cli()
