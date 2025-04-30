#!/usr/bin/env python
# fastmri_examples/unet/train_unet_demo_JW_multi.py

import resource
import torch
torch.set_float32_matmul_precision('high')

# # 1) optionally cap CPU address‐space to 20 GB
# try:
#     resource.setrlimit(resource.RLIMIT_AS, (20 * 1024**3, 20 * 1024**3))
# except Exception:
#     pass

# # 2) cap each GPU to 20 GB
# max_bytes = 20 * 1024**3
# for dev in range(torch.cuda.device_count()):
#     total = torch.cuda.get_device_properties(dev).total_memory
#     frac = max_bytes / total
#     frac = min(max(frac, 0.0), 1.0)
#     torch.cuda.set_per_process_memory_fraction(frac, device=dev)




import pytorch_lightning as pl

import time
from pytorch_lightning.callbacks import Callback

# class WallClockCallback(Callback):
#     def on_fit_start(self, trainer, pl_module):
#         self._start_time = time.time()
#     def on_fit_end(self, trainer, pl_module):
#         elapsed = time.time() - self._start_time
#         print(f"⏱️  Total wall-clock training time: {elapsed:.1f} s")

from pytorch_lightning.utilities import rank_zero_only

class WallClockCallback(Callback):
    """Record wall-clock time between the start and end of trainer.fit()."""
    def on_fit_start(self, trainer, pl_module):
        # stamp the time when .fit() begins
        self._t0 = time.time()

    @rank_zero_only
    def on_fit_end(self, trainer, pl_module):
        # only print from rank zero in DDP
        elapsed = time.time() - self._t0
        print(f"⏱️  Total wall-clock training time: {elapsed:.1f} seconds")



import multiprocessing as mp
mp.set_start_method("fork", force=True)

import os
import pathlib
from argparse import ArgumentParser

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# from pytorch_lightning.plugins.training_type.ddp import DDPPlugin
from pytorch_lightning.strategies import DDPStrategy

from fastmri.data.mri_data import fetch_dir
from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import UnetDataTransform
from fastmri.pl_modules import FastMriDataModule, UnetModule
from fastmri.models.unet_lora import UnetLoRA

# ── HOTFIX: override UnetModule.configure_optimizers to use self.hparams correctly ──
def _configure_optimizers(self):
    optim = torch.optim.RMSprop(
        self.parameters(),
        lr=self.hparams.lr,
        weight_decay=self.hparams.weight_decay,
    )
    sched = torch.optim.lr_scheduler.StepLR(
        optim,
        step_size=self.hparams.lr_step_size,
        gamma=self.hparams.lr_gamma,
    )
    return [optim], [{"scheduler": sched, "interval": "epoch"}]

UnetModule.configure_optimizers = _configure_optimizers


def cli_main(args):
    pl.seed_everything(args.seed)

    args.data_path = pathlib.Path(args.data_path)
    args.default_root_dir = pathlib.Path(args.default_root_dir)

    # ── build mask & transforms ──
    mask = create_mask_for_mask_type(
        args.mask_type, [args.center_fractions], [args.accelerations]
    )
    train_tf = UnetDataTransform(args.challenge, mask_func=mask, use_seed=False)
    val_tf   = UnetDataTransform(args.challenge, mask_func=mask)

    # ── data module ──
    dm = FastMriDataModule(
        data_path           = args.data_path,
        challenge           = args.challenge,
        train_transform     = train_tf,
        val_transform       = val_tf,
        test_transform      = val_tf,
        test_split          = "val",
        sample_rate         = 1.0,
        batch_size          = args.batch_size,
        num_workers         = args.num_workers,
        distributed_sampler = False,   # DDPPlugin will handle sampling
    )

    # ── model ──
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

    # # ── freeze encoder if requested ──
    # if args.freeze_encoder:
    #     # down‐sampling path
    #     for blk in model.unet.down_sample_layers:
    #         for p in blk.parameters():
    #             p.requires_grad = False
    #     # the very first conv
    #     for p in model.unet.conv.parameters():
    #         p.requires_grad = False
    #     print("Encoder frozen: those weights will not be updated.")

    # # ── freeze decoder if requested ──
    # if args.freeze_decoder:
    #     for blk in getattr(model.unet, "up_transpose_conv", []):
    #         for p in blk.parameters(): p.requires_grad = False
    #     for blk in getattr(model.unet, "up_conv", []):
    #         for p in blk.parameters(): p.requires_grad = False
    #     print(" Decoder frozen")


    # ── load pretrained U-Net weights if given ──
    if args.pretrained_ckpt:
        # load full checkpoint onto CPU
        ck = torch.load(args.pretrained_ckpt, map_location="cpu")
        sd = ck.get("state_dict", ck)
        # copy into model (still on CPU)
        model.unet.load_state_dict(sd, strict=False)

        print("⟳ Loaded pretrained weights via Lightning from", args.pretrained_ckpt)

    # Build the model with or without LoRA based on args
    if args.lora_mode != "none":
        # Create LoRA-enabled model
        lora_net = UnetLoRA(
            args.in_chans, args.out_chans,
            args.chans, args.num_pool_layers,
            args.drop_prob,
            r=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            mode=args.lora_mode,
        )
        
        # Load pretrained weights if provided
        if args.pretrained_ckpt:
            ck = torch.load(args.pretrained_ckpt, map_location="cpu")
            sd = ck.get("state_dict", ck)
            lora_net.load_state_dict(sd, strict=False)
            print(f"⟳ Loaded pretrained weights from {args.pretrained_ckpt}")
        
        # Apply the selected training strategy
        if args.train_strategy == "full":
            # Train all parameters (both base and LoRA)
            print("Training all parameters (base network + LoRA adapters)")
        
        elif args.train_strategy == "freeze_encoder":
            # Freeze encoder, train decoder (including LoRA adapters)
            lora_net.freeze_encoder()
            print("Encoder frozen: training decoder (base weights + LoRA adapters)")
        
        elif args.train_strategy == "freeze_decoder":
            # Freeze decoder, train encoder (including LoRA adapters)
            lora_net.freeze_decoder()
            print("Decoder frozen: training encoder (base weights + LoRA adapters)")
            
        elif args.train_strategy == "lora_only":
            # Only train LoRA parameters, freeze everything else
            # lora_net.unfreeze_lora_only()
            lora_net.freeze_base()
            print("Base network frozen: training only LoRA adapters")
        
        # Replace the UNet in the model with our LoRA-enabled version
        model.unet = lora_net
        
        # Print parameter statistics
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Training {trainable:,}/{total:,} params ({100*trainable/total:.1f}%)")

    else:
        # Regular UNet without LoRA
        if args.train_strategy == "freeze_encoder":
            # Manually freeze encoder layers
            for blk in model.unet.down_sample_layers:
                for p in blk.parameters():
                    p.requires_grad = False
            print("Encoder frozen: those weights will not be updated")
            
        elif args.train_strategy == "freeze_decoder":
            # Manually freeze decoder layers
            for p in model.unet.conv.parameters():
                p.requires_grad = False
            for blk in getattr(model.unet, "up_transpose_conv", []):
                for p in blk.parameters():
                    p.requires_grad = False
            for blk in getattr(model.unet, "up_conv", []):
                for p in blk.parameters():
                    p.requires_grad = False
            print("Decoder frozen: those weights will not be updated")


    # ── callbacks & loggers ──
    ckpt_dir = args.default_root_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    ckpt_cb = ModelCheckpoint(
        dirpath                 = ckpt_dir,
        monitor                 = "val_loss",
        mode                    = "min",
        save_top_k              = 1,
        save_on_train_epoch_end = False,
    )
    lr_mon = LearningRateMonitor(logging_interval="epoch")
    tb_logger  = TensorBoardLogger(save_dir=args.default_root_dir, name="tb_logs", default_hp_metric=False)
    csv_logger = CSVLogger(save_dir=args.default_root_dir, name="csv_logs")
    callbacks = [WallClockCallback(), ckpt_cb, lr_mon]

    # ── trainer ──
    trainer = Trainer(
        accelerator               = "gpu",
        devices                   = args.gpus,
        strategy                  = "ddp",  #DDPPlugin(find_unused_parameters=False),
        # strategy                  = DDPStrategy(find_unused_parameters=False),
        precision                 = "16-mixed", #args.precision,
        max_epochs                = args.max_epochs,
        default_root_dir          = args.default_root_dir,
        callbacks                 = callbacks,
        logger                    = [tb_logger, csv_logger],
        deterministic             = True,
        check_val_every_n_epoch   = args.check_val_every_n_epoch,
        limit_val_batches         = args.limit_val_batches,
        accumulate_grad_batches   = args.accumulate_grad_batches,
    )

    if args.mode == "train":
        trainer.fit(model, datamodule=dm)
        print("✅ Best checkpoint path:", ckpt_cb.best_model_path)
    else:
        trainer.test(model, datamodule=dm)


def build_args():
    parser = ArgumentParser()

    # core fastMRI args
    parser.add_argument("--mode", choices=("train","test"), default="train")
    parser.add_argument("--challenge", choices=("singlecoil","multicoil"), required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--mask_type", choices=("random","equispaced_fraction"), default="random")
    parser.add_argument("--center_fractions", type=float, default=0.08)
    parser.add_argument("--accelerations", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)

    # Unet hyperparameters
    parser.add_argument("--in_chans", type=int, default=1)
    parser.add_argument("--out_chans", type=int, default=1)
    parser.add_argument("--chans", type=int, default=32)
    parser.add_argument("--num_pool_layers", type=int, default=4)
    parser.add_argument("--drop_prob", type=float, default=0.0)

    # optimizer / scheduler
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_step_size", type=int, default=40)
    parser.add_argument("--lr_gamma", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # training config
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--gpus", type=int, default=torch.cuda.device_count())
    parser.add_argument("--precision", type=str, default="32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="how many batches to accumulate before stepping optimizer")
    
    # LoRA options
    parser.add_argument("--pretrained_ckpt", type=str, default=None)
    # Modify LoRA options
    parser.add_argument("--lora_mode", 
                        choices=("none", "bottleneck_only", "bottleneck_final", "decoder_only", "all"), 
                        default="none",
                        help="Where to apply LoRA adapters in the network")
    parser.add_argument("--lora_rank", type=int, default=4)
    parser.add_argument("--lora_alpha", type=float, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    parser.add_argument("--freeze_encoder", action="store_true",help="freeze all encoder (down-sampling) weights")
    parser.add_argument("--freeze_decoder", action="store_true",help="freeze all decoder (up-sampling) weights")

    
    # Training strategy
    parser.add_argument("--train_strategy", 
                        choices=("full", "freeze_encoder", "freeze_decoder", "lora_only"), 
                        default="full",
                        help="Which training strategy to use")
    

    # validation frequency
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1,
                        help="how often to run validation")
    parser.add_argument("--limit_val_batches", type=float, default=1.0,
                        help="fraction of validation batches to run")

    # logging & checkpointing
    from pathlib import Path
    config_path = Path(os.environ.get("FASTMRI_CONFIG", Path.home()/ "fastmri_dirs.yaml"))
    default_root = fetch_dir("log_path", config_path) / "unet" / "finetune"
    parser.add_argument("--default_root_dir", type=str, default=str(default_root),
                        help="Where to write logs & checkpoints")

    return parser.parse_args()


if __name__=="__main__":
    cli_main(build_args())
