# ==================================================================================
# File: train_evaluator_demo.py
# Purpose: Train a light-weight uncertainty evaluator on top of a frozen pretrained U-Net
# ==================================================================================
#!/usr/bin/env python
import multiprocessing as mp
mp.set_start_method("fork", force=True)

import os, pathlib
from argparse import ArgumentParser
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch.optim import RMSprop
from torch.optim.lr_scheduler import StepLR

from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import UnetDataTransform
from fastmri.pl_modules import FastMriDataModule
from fastmri.models import Unet
from fastmri.models.uncertainty_evaluator import UncertaintyEvaluator

# ── HOTFIX: Match optimizer/scheduler signature for UnetModule and EvalModule ──
def configure_optimizers(self):
    optim = RMSprop(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
    sched = StepLR(optim, step_size=self.lr_step_size, gamma=self.lr_gamma)
    return [optim], [{"scheduler": sched, "interval": "epoch"}]

class EvalModule(pl.LightningModule):
    """LightningModule to run frozen pretrained U-Net + trainable evaluator"""
    def __init__(self, unet_ckpt, lr=1e-3, lr_step_size=40, lr_gamma=0.1, weight_decay=0.0):
        super().__init__()
        # Load & freeze U-Net
        state = torch.load(unet_ckpt, map_location='cpu')
        sd = state.get('state_dict', state)
        clean = {k.replace('model.', '').replace('unet.', ''): v for k, v in sd.items()}
        self.unet = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=4)
        self.unet.load_state_dict(clean, strict=False)
        self.unet.eval()
        for p in self.unet.parameters():
            p.requires_grad = False

        # Uncertainty evaluator
        self.evaluator = UncertaintyEvaluator(chans=32, num_layers=4)
        # optimizer hyperparams
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay

    def forward(self, image, mean, std):
        with torch.no_grad():
            recon = self.unet(image.unsqueeze(1)).squeeze(1)
            recon = recon * std.unsqueeze(-1).unsqueeze(-1) + mean.unsqueeze(-1).unsqueeze(-1)
        pred_unc = self.evaluator(recon.unsqueeze(1))
        return pred_unc, recon

    def training_step(self, batch, batch_idx):
        image, target, mean, std, *_ = batch
        pred_unc, recon = self(image, mean, std)
        true_err = (recon - target).abs().unsqueeze(1)
        loss = torch.nn.functional.mse_loss(pred_unc, true_err)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        image, target, mean, std, *_ = batch
        pred_unc, recon = self(image, mean, std)
        true_err = (recon - target).abs().unsqueeze(1)
        loss = torch.nn.functional.mse_loss(pred_unc, true_err)
        self.log('val_loss', loss)

    # attach optimizer configuration
    configure_optimizers = configure_optimizers

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--data_path',       type=pathlib.Path, required=True)
    parser.add_argument('--challenge',       choices=('singlecoil','multicoil'), required=True)
    parser.add_argument('--mask_type',       choices=('random','equispaced_fraction'), default='random')
    parser.add_argument('--center_fractions',type=float, default=0.08)
    parser.add_argument('--accelerations',   type=int,   default=4)
    parser.add_argument('--batch_size',      type=int,   default=4)
    parser.add_argument('--num_workers',     type=int,   default=4)
    parser.add_argument('--pretrained_unet_ckpt', type=pathlib.Path, required=True,
                        help='Path to pretrained U-Net state_dict')
    parser.add_argument('--lr',             type=float, default=1e-3)
    parser.add_argument('--lr_step_size',   type=int,   default=40)
    parser.add_argument('--lr_gamma',       type=float, default=0.1)
    parser.add_argument('--weight_decay',   type=float, default=0.0)
    parser.add_argument('--max_epochs',     type=int,   default=10)
    parser.add_argument('--gpus',           type=int,   default=(torch.cuda.device_count() or 0))
    parser.add_argument('--precision',      type=str,   default='32')
    parser.add_argument('--seed',           type=int,   default=42)
    parser.add_argument('--freeze_encoder', action='store_true')
    parser.add_argument('--default_root_dir', type=pathlib.Path, default=pathlib.Path('./logs_evaluator'))
    args = parser.parse_args()

    pl.seed_everything(args.seed)

    # transforms + data module
    mask = create_mask_for_mask_type(args.mask_type, [args.center_fractions], [args.accelerations])
    train_t = UnetDataTransform(args.challenge, mask_func=mask, use_seed=False)
    val_t   = UnetDataTransform(args.challenge, mask_func=mask)
    dm = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_t,
        val_transform=val_t,
        test_transform=val_t,
        test_split='val',
        sample_rate=1.0,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup(stage='fit')

    # init model
    model = EvalModule(
        unet_ckpt=args.pretrained_unet_ckpt,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
    )
    if args.freeze_encoder:
        for blk in model.unet.down_sample_layers:
            for p in blk.parameters(): p.requires_grad=False
    
    # callbacks + logger
    ckpt_dir = args.default_root_dir / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_cb = ModelCheckpoint(dirpath=ckpt_dir, monitor='val_loss', mode='min', save_top_k=1)
    lr_mon  = LearningRateMonitor(logging_interval='epoch')
    tb_log  = TensorBoardLogger(save_dir=args.default_root_dir, name='tb_logs')
    csv_log = CSVLogger(save_dir=args.default_root_dir, name='csv_logs')

    trainer = pl.Trainer(
        accelerator='gpu' if args.gpus>0 else 'cpu',
        devices=args.gpus or 1,
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=[ckpt_cb, lr_mon],
        logger=[tb_log, csv_log]
    )
    trainer.fit(model, datamodule=dm)
