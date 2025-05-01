#!/usr/bin/env python3

import os
import time
import itertools
import torch
import pathlib
import numpy as np
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.regression import MeanSquaredError
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
from torch.utils.data._utils import pin_memory
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from pytorch_lightning.strategies import DDPStrategy


from fastmri.data.subsample import create_mask_for_mask_type
from fastmri.data.transforms import UnetDataTransform
from fastmri.pl_modules import FastMriDataModule
from fastmri.losses import SSIMLoss
from fastmri.evaluate import ssim as compute_ssim, psnr as compute_psnr, nmse as compute_nmse


from mymodels import VisionTransformer, ReconNet

# Let cuDNN pick the fastest convolution algorithms
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True


class LitReconNet(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        # save all hyperparameters
        self.save_hyperparameters(hparams)
        args = self.hparams

        self.ssim_metric_train = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.psnr_metric_train = PeakSignalNoiseRatio(data_range=1.0)
        self.nmse_metric_train = MeanSquaredError()

        self.ssim_metric_val = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.psnr_metric_val = PeakSignalNoiseRatio(data_range=1.0)
        self.nmse_metric_val = MeanSquaredError()

        # build Vision Transformer model
        self.vit = VisionTransformer(
            avrg_img_size=args.avrg_img_size,
            patch_size=args.patch_size,
            in_chans=args.in_chans,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
            drop_rate=args.drop_rate,
            attn_drop_rate=args.attn_drop_rate,
            drop_path_rate=args.drop_path_rate,
            use_pos_embed=args.use_pos_embed,
            pretrained=args.pretrained,
        )
        # wrap in ReconNet for padding/normalization logic
        self.net = ReconNet(self.vit)
        self.criterion = SSIMLoss()

    def forward(self, x):
        # centralize the unsqueeze so every input is 4-D
        if x.dim() == 3:  # [B,H,W]
            x = x.unsqueeze(1)  # → [B,1,H,W]
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y, maxval, *_ = batch

        if x.dim() == 3:
            x, y = x.unsqueeze(1), y.unsqueeze(1)

        if self.training:
            i, j, h_crop, w_crop = T.RandomCrop.get_params(x, (272, 272))
            x = TF.crop(x, i, j, h_crop, w_crop)
            y = TF.crop(y, i, j, h_crop, w_crop)
            if torch.rand(1) < 0.5:
                x, y = TF.hflip(x), TF.hflip(y)
            if torch.rand(1) < 0.5:
                x, y = TF.vflip(x), TF.vflip(y)
            k = int(torch.randint(0, 4, (1,)).item())
            x = torch.rot90(x, k, dims=[2, 3])
            y = torch.rot90(y, k, dims=[2, 3])

        if x.shape[1] > 1:
            x = torch.sqrt((x ** 2).sum(dim=1, keepdim=True))
            y = torch.sqrt((y ** 2).sum(dim=1, keepdim=True))

        y_hat = self.net(x)
        loss = self.criterion(y_hat, y, maxval)

        # metric calculations
        y_hat_clamped = torch.clamp(y_hat, 0, 1)
        self.ssim_metric_train.update(y_hat_clamped, y)
        self.psnr_metric_train.update(y_hat_clamped, y)
        self.nmse_metric_train.update(
            y_hat_clamped.contiguous().view(y_hat_clamped.size(0), -1),
            y.contiguous().view(y.size(0), -1)
        )


        self.log('train_loss', loss, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        return 


    def validation_step(self, batch, batch_idx):
        x, y, maxval, *_ = batch
        if x.dim() == 3:
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)

        if x.size(1) > 1:
            x = torch.sqrt((x**2).sum(dim=1, keepdim=True))
            y = torch.sqrt((y**2).sum(dim=1, keepdim=True))

        y_hat = self.net(x)
        loss = self.criterion(y_hat, y, maxval)

        y_hat_clamped = torch.clamp(y_hat, 0, 1)
        self.ssim_metric_val.update(y_hat_clamped, y)
        self.psnr_metric_val.update(y_hat_clamped, y)
        self.nmse_metric_val.update(y_hat_clamped, y)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True, batch_size=x.size(0))
        return loss


    def test_step(self, batch, batch_idx):
        x, y, maxval, *_ = batch

        # 1) make sure we have a channel-dim
        if x.dim() == 3:              # [B, H, W]
            x = x.unsqueeze(1)        # → [B, 1, H, W]
            y = y.unsqueeze(1)

        # 2) only now collapse multi-coil
        if x.size(1) > 1:             # real coil dim
            x = torch.sqrt((x**2).sum(dim=1, keepdim=True))
            y = torch.sqrt((y**2).sum(dim=1, keepdim=True))

        # proceed with forward / loss
        y_hat = self.net(x)
        loss  = self.criterion(y_hat, y, maxval)

         # normalize predictions
        y_hat_norm = torch.clamp(y_hat / maxval.view(-1, 1, 1, 1), 0, 1)
        y_norm     = torch.clamp(y     / maxval.view(-1, 1, 1, 1), 0, 1)

        ssim_val = self.ssim_metric(y_hat_norm, y_norm)
        psnr_val = self.psnr_metric(y_hat_norm, y_norm)

        # NMSE = ||y_hat - y||^2 / ||y||^2
        nmse_val = torch.norm(y_hat - y) ** 2 / (torch.norm(y) ** 2 + 1e-8)

        self.log("test_loss", loss)
        self.log("test_ssim", ssim_val, prog_bar=True)
        self.log("test_psnr", psnr_val)
        self.log("test_nmse", nmse_val)

        return {"loss": loss, "ssim": ssim_val, "psnr": psnr_val, "nmse": nmse_val}


    def configure_optimizers(self):
        # base LR 0.0005for ViT-L fine-tuning
        base_lr = 0.0005

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=base_lr,
            weight_decay=self.hparams.weight_decay,
        )

        total_epochs = self.hparams.max_epochs  # e.g. 30
        warmup_epochs = 4

        scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_epochs),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
        ],
        milestones=[warmup_epochs],
        )   

        return {
        'optimizer': optimizer,
        'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        'monitor': 'val_ssim',  # we want to maximize SSIM
        }


class WallClockCallback(Callback):
    def on_fit_start(self, trainer, pl_module):
        self._start_time = time.time()

    def on_fit_end(self, trainer, pl_module):
        elapsed = time.time() - self._start_time
        print(f"⏱️  Total wall-clock training time: {elapsed:.1f} seconds")


def build_args():
    parser = ArgumentParser()
    # mode
    parser.add_argument('--mode', choices=('train','test'), default='train')
    # data parameters
    parser.add_argument('--challenge', choices=('singlecoil','multicoil'), required=True)
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--mask_type', choices=('random','equispaced_fraction'), default='random')
    parser.add_argument('--center_fractions', type=float, default=0.08)
    parser.add_argument('--accelerations', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    # model parameters
    parser.add_argument('--avrg_img_size', type=int, default=340)
    parser.add_argument('--patch_size', type=int, default=10)
    parser.add_argument('--depth', type=int, default=8)
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument('--num_heads', type=int, default=16)
    parser.add_argument('--drop_rate', type=float, default=0.0)
    parser.add_argument('--attn_drop_rate', type=float, default=0.0)
    parser.add_argument('--drop_path_rate', type=float, default=0.0)
    parser.add_argument('--use_pos_embed', action='store_true')
    parser.add_argument('--pretrained', action='store_true', help='Use timm pretrained ViT-Base')
    parser.add_argument('--in_chans', type=int, default=1)
    parser.add_argument('--out_chans', type=int, default=1)
    # optimizer
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr_step_size', type=int, default=30)
    parser.add_argument('--lr_gamma', type=float, default=0.1)
    # training setup
    parser.add_argument('--max_epochs', type=int, default=5)
    parser.add_argument('--gpus', type=int, default=torch.cuda.device_count())
    parser.add_argument('--precision', type=str, default='32')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--limit_val_batches', type=float, default=1.0)
    # output and samples
    parser.add_argument('--output_dir', type=str, default='outputs-omar')
    parser.add_argument('--num_samples', type=int, default=5)
    # testing setup
    parser.add_argument('--ckpt_path',  type=str,  default=None,  help='Path to .ckpt file when running in test mode')

    return parser.parse_args()


def cli_main(args):
    import os, itertools
    import pathlib
    import pandas as pd
    import matplotlib.pyplot as plt
    import torch
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger
    from fastmri.data.subsample import create_mask_for_mask_type
    from fastmri.data.transforms import UnetDataTransform
    from fastmri.pl_modules import FastMriDataModule
    from fastmri.evaluate import ssim, psnr, nmse
    from mymodels import VisionTransformer, ReconNet

    # ─── reproducibility & dirs ───
    pl.seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── setup data ──
    mask = create_mask_for_mask_type(
        args.mask_type, [args.center_fractions], [args.accelerations]
    )
    train_tf = UnetDataTransform(args.challenge, mask_func=mask, use_seed=True)
    val_tf   = UnetDataTransform(args.challenge, mask_func=mask)
    args.data_path = pathlib.Path(args.data_path)

    dm = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_tf,
        val_transform=val_tf,
        test_transform=val_tf,
        test_split='val',
        sample_rate=1.0,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        distributed_sampler=False,
    ) 

    # ── override DataModule’s DataLoaders ──
    orig_train = dm.train_dataloader()
    orig_val   = dm.val_dataloader()
    orig_test  = dm.test_dataloader()

    dm.train_dataloader = lambda: DataLoader(
        orig_train.dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=2,
    )
    dm.val_dataloader = lambda: DataLoader(
        orig_val.dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    dm.test_dataloader = lambda: DataLoader(
        orig_test.dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True,
    )

    # ─── model ───
    model = LitReconNet(args)

    # ─── callbacks & loggers ───
    ckpt_dir = os.path.join(args.output_dir, 'checkpoints-omar')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(
        dirpath   = ckpt_dir,
        monitor   = 'val_loss',
        mode      = 'min',
        save_top_k= 1,
    )
    lr_mon    = LearningRateMonitor(logging_interval='epoch')
    tb_logger = TensorBoardLogger(
        save_dir        = args.output_dir,
        name            = 'tb_logs',
        default_hp_metric=False
    )
    csv_logger = CSVLogger(
        save_dir = args.output_dir,
        name     = 'csv_logs'
    )
    callbacks = [WallClockCallback(), ckpt_cb, lr_mon]

    # ─── trainer ───
    trainer = Trainer(
        accelerator='gpu',
        devices    = args.gpus,
        #strategy   = DDPStrategy(find_unused_parameters=True),   # USE THIS IF YOU ARE USING MULTI GPU.  !!!!
        precision  = args.precision,
        max_epochs = args.max_epochs,
        #profiler   = 'simple',
        callbacks  = callbacks,
        logger     = [tb_logger, csv_logger],
        deterministic           = True,
        check_val_every_n_epoch = args.check_val_every_n_epoch,
        limit_val_batches       = args.limit_val_batches,
        accumulate_grad_batches = args.accumulate_grad_batches,
    )

    # ─── train or test ───
    if args.mode == 'train':
        trainer.fit(model, datamodule=dm)
        print("✅ Best checkpoint path:", ckpt_cb.best_model_path)

        # ─── log final metrics ───
        ssim_tensor = model.ssim_metric_val.compute()
        ssim = ssim_tensor[0].item() if isinstance(ssim_tensor, tuple) else ssim_tensor.item()

        psnr_tensor = model.psnr_metric_val.compute()
        psnr = psnr_tensor[0].item() if isinstance(psnr_tensor, tuple) else psnr_tensor.item()

        nmse_tensor = model.nmse_metric_val.compute()
        nmse = nmse_tensor[0].item() if isinstance(nmse_tensor, tuple) else nmse_tensor.item()

        print(f"\nFinal Val SSIM: {ssim:.4f}")
        model.ssim_metric_val.reset()

        print(f"\nFinal Val PSNR: {psnr:.4f}")
        model.psnr_metric_val.reset()

        print(f"\nFinal Val NMSE: {nmse:.4f}")
        model.nmse_metric_val.reset()

        # ─── 1) plot loss & SSIM ───
        csv_logs_dir = csv_logger.log_dir
        metrics_path = os.path.join(csv_logs_dir, 'metrics.csv')
        if not os.path.exists(metrics_path):
            print(f"❌ CSV file not found at {metrics_path}")
        else:
            df = pd.read_csv(metrics_path)
            df_e = df[df['step'] == 0].reset_index(drop=True)

            plt.figure(figsize=(6, 4))
            if 'train_loss_epoch' in df_e:
                plt.plot(df_e['epoch'], df_e['train_loss_epoch'], 'r+-', label='Train Loss')
            if 'val_ssim' in df_e:
                plt.plot(df_e['epoch'], df_e['val_ssim'], 'g*-', label='Val SSIM')
            if 'val_loss' in df_e:
                plt.plot(df_e['epoch'], df_e['val_loss'], 'b^-', label='Val Loss')

            plt.xlabel('Epoch')
            plt.ylabel('Metric')
            plt.grid(True)
            plt.legend()
            os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)
            plt.savefig(os.path.join(args.output_dir, 'plots', 'loss_ssim_history.png'),
                        bbox_inches='tight', dpi=150)
            print("📈 Exported Loss/SSIM plot.")
            plt.close()

        # ─── 2) sample reconstructions & metrics ───
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        best_model = LitReconNet.load_from_checkpoint(
            ckpt_cb.best_model_path,
            hparams=args
        ).to(device).eval()

        os.makedirs(os.path.join(args.output_dir, 'samples'), exist_ok=True)
        samples = []
        for idx, batch in enumerate(itertools.islice(dm.test_dataloader(), args.num_samples)):
            x, y, maxval, *_ = batch
            if x.dim() == 3:
                x, y = x.unsqueeze(1), y.unsqueeze(1)

            with torch.no_grad():
                out = best_model(x.to(device)).cpu()

           # get 2D tensors
            xi_t, xr_t, xt_t = x[0, 0], out[0, 0], y[0, 0]

            # convert to NumPy for metrics
            xi = xi_t.squeeze().numpy()
            xr = xr_t.squeeze().numpy()
            xt = xt_t.squeeze().numpy()

            # compute metrics
            mssim = compute_ssim(xt[None], xr[None], maxval[0].item())
            mpsnr = compute_psnr(xt[None], xr[None], maxval[0].item())
            mnmse = compute_nmse(xt, xr)

            samples.append({'idx': idx+1, 'ssim': mssim, 'psnr': mpsnr, 'nmse': mnmse})

            # back to torch for make_grid
            grid = make_grid([xi_t, xr_t, xt_t, (xt_t - xr_t).abs()],
                            nrow=4, normalize=True, value_range=(0, maxval[0].item()))

            plt.figure(figsize=(12, 3))
            plt.axis('off')
            plt.imshow(grid.permute(1, 2, 0), cmap='gray')
            fname = f"sample_{idx+1:02d}.png"
            plt.savefig(os.path.join(args.output_dir, 'samples', fname),
                        bbox_inches='tight', dpi=150)
            plt.close()

        # save the per-sample metrics
        pd.DataFrame(samples).to_csv(
            os.path.join(args.output_dir, 'samples', 'metrics.csv'),
            index=False
        )
        print("📸 Sample images + metrics saved.")

    else:  # test-only path
        assert args.ckpt_path, "--ckpt_path is required in test mode"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LitReconNet.load_from_checkpoint(
            args.ckpt_path,
            hparams=args
        ).to(device).eval()
        results = trainer.test(model, datamodule=dm, verbose=True)
        print("➡️ Test results:", results)





if __name__ == '__main__':
    args = build_args()
    cli_main(args)
