"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from argparse import ArgumentParser
import pytorch_lightning as pl

import torch
from torch.nn import functional as F
from pathlib import Path

import pandas as pd 
from fastmri.models import Unet
from .mri_module import MriModule
from lightning.pytorch import LightningModule
import torchmetrics.functional as TMF


class MriModule(LightningModule):
    """Base module for MRI reconstruction models"""
    pass
    

class UnetModule(MriModule):
    """
    Unet training module.

    This can be used to train baseline U-Nets from the paper:

    J. Zbontar et al. fastMRI: An Open Dataset and Benchmarks for Accelerated
    MRI. arXiv:1811.08839. 2018.
    """

    def __init__(
        self,
        in_chans=1,
        out_chans=1,
        chans=32,
        num_pool_layers=4,
        drop_prob=0.0,
        lr=0.001,
        lr_step_size=40,
        lr_gamma=0.1,
        weight_decay=0.0,
        bbox_csv="training_bounding_boxes.csv",
        roi_weight=2.0,
        **kwargs,
    ):
        """
        Args:
            in_chans (int, optional): Number of channels in the input to the
                U-Net model. Defaults to 1.
            out_chans (int, optional): Number of channels in the output to the
                U-Net model. Defaults to 1.
            chans (int, optional): Number of output channels of the first
                convolution layer. Defaults to 32.
            num_pool_layers (int, optional): Number of down-sampling and
                up-sampling layers. Defaults to 4.
            drop_prob (float, optional): Dropout probability. Defaults to 0.0.
            lr (float, optional): Learning rate. Defaults to 0.001.
            lr_step_size (int, optional): Learning rate step size. Defaults to
                40.
            lr_gamma (float, optional): Learning rate gamma decay. Defaults to
                0.1.
            weight_decay (float, optional): Parameter for penalizing weights
                norm. Defaults to 0.0.
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay


        self.unet = Unet(
            in_chans=self.in_chans,
            out_chans=self.out_chans,
            chans=self.chans,
            num_pool_layers=self.num_pool_layers,
            drop_prob=self.drop_prob,
        )

        self.roi_weight = roi_weight

        # Load bounding boxes
        self.df = pd.read_csv(bbox_csv)
        df = self.df
        df["area"] = (df["x1"] - df["x0"]) * (df["y1"] - df["y0"])
        median_area = df["area"].median()
        self.filtered_boxes = {
            (row["filename"], row["slice"]): (row["x0"], row["y0"], row["x1"], row["y1"])
            for _, row in df.iterrows() if row["area"] >= median_area
        }

    def forward(self, image):
        return self.unet(image.unsqueeze(1)).squeeze(1)

    def training_step(self, batch, batch_idx):
        output = self(batch.image)
        df = self.df
        target = batch.target 
        median_area = df["area"].median()
        fname = Path(batch.fname[0]).stem
        slice_num = int(batch.slice_num[0])

        key = (fname, slice_num)
        mask = torch.ones_like(target)

        if key in self.filtered_boxes:
            x0, y0, x1, y1 = map(int, self.filtered_boxes[key])
            mask[..., y0:y1, x0:x1] = self.roi_weight

        loss = torch.abs(output - target) * mask
        loss = loss.mean()

        #self.log("train_loss", loss)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss
    """
    def validation_step(self, batch, batch_idx):
        output = self(batch.image)
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)

        return {
            "batch_idx": batch_idx,
            "fname": batch.fname,
            "slice_num": batch.slice_num,
            "max_value": batch.max_value,
            "output": output * std + mean,
            "target": batch.target * std + mean,
            "val_loss": F.l1_loss(output, batch.target),
        }
    """
    def validation_step(self, batch, batch_idx):
        output = self(batch.image)
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)

        recons = output * std + mean
        target = batch.target * std + mean

        val_loss = F.l1_loss(output, batch.target)
        psnr = TMF.peak_signal_noise_ratio(recons, target, data_range=1.0)
        ssim = TMF.structural_similarity_index_measure(recons, target, data_range=1.0)

        self.log("val_loss", val_loss)
        self.log("val_psnr", psnr)
        self.log("val_ssim", ssim)

        return val_loss

    def test_step(self, batch, batch_idx):
        output = self.forward(batch.image)
        mean = batch.mean.unsqueeze(1).unsqueeze(2)
        std = batch.std.unsqueeze(1).unsqueeze(2)
        target = batch.target * std + mean

        return {
            "fname": batch.fname,
            "slice": batch.slice_num,
            "output": (output * std + mean).cpu().numpy(),
            "target": target.cpu(),
        }
    
    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(
        self.parameters(),
        lr=self.lr,
        weight_decay=self.weight_decay,
    )
        scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        self.lr_step_size,
        self.lr_gamma,
    )
    
        # Simpler format: Just returning the optimizer and scheduler directly
        return [optimizer], [scheduler]
    
    """ 
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

        return {
           "optimizer": optimizer,
           "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "val_loss",  # must log this in validation_step
            "interval": "epoch",
            "frequency": 1,
            "reduce_on_plateau": True,
           }
        }



    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(
           self.parameters(),
           lr=self.lr,
           weight_decay=self.weight_decay,
         )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size = self.lr_step_size,
            gamma = self.lr_gamma,
            )

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    """

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
   
        # Define parameters that only apply to this model
  
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = MriModule.add_model_specific_args(parser)

        # network params
        parser.add_argument(
            "--in_chans", default=1, type=int, help="Number of U-Net input channels"
        )
        parser.add_argument(
            "--out_chans", default=1, type=int, help="Number of U-Net output chanenls"
        )
        parser.add_argument(
            "--chans", default=1, type=int, help="Number of top-level U-Net filters."
        )
        parser.add_argument(
            "--num_pool_layers",
            default=4,
            type=int,
            help="Number of U-Net pooling layers.",
        )
        parser.add_argument(
            "--drop_prob", default=0.0, type=float, help="U-Net dropout probability"
        )

        # training params (opt)
        parser.add_argument(
            "--lr", default=0.001, type=float, help="RMSProp learning rate"
        )
        parser.add_argument(
            "--lr_step_size",
            default=40,
            type=int,
            help="Epoch at which to decrease step size",
        )
        parser.add_argument(
            "--lr_gamma", default=0.1, type=float, help="Amount to decrease step size"
        )
        parser.add_argument(
            "--weight_decay",
            default=0.0,
            type=float,
            help="Strength of weight decay regularization",
        )

        return parser

        
