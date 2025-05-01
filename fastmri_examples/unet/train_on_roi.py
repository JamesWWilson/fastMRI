import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import pytorch_lightning as pl
from fastmri.models import Unet

# Dataset for grayscale PNG knee ROI images
class ROIDataset(Dataset):
    def __init__(self, img_dir):
        self.img_dir = img_dir
        self.files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        image = Image.open(img_path).convert("L")
        image = self.transform(image)
        return image, image  # input and target are same for autoencoder

# LightningModule for U-Net fine-tuning
class LitUnet(pl.LightningModule):
    def __init__(self, pretrained_path=None, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = Unet(in_chans=1, out_chans=1, chans=256, num_pool_layers=4)
        self.loss_fn = nn.L1Loss()
        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location="cpu")
            self.model.load_state_dict(state_dict)
            print("Loaded pretrained model from:", pretrained_path)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

# DataModule for better separation (optional)
class ROIDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=16, num_workers=12):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.dataset = ROIDataset(self.data_dir)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

# Main training script
def main():
    pl.seed_everything(42)

    data_dir = "G:/Deep Learning/Group Project/fastMRI/data/Filtered_Training/random_1000"
    pretrained_model_path = "knee_sc_leaderboard_state_dict.pt"
    checkpoint_path = "unet_finetuned_roi.ckpt"

    datamodule = ROIDataModule(data_dir=data_dir, batch_size=16, num_workers=12)  # Use 0 for Windows safety

    model = LitUnet(pretrained_path=pretrained_model_path)

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices=1,
        log_every_n_steps=10,
        default_root_dir="checkpoints",
        enable_checkpointing=True,
    )

    trainer.fit(model, datamodule=datamodule)

    # Save final weights
    torch.save(model.model.state_dict(), "unet_finetuned_roi_final.pth")

if __name__ == "__main__":
    from torch.multiprocessing import freeze_support
    freeze_support()
    main()


