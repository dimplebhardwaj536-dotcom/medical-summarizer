# train.py — Training loop with PyTorch Lightning

import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os

from model.transformer import Transformer
from data.dataset import get_dataloaders
from config import config


class MedicalSummarizer(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Transformer()
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=config.PAD_TOKEN_ID,
            label_smoothing=config.LABEL_SMOOTHING
        )

    def forward(self, src_ids, tgt_ids, src_mask=None):
        return self.model(src_ids, tgt_ids, src_mask)

    def _shared_step(self, batch):
        src_ids   = batch["src_ids"]
        src_mask  = batch["src_mask"]
        tgt_input = batch["tgt_input"]
        tgt_label = batch["tgt_label"]

        # fix mask shape: (batch, 512) → (batch, 1, 1, 512)
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)

        logits = self(src_ids, tgt_input, src_mask)

        loss = self.criterion(
            logits.view(-1, config.VOCAB_SIZE),
            tgt_label.view(-1)
        )
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=config.LEARNING_RATE,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.LEARNING_RATE,
            steps_per_epoch=625,
            epochs=config.EPOCHS,
            pct_start=0.1
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"}
        }


def train():
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    train_loader, val_loader = get_dataloaders()

    model = MedicalSummarizer()

    checkpoint_cb = ModelCheckpoint(
        dirpath=config.CHECKPOINT_DIR,
        filename="best_model",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=True
    )

    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=3,
        mode="min",
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=config.EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=config.GRAD_CLIP,
        log_every_n_steps=config.LOG_EVERY_N_STEPS,
        callbacks=[checkpoint_cb, early_stop_cb],
        precision="16-mixed" if torch.cuda.is_available() else 32,
    )

    trainer.fit(model, train_loader, val_loader)
    print(f"Best model saved at: {checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    train()