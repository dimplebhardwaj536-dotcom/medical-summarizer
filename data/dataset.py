# data/dataset.py — Load, tokenize and prepare PubMed dataset

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
from config import config


class MedicalDataset(Dataset):
    def __init__(self, split="train", max_samples=None):
        print(f"Loading dataset: {split}...")
        dataset = load_dataset(config.DATASET_NAME, split=split)

        # use subset for faster training
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        self.tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
        self.data = dataset
        print(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # tokenize input (article) 
        src = self.tokenizer(
            item["article"],
            max_length=config.MAX_INPUT_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # tokenize target (summary)
        tgt = self.tokenizer(
            item["abstract"],
            max_length=config.MAX_TARGET_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        src_ids = src["input_ids"].squeeze(0)        # (MAX_INPUT_LEN,)
        src_mask = src["attention_mask"].squeeze(0)  # (MAX_INPUT_LEN,)
        tgt_ids = tgt["input_ids"].squeeze(0)        # (MAX_TARGET_LEN,)

        # decoder input  = tgt[:-1] (shift right)
        # decoder target = tgt[1:]  (what we predict)
        return {
            "src_ids":   src_ids,
            "src_mask":  src_mask,
            "tgt_input": tgt_ids[:-1],   # (MAX_TARGET_LEN - 1,)
            "tgt_label": tgt_ids[1:],    # (MAX_TARGET_LEN - 1,)
        }


def get_dataloaders():
    train_dataset = MedicalDataset(split="train", max_samples=config.TRAIN_SAMPLES)
    val_dataset   = MedicalDataset(split="validation", max_samples=config.VAL_SAMPLES)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,     # 0 = no multiprocessing (safe on Windows)
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders()
    batch = next(iter(train_loader))
    print("src_ids shape  :", batch["src_ids"].shape)
    print("tgt_input shape:", batch["tgt_input"].shape)
    print("tgt_label shape:", batch["tgt_label"].shape)
    print("Dataset pipeline works!")