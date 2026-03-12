# bart_summarizer.py — Fine-tune Facebook BART on PubMed medical data

import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from config import config
import os


class PubMedDataset(Dataset):
    def __init__(self, split="train", max_samples=5000):
        print(f"Loading dataset: {split}...")
        dataset = load_dataset(config.DATASET_NAME, split=split)
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        self.data      = dataset
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
        print(f"Loaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        src  = self.tokenizer(
            item["article"],
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        tgt  = self.tokenizer(
            item["abstract"],
            max_length=128,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        labels = tgt["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids":      src["input_ids"].squeeze(0),
            "attention_mask": src["attention_mask"].squeeze(0),
            "labels":         labels
        }


def train_bart():
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model     = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)

    train_dataset = PubMedDataset(split="train",      max_samples=5000)
    val_dataset   = PubMedDataset(split="validation", max_samples=500)

    train_loader  = DataLoader(train_dataset, batch_size=4, shuffle=True,  num_workers=0)
    val_loader    = DataLoader(val_dataset,   batch_size=4, shuffle=False, num_workers=0)

    optimizer     = torch.optim.AdamW(model.parameters(), lr=3e-5)
    best_val_loss = float("inf")
    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(3):
        model.train()
        train_loss = 0
        for i, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss           = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.item()
            if i % 50 == 0:
                print(f"Epoch {epoch} | Step {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                outputs   = model(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    labels=batch["labels"].to(device)
                )
                val_loss += outputs.loss.item()

        avg_val = val_loss / len(val_loader)
        avg_trn = train_loss / len(train_loader)
        print(f"Epoch {epoch} DONE | Train: {avg_trn:.4f} | Val: {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), "checkpoints/bart_best.pt")
            print(f"Best model saved! Val loss: {avg_val:.4f}")

    print("Training complete!")
    return tokenizer, model


def summarize_bart(text, model_path="checkpoints/bart_best.pt", device="cpu"):
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    model     = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    inputs = tokenizer(
        text,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=128,
            min_length=30,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    train_bart()