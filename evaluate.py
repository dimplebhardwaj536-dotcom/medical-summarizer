# evaluate.py — ROUGE + BERTScore evaluation

import torch
from transformers import AutoTokenizer
from rouge_score import rouge_scorer
from bert_score import score as bert_score
from model.transformer import Transformer
from config import config


def greedy_decode(model, src_ids, src_mask, max_len=config.MAX_TARGET_LEN, device="cpu"):
    """Generate summary token by token using greedy decoding."""
    model.eval()
    with torch.no_grad():
        # encode source once
        src_emb     = model.src_pos(model.src_embedding(src_ids))
        encoder_out = model.encoder(src_emb, src_mask)

        # start with BOS token
        tgt = torch.tensor([[config.BOS_TOKEN_ID]], device=device)

        for _ in range(max_len - 1):
            tgt_emb    = model.tgt_pos(model.tgt_embedding(tgt))
            tgt_mask   = model.make_tgt_mask(tgt)
            decoder_out = model.decoder(tgt_emb, encoder_out, src_mask, tgt_mask)
            logits      = model.output_layer(decoder_out)

            # pick highest probability token
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt        = torch.cat([tgt, next_token], dim=1)

            # stop if EOS token generated
            if next_token.item() == config.EOS_TOKEN_ID:
                break

    return tgt.squeeze(0).tolist()  # list of token ids


def evaluate(model_path, test_samples):
    """
    Evaluate model on test samples.
    test_samples: list of {"article": ..., "abstract": ...}
    """
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    device    = torch.device(config.DEVICE)

    # load model
    model = Transformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictions = []
    references  = []

    for sample in test_samples:
        # tokenize input
        src = tokenizer(
            sample["article"],
            max_length=config.MAX_INPUT_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        src_ids  = src["input_ids"].to(device)
        src_mask = src["attention_mask"].to(device)
        src_mask = src_mask.unsqueeze(1).unsqueeze(2)

        # generate summary
        token_ids = greedy_decode(model, src_ids, src_mask, device=device)
        summary   = tokenizer.decode(token_ids, skip_special_tokens=True)

        predictions.append(summary)
        references.append(sample["abstract"])

    # ROUGE scores
    scorer  = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1, r2, rL = 0, 0, 0

    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        r1 += scores["rouge1"].fmeasure
        r2 += scores["rouge2"].fmeasure
        rL += scores["rougeL"].fmeasure

    n = len(predictions)
    print(f"ROUGE-1: {r1/n:.4f}")
    print(f"ROUGE-2: {r2/n:.4f}")
    print(f"ROUGE-L: {rL/n:.4f}")

    # BERTScore
    P, R, F1 = bert_score(predictions, references, lang="en", verbose=False)
    print(f"BERTScore F1: {F1.mean():.4f}")

    return predictions


if __name__ == "__main__":
    # quick test with dummy data (no model needed)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    pred   = "patient has chest pain and breathing difficulty"
    ref    = "patient experiences chest pain with breathing problems"
    scores = scorer.score(ref, pred)
    print("ROUGE-1:", round(scores["rouge1"].fmeasure, 4))
    print("ROUGE-2:", round(scores["rouge2"].fmeasure, 4))
    print("ROUGE-L:", round(scores["rougeL"].fmeasure, 4))
    print("Evaluate pipeline works!")