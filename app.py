# app.py — Gradio web app for Medical Report Summarizer

import torch
import gradio as gr
from transformers import AutoTokenizer
from model.transformer import Transformer
from evaluate import greedy_decode
from config import config


# ── Load model + tokenizer ────────────────────────────────
def load_model(model_path=config.BEST_MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
    device    = torch.device(config.DEVICE)
    model     = Transformer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model, tokenizer, device


# ── Inference function ────────────────────────────────────
def summarize(report_text, model, tokenizer, device):
    if not report_text.strip():
        return "Please enter a medical report."

    src = tokenizer(
        report_text,
        max_length=config.MAX_INPUT_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    src_ids  = src["input_ids"].to(device)
    src_mask = src["attention_mask"].to(device).unsqueeze(1).unsqueeze(2)

    token_ids = greedy_decode(model, src_ids, src_mask, device=device)
    summary   = tokenizer.decode(token_ids, skip_special_tokens=True)

    return summary if summary.strip() else "Model generated an empty summary. Try after training."


# ── Gradio UI ─────────────────────────────────────────────
def build_app():
    try:
        model, tokenizer, device = load_model()
        print("Loaded trained model.")
    except FileNotFoundError:
        print("No trained model found. Running in demo mode (untrained).")
        tokenizer = AutoTokenizer.from_pretrained(config.TOKENIZER_NAME)
        device    = torch.device(config.DEVICE)
        model     = Transformer().to(device)
        model.eval()

    def predict(text):
        return summarize(text, model, tokenizer, device)

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Textbox(
            lines=10,
            placeholder="Paste a medical report here...",
            label="Medical Report"
        ),
        outputs=gr.Textbox(
            lines=4,
            label="Plain-English Summary"
        ),
        title="🏥 Medical Report Summarizer",
        description="Paste a clinical medical report and get a plain-English patient summary. Built with a Transformer from scratch in PyTorch.",
        examples=[
            ["The patient presents with acute myocardial infarction with ST elevation in leads II, III, and aVF consistent with inferior wall MI. Troponin levels are elevated at 2.4 ng/mL."],
            ["MRI findings reveal a 2.3cm lesion in the right temporal lobe with surrounding edema. No midline shift is observed. Contrast enhancement suggests possible high-grade glioma."],
        ],
    )

    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(share=False, theme=gr.themes.Soft())