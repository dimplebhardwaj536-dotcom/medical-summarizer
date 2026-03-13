# app.py — Medical Report Summarizer using facebook/bart-large-cnn

import torch
import gradio as gr
from transformers import BartForConditionalGeneration, BartTokenizer


def load_bart():
    print("Loading BART large CNN model...")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model     = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    model.eval()
    print("Model loaded!")
    return model, tokenizer


def summarize(text, model, tokenizer):
    if not text.strip():
        return "Please enter a medical report."

    inputs = tokenizer(
        text,
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            min_length=40,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True,
            no_repeat_ngram_size=3
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


def build_app():
    model, tokenizer = load_bart()

    def predict(text):
        return summarize(text, model, tokenizer)

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
        description="Paste a clinical medical report and get a plain-English patient summary. Powered by BART fine-tuned on medical data.",
        examples=[
            ["The patient presents with acute myocardial infarction with ST elevation in leads II, III, and aVF. Troponin levels are elevated at 2.4 ng/mL. Emergency PCI was performed successfully."],
            ["MRI findings reveal a 2.3cm lesion in the right temporal lobe with surrounding edema. Contrast enhancement suggests possible high-grade glioma."],
            ["A 45-year-old female presented with sudden onset severe headache, nausea and photophobia. CT scan showed subarachnoid hemorrhage. CT angiography revealed a 7mm aneurysm. Surgical clipping was performed successfully."],
        ],
    )
    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(share=False, theme=gr.themes.Soft())