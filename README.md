# 🏥 Medical Report Summarizer
### Transformer from Scratch → Fine-tuned BART → Production Deployment

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![Gradio](https://img.shields.io/badge/Gradio-4.0-green)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

> **Paste a clinical medical report. Get a plain-English patient summary.**
> Built a full Transformer architecture from scratch, trained on real PubMed medical data, and deployed as a live web application.

---

## 🎬 Demo

**Input:**
```
A 45-year-old female patient presented with sudden onset severe headache,
nausea, vomiting, and photophobia. CT scan showed subarachnoid hemorrhage.
CT angiography revealed a 7mm saccular aneurysm. Surgical clipping was
performed successfully within 24 hours.
```

**Output:**
```
A 45-year-old female patient presented with sudden onset severe headache,
nausea, vomiting and photophobia. CT scan showed subarachnoid hemorrhage
with blood in the basal cisterns. Patient was started on nimodipine to
prevent vasospasm.
```

---
### Phase 1 — Build Transformer from Scratch 🔨
Built every component of the Transformer architecture in pure PyTorch
with zero shortcuts:

- ✅ Multi-Head Attention (scaled dot-product, 8 heads)
- ✅ Positional Encoding (sinusoidal)
- ✅ Encoder stack (3 layers, residual connections, layer norm)
- ✅ Decoder stack (3 layers, masked self-attention + cross-attention)
- ✅ Full forward pass: token IDs → embeddings → encoder → decoder → logits
- ✅ 27.4 million parameters

### Phase 2 — First Training Run ❌ Then Fixed ✅

**What went wrong:**
```
RuntimeError: The size of tensor a (16) must match tensor b (512)
at non-singleton dimension 2
```
The `src_mask` shape from the DataLoader was `(batch, 512)` but
the attention mechanism expected `(batch, 1, 1, 512)`.

**Fix:**
```python
src_mask = src_mask.unsqueeze(1).unsqueeze(2)
```

**Lesson:** Shape errors are the #1 bug in transformer implementations.
Always trace tensor shapes through every operation.

---

### Phase 3 — Training on Google Colab T4 GPU ⚡

Trained from scratch on 10,000 PubMed medical papers:
```
Epoch 0:  val_loss = 7.726
Epoch 1:  val_loss = 7.545  ↓
Epoch 2:  val_loss = 7.212  ↓
Epoch 3:  val_loss = 7.048  ↓
Epoch 4:  val_loss = 6.919  ↓
Epoch 5:  val_loss = 6.802  ↓
Epoch 6:  val_loss = 6.747  ↓
Epoch 7:  val_loss = 6.707  ↓
Epoch 8:  val_loss = 6.694  ↓
Epoch 9:  val_loss = 6.693  ✅ Best
```

Loss decreased every single epoch — the model was genuinely learning.

---

### Phase 4 — Output Quality Issue ⚠️

**Problem:** Model generated repetitive outputs:
```
"the aim of this study was to evaluate the effect of the study
of the study of the study of the study of the study..."
```

**Root cause analysis:**
- Only 10 epochs — not enough for a 27M param model trained from random init
- Greedy decoding always picks the most probable next token — leads to loops
- 10K samples is small — real summarization models train on millions

**Fix 1 — Repetition penalty in greedy decode:**
```python
generated = tgt.squeeze().tolist()
if isinstance(generated, int):
    generated = [generated]
logits[:, -1, list(set(generated))] -= 1.5
```

**Fix 2 — Beam search (4 beams):**
Instead of picking the single best token at each step,
maintain 4 candidate sequences and pick the globally best one.

**Outcome:** Better but still not production quality.
**Decision:** Fine-tune a pretrained model instead.

---

### Phase 5 — BART Fine-tuning 🔬

**Why BART?**
BART (Bidirectional and Auto-Regressive Transformer) by Facebook AI
is specifically designed for sequence-to-sequence tasks like summarization.
It's pretrained on 160GB of text — a massive head start.

Fine-tuned `facebook/bart-base` on 5,000 PubMed samples:
```
Epoch 0 | Val Loss: 3.21
Epoch 1 | Val Loss: 2.68
Epoch 2 | Val Loss: 2.41  ✅ Best
```

**Problem:** Still copying input text instead of summarizing.
**Root cause:** 3 epochs on 5K samples — BART needs more data to learn
compression behavior, not just paraphrasing.

---

### Phase 6 

**Options considered:**
1. Train for 30 epochs on 50K samples (~4 hours on Colab)
2. Use `facebook/bart-large-cnn` — already fine-tuned on CNN/DailyMail summarization

**Decision: Option 2**

Real ML engineering isn't about training everything from scratch.
It's about knowing *when* to train and *when* to leverage existing work.
`bart-large-cnn` was trained specifically for summarization on 300K+ articles.
Using it is the right engineering call.

**Result:** Excellent coherent medical summaries immediately. ✅

---

### Phase 7 — Colab Session Reset Problem 

Free Google Colab resets sessions and deletes all files after disconnect.
Lost the trained model twice.

**Fix:** Mount Google Drive before training and save checkpoints there:
```python
Config.CHECKPOINT_DIR = "/content/drive/MyDrive/medical-summarizer-checkpoints"
```

**Lesson:** Always persist artifacts to permanent storage.
This is why production ML uses S3/GCS, not local disk.

---

## 🏗️ Architecture
```
Medical Report (text)
        │
        ▼
  BERT Tokenizer
(text → token IDs)
        │
        ▼
Token Embeddings + Positional Encoding
        │
        ▼
┌─────────────────┐
│    ENCODER      │  ← reads the full report
│  ┌───────────┐  │
│  │Self-Attn  │  │  each token attends to all others
│  │Add & Norm │  │
│  │Feed Fwd   │  │
│  │Add & Norm │  │
│  └───────────┘  │
│   × 3 layers    │
└────────┬────────┘
         │ encoder_out
         ▼
┌─────────────────┐
│    DECODER      │  ← generates summary token by token
│  ┌───────────┐  │
│  │Masked Attn│  │  can't see future tokens
│  │Cross-Attn │  │  attends to encoder output
│  │Feed Fwd   │  │
│  └───────────┘  │
│   × 3 layers    │
└────────┬────────┘
         │
         ▼
  Linear → Softmax
         │
         ▼
  Next token (repeat until [EOS])
         │
         ▼
Plain-English Summary
```

---

## 🛠️ Tech Stack

| Component | Technology | Why |
|---|---|---|
| Deep Learning | PyTorch 2.5.1 | Industry standard, full control |
| Training Loop | PyTorch Lightning | Clean, less boilerplate |
| Pretrained Models | HuggingFace Transformers | Access to SOTA models |
| Dataset | HuggingFace Datasets | Easy PubMed access |
| Tokenizer | BERT / BART Tokenizer | Subword tokenization |
| Evaluation | ROUGE + BERTScore | Standard NLP metrics |
| GPU Training | Google Colab T4 | Free cloud GPU |
| Deployment | Gradio | Fast ML app prototyping |
| Version Control | GitHub | Code management |

---

## 📁 Project Structure
```
medical-summarizer/
├── model/
│   ├── attention.py       # Multi-Head Attention from scratch
│   ├── encoder.py         # Encoder block + stack
│   ├── decoder.py         # Decoder block + stack
│   └── transformer.py     # Full Transformer (embeddings + pos enc)
├── data/
│   └── dataset.py         # PubMed DataLoader pipeline
├── config.py              # Single source of truth for hyperparams
├── train.py               # PyTorch Lightning training loop
├── evaluate.py            # ROUGE + BERTScore + beam search decode
├── bart_summarizer.py     # BART fine-tuning pipeline
├── app.py                 # Gradio web application
└── requirements.txt       # Dependencies
```

---

## 🚀 Run Locally
```bash
# clone
git clone https://github.com/dimplebhardwaj536-dotcom/medical-summarizer.git
cd medical-summarizer

# setup
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# run app (downloads bart-large-cnn ~1.6GB on first run)
python app.py
```

Open `http://127.0.0.1:7860`

---

## 📊 Results Across All Phases

| Phase | Model | Val Loss | Output Quality |
|---|---|---|---|
| 1 | Transformer from scratch | 6.69 | Repetitive ⚠️ |
| 2 | Scratch + repetition penalty | 6.69 | Slightly better ⚠️ |
| 3 | Scratch + beam search | 6.69 | Marginally better ⚠️ |
| 4 | BART fine-tuned 3 epochs | 2.41 | Copying input ⚠️ |
| 5 | bart-large-cnn | pretrained | Coherent summaries ✅ |

---

## 💡 Key Learnings

**Technical:**
- Tensor shape bugs are the #1 issue in transformer implementations
- From-scratch models need millions of samples + days of training to match pretrained
- Beam search > greedy decoding for sequence generation
- Always save checkpoints to persistent storage (learned the hard way)
- `unsqueeze(1).unsqueeze(2)` for mask broadcasting in attention

**Engineering:**
- Know when to build vs when to leverage existing work
- Free Colab resets — always use Google Drive for checkpoints
- Real ML engineering is debugging shapes, not just writing forward passes

---

## 🎯 What This Project Demonstrates
```
✅ Deep understanding of Transformer architecture
   (not just calling a library — built every component)

✅ Real training experience
   (data pipeline, loss curves, hyperparameter decisions)

✅ Debugging complex ML bugs
   (tensor shape errors, repetition loops, checkpoint issues)

✅ Practical engineering judgment
   (knowing when to use pretrained vs train from scratch)

✅ End-to-end ML project delivery
   (idea → code → train → evaluate → deploy)

✅ Cloud ML workflow
   (Colab GPU training, Drive persistence, GitHub CI)
```
---

*Built with PyTorch, trained on Google Colab, deployed with Gradio*