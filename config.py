# config.py — All hyperparameters and settings in one place

class Config:
    # ── Data ──────────────────────────────────────────
    DATASET_NAME     = "ccdv/pubmed-summarization"
    DATASET_SPLIT    = "train"
    MAX_INPUT_LEN    = 512      # max tokens for input report
    MAX_TARGET_LEN   = 128      # max tokens for output summary
    TRAIN_SAMPLES    = 10000    # use subset for faster training
    VAL_SAMPLES      = 1000

    # ── Tokenizer ─────────────────────────────────────
    TOKENIZER_NAME   = "bert-base-uncased"
    PAD_TOKEN_ID     = 0
    BOS_TOKEN_ID     = 101      # [CLS] token in BERT tokenizer
    EOS_TOKEN_ID     = 102      # [SEP] token in BERT tokenizer

    # ── Model Architecture ────────────────────────────
    VOCAB_SIZE       = 30522    # bert-base-uncased vocab size
    D_MODEL          = 256      # embedding dimension
    N_HEADS          = 8        # attention heads (D_MODEL % N_HEADS == 0)
    N_ENCODER_LAYERS = 3        # encoder stack depth
    N_DECODER_LAYERS = 3        # decoder stack depth
    D_FF             = 512      # feed-forward inner dimension
    DROPOUT          = 0.1      # dropout rate

    # ── Training ──────────────────────────────────────
    BATCH_SIZE       = 16
    EPOCHS           = 10
    LEARNING_RATE    = 1e-4
    WARMUP_STEPS     = 400
    GRAD_CLIP        = 1.0      # gradient clipping
    LABEL_SMOOTHING  = 0.1

    # ── Paths ─────────────────────────────────────────
    CHECKPOINT_DIR   = "checkpoints"
    BEST_MODEL_PATH  = "checkpoints/best_model.pt"
    LOG_DIR          = "logs"

    # ── Device ────────────────────────────────────────
    DEVICE           = "cpu"    # change to "cuda" on Colab

    # ── Logging ───────────────────────────────────────
    WANDB_PROJECT    = "medical-summarizer"
    LOG_EVERY_N_STEPS = 50


# single instance used everywhere
config = Config()