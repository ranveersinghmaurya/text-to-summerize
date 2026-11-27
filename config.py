MODEL_CONFIG = {
    "tokenizer_name": "facebook/bart-base",
    "weights_path": "custom_summarizer.pt",
    "max_src_len": 256,
    "d_model": 384,
    "n_heads": 6,
    "num_enc_layers": 4,
    "num_dec_layers": 4,
    "dropout": 0.1,
}