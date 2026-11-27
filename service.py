import os
import re
import time
import torch
from transformers import AutoTokenizer
from config import MODEL_CONFIG
from model import AdvancedSummarizer


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def clean_text(text: str) -> str:
    text = re.sub(r"<.*?>|\r\n|\r|\n", " ", text)
    return re.sub(r"\s+", " ", text).strip()


class SummarizerService:
    def __init__(self):
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIG["tokenizer_name"])

        self.model = AdvancedSummarizer(
            vocab_size=self.tokenizer.vocab_size,
            d_model=MODEL_CONFIG["d_model"],
            n_heads=MODEL_CONFIG["n_heads"],
            num_enc=MODEL_CONFIG["num_enc_layers"],
            num_dec=MODEL_CONFIG["num_dec_layers"],
            dropout=MODEL_CONFIG["dropout"],
            pad_id=self.tokenizer.pad_token_id,
        ).to(self.device)

        weights_path = MODEL_CONFIG["weights_path"]
        if not os.path.exists(weights_path):
            raise FileNotFoundError(
                f"Model weights not found at '{weights_path}'. Train and save the model first."
            )

        state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"Custom RMSNorm Transformer loaded on {self.device}")

    def summarize(self, text: str, max_len: int = 80, beam_size: int = 4) -> dict:
        start_time = time.perf_counter()
        cleaned = clean_text(text)

        src = self.tokenizer(
            cleaned,
            max_length=MODEL_CONFIG["max_src_len"],
            truncation=True,
            return_tensors="pt",
        )["input_ids"].to(self.device)

        output_ids = self.model.generate(
            src,
            bos_id=self.tokenizer.bos_token_id,
            eos_id=self.tokenizer.eos_token_id,
            max_len=max_len,
            beam_size=beam_size,
        )

        summary = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        elapsed_ms = round((time.perf_counter() - start_time) * 1000)

        return {
            "summary": summary,
            "latency_ms": elapsed_ms,
            "device": str(self.device),
        }