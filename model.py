import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        hidden_dim = int(d_model * 8 / 3)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query, key, value, attn_mask=None):
        bsz, q_len, dim = query.shape
        k_len = key.size(1)

        q = self.q_proj(query).view(bsz, q_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(bsz, k_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(bsz, k_len, self.n_heads, self.head_dim).transpose(1, 2)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(bsz, q_len, dim)
        return self.out_proj(out)


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, src_mask):
        x = x + self.dropout(self.attn(self.norm1(x), self.norm1(x), self.norm1(x), src_mask))
        x = x + self.ffn(self.norm2(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.norm3 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask, src_mask):
        x = x + self.dropout(self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), tgt_mask))
        x = x + self.dropout(self.cross_attn(self.norm2(x), enc_out, enc_out, src_mask))
        x = x + self.ffn(self.norm3(x))
        return x


class AdvancedSummarizer(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, num_enc, num_dec, dropout, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.scale = math.sqrt(d_model)

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.drop = nn.Dropout(dropout)

        self.encoder = nn.ModuleList([EncoderBlock(d_model, n_heads, dropout) for _ in range(num_enc)])
        self.enc_norm = RMSNorm(d_model)

        self.decoder = nn.ModuleList([DecoderBlock(d_model, n_heads, dropout) for _ in range(num_dec)])
        self.dec_norm = RMSNorm(d_model)

        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

    def encode(self, src, src_mask):
        x = self.drop(self.pos_enc(self.embedding(src) * self.scale))
        for layer in self.encoder:
            x = layer(x, src_mask)
        return self.enc_norm(x)

    def decode(self, tgt, enc_out, tgt_mask, src_mask):
        x = self.drop(self.pos_enc(self.embedding(tgt) * self.scale))
        for layer in self.decoder:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return self.lm_head(self.dec_norm(x))

    @torch.no_grad()
    def generate(self, src, bos_id, eos_id, max_len=80, beam_size=4, length_penalty=1.2):
        self.eval()
        src_mask = (src != self.pad_id).unsqueeze(1).unsqueeze(2)
        enc_out = self.encode(src, src_mask)

        beams = [(torch.tensor([[bos_id]], device=src.device), 0.0)]
        completed = []

        for _ in range(max_len):
            candidates = []
            for seq, score in beams:
                if seq[0, -1].item() == eos_id:
                    completed.append((seq, score))
                    continue

                seq_len = seq.size(1)
                causal = torch.tril(torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device=src.device))
                logits = self.decode(seq, enc_out, causal, src_mask)[:, -1, :]
                log_probs = F.log_softmax(logits, dim=-1)

                top_probs, top_ids = torch.topk(log_probs, beam_size, dim=-1)
                for i in range(beam_size):
                    next_seq = torch.cat([seq, top_ids[:, i : i + 1]], dim=1)
                    candidates.append((next_seq, score + top_probs[0, i].item()))

            if not candidates:
                break

            candidates.sort(key=lambda x: x[1] / (x[0].size(1) ** length_penalty), reverse=True)
            beams = candidates[:beam_size]

        completed.extend(beams)
        completed.sort(key=lambda x: x[1] / (x[0].size(1) ** length_penalty), reverse=True)
        return completed[0][0][0]