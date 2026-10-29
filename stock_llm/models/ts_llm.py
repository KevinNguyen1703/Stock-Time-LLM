"""Event-driven text + time-series LLM.

Prompt = event headlines as Vietnamese text + <ts> soft tokens. The soft tokens come from a patch
encoder whose outputs are reprogrammed into the LLM embedding space by cross-attention over a small
set of text prototypes (linear combinations of the LLM's word embeddings), as in Time-LLM.
The LLM is adapted with LoRA; a linear head on the last token predicts P(up over the horizon).
"""
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

EVENT_VI = {"earnings": "kết quả kinh doanh", "guidance": "kế hoạch/ĐHCĐ", "capital": "vốn/cổ tức",
            "ownership": "sở hữu/giao dịch cổ đông", "deal": "hợp đồng/dự án", "legal": "pháp lý",
            "management": "nhân sự lãnh đạo", "macro": "vĩ mô/ngành"}
TS_TOKEN = "<|ts|>"


class Reprogramming(nn.Module):
    """Patch features -> LLM embedding space via attention over learned text prototypes."""
    def __init__(self, n_feat, d_llm, word_emb, patch=8, stride=4, d=256, n_proto=512, heads=8, dropout=0.1):
        super().__init__()
        self.patch, self.stride = patch, stride
        self.embed = nn.Linear(patch * n_feat, d)
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, 1)
        self.register_buffer("word_emb", word_emb, persistent=False)          # [V, d_llm], frozen
        self.to_proto = nn.Linear(word_emb.shape[0], n_proto, bias=False)     # prototypes = W @ word_emb
        self.q = nn.Linear(d, d_llm)
        self.attn = nn.MultiheadAttention(d_llm, heads, dropout=dropout, batch_first=True)
        self.out = nn.Linear(d_llm, d_llm)

    def forward(self, x):                                   # x: [B, T, F]
        p = x.unfold(1, self.patch, self.stride)            # [B, P, F, patch]
        h = self.enc(self.embed(p.reshape(p.shape[0], p.shape[1], -1)))
        proto = self.to_proto(self.word_emb.T.float()).T    # [n_proto, d_llm]
        proto = proto.unsqueeze(0).expand(x.shape[0], -1, -1)
        a, _ = self.attn(self.q(h), proto, proto)
        return self.out(a)                                  # [B, P, d_llm]


class TSLLM(nn.Module):
    def __init__(self, base, n_feat, seq_len, lora_r=16):
        super().__init__()
        self.tok = AutoTokenizer.from_pretrained(base)
        self.tok.add_special_tokens({"additional_special_tokens": [TS_TOKEN]})
        self.ts_id = self.tok.convert_tokens_to_ids(TS_TOKEN)
        llm = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)
        llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        llm.enable_input_require_grads()
        emb = llm.get_input_embeddings().weight.detach()
        self.llm = get_peft_model(llm, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                                                  target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        d_llm = emb.shape[1]
        self.n_patch = (seq_len - 8) // 4 + 1
        # prototypes are built from the 20k most frequent (lowest-id) tokens to keep the map small
        self.reprog = Reprogramming(n_feat, d_llm, emb[:20000].clone())
        self.head = nn.Sequential(nn.Dropout(0.1), nn.Linear(d_llm, 1))

    def forward(self, input_ids, attention_mask, ts=None):
        emb = self.llm.get_input_embeddings()(torch.where(input_ids == self.ts_id, 0, input_ids))
        if ts is not None:
            soft = self.reprog(ts).to(emb.dtype)            # [B, P, d]
            m = input_ids == self.ts_id                     # exactly P slots per row
            emb = emb.masked_scatter(m.unsqueeze(-1), soft.reshape(-1, soft.shape[-1])[: int(m.sum())])
        out = self.llm(inputs_embeds=emb, attention_mask=attention_mask, output_hidden_states=True)
        h = out.hidden_states[-1]
        last = attention_mask.sum(1) - 1                    # right padding
        return self.head(h[torch.arange(h.shape[0]), last].float()).squeeze(-1)
