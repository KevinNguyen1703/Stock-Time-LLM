"""Zero-shot LLM scoring of each (headline, tagged company) pair: expected impact on the stock.

Score in {-2..2}; -2 = clearly negative for the share price, 2 = clearly positive, 0 = neutral/irrelevant.
Leakage note: the LLM's pretraining may include hindsight about events before its cutoff
(Qwen2.5: ~2024). Only test periods after the cutoff are free of that risk.
Output: data/text/llm_<name>.parquet (url, ticker, score).
"""
import argparse
import re

import pandas as pd
from vllm import LLM, SamplingParams

from stock_llm.text.corpus import ALIASES

MODELS = {"qwen25_7b": "Qwen/Qwen2.5-7B-Instruct"}
SYSTEM = ("Bạn là chuyên gia phân tích chứng khoán Việt Nam. Đánh giá tác động của tin tức tới giá cổ phiếu "
          "của công ty được hỏi trong vài phiên giao dịch tới. Chỉ trả lời đúng một số nguyên trong "
          "{-2, -1, 0, 1, 2}: -2 rất tiêu cực, -1 tiêu cực, 0 trung lập hoặc không liên quan, 1 tích cực, 2 rất tích cực.")


def prompt(tok, title, lead, ticker):
    names = " / ".join(ALIASES[ticker])
    user = f"Công ty: {names}\nTiêu đề: {title}\n" + (f"Tóm tắt: {lead}\n" if lead else "") + "Điểm:"
    return tok.apply_chat_template([{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


def main(name):
    c = pd.read_parquet("data/text/corpus.parquet")
    pairs = c[c.tickers != ""].assign(ticker=lambda d: d.tickers.str.split("|")).explode("ticker")
    llm = LLM(MODELS[name], dtype="bfloat16", max_model_len=1024, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    texts = [prompt(tok, r.title, r.lead, r.ticker) for r in pairs.itertuples()]
    outs = llm.generate(texts, SamplingParams(temperature=0, max_tokens=4))
    raw = [o.outputs[0].text for o in outs]
    pairs["score"] = [int(m.group()) if (m := re.search(r"-?[0-2]", t)) else 0 for t in raw]
    pairs["parsed"] = [bool(re.search(r"-?[0-2]", t)) for t in raw]
    pairs[["url", "ticker", "score", "parsed"]].to_parquet(f"data/text/llm_{name}.parquet", index=False)
    print(f"{len(pairs)} pairs, parsed {pairs.parsed.mean()*100:.1f}%")
    print(pairs.score.value_counts().sort_index().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="qwen25_7b", choices=list(MODELS))
    main(ap.parse_args().name)
