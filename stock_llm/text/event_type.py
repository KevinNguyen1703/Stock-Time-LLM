"""Zero-shot event typing of each (headline, tagged company) pair with the same local LLM.

Events are defined from text only (known before the open), never from the price outcome.
Output: data/text/event_<name>.parquet (url, ticker, event).
"""
import argparse

import pandas as pd
from vllm import LLM, SamplingParams

from stock_llm.text.corpus import ALIASES
from stock_llm.text.llm_score import MODELS

EVENTS = {
    "earnings": "kết quả kinh doanh, lợi nhuận, doanh thu, báo cáo tài chính",
    "guidance": "kế hoạch kinh doanh, mục tiêu năm, đại hội cổ đông, dự báo",
    "capital": "cổ tức, phát hành cổ phiếu, tăng vốn, chia thưởng, mua lại cổ phiếu",
    "ownership": "cổ đông lớn, người nội bộ, khối ngoại, quỹ mua/bán, thoái vốn",
    "deal": "hợp đồng, dự án, đầu tư, mở rộng, hợp tác, M&A",
    "legal": "xử phạt, thanh tra, khởi tố, vi phạm, tranh chấp",
    "management": "bổ nhiệm, miễn nhiệm, thay đổi lãnh đạo",
    "macro": "chính sách, lãi suất, tín dụng, vĩ mô, ngành, thị trường chung",
    "routine": "thủ tục hành chính, thông báo định kỳ, không có thông tin mới quan trọng",
}
SYSTEM = ("Phân loại tin tức theo loại sự kiện đối với công ty được hỏi. Trả lời đúng một nhãn trong danh sách:\n"
          + "\n".join(f"- {k}: {v}" for k, v in EVENTS.items()))


def main(name):
    c = pd.read_parquet("data/text/corpus.parquet")
    pairs = c[c.tickers != ""].assign(ticker=lambda d: d.tickers.str.split("|")).explode("ticker")
    llm = LLM(MODELS[name], dtype="bfloat16", max_model_len=1024, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    texts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM},
         {"role": "user", "content": f"Công ty: {' / '.join(ALIASES[r.ticker])}\nTiêu đề: {r.title}\n"
                                     + (f"Tóm tắt: {r.lead}\n" if r.lead else "") + "Nhãn:"}],
        tokenize=False, add_generation_prompt=True) for r in pairs.itertuples()]
    outs = llm.generate(texts, SamplingParams(temperature=0, max_tokens=6))
    raw = [o.outputs[0].text.strip().lower() for o in outs]
    pairs["event"] = [next((k for k in EVENTS if k in t), "routine") for t in raw]
    pairs["parsed"] = [any(k in t for k in EVENTS) for t in raw]
    pairs[["url", "ticker", "event", "parsed"]].to_parquet(f"data/text/event_{name}.parquet", index=False)
    print(f"{len(pairs)} pairs, parsed {pairs.parsed.mean()*100:.1f}%")
    print(pairs.event.value_counts().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="qwen25_7b", choices=list(MODELS))
    main(ap.parse_args().name)
