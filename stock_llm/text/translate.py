"""Translate the company-tagged headlines (title + lead) to English with the local LLM, then score
the English text with an English version of the sentiment prompt, so VI vs EN can be compared.
Outputs: data/text/translated_<name>.parquet (url, title_en, lead_en)
         data/text/llm_<name>_en.parquet     (url, ticker, score, parsed) - same format as the VI scores
"""
import argparse
import re

import pandas as pd
from vllm import LLM, SamplingParams

from stock_llm.text.corpus import ALIASES
from stock_llm.text.llm_score import MODELS

TR_SYSTEM = ("Translate the Vietnamese financial news text to English. Keep stock tickers, company names and "
             "numbers exactly. Output only the translation.")
SC_SYSTEM = ("You are a Vietnamese equity analyst. Rate the impact of the news on the share price of the given "
             "company over the next few trading sessions. Answer with exactly one integer in {-2, -1, 0, 1, 2}: "
             "-2 very negative, -1 negative, 0 neutral or irrelevant, 1 positive, 2 very positive.")


def chat(tok, system, user):
    return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True)


def main(name):
    c = pd.read_parquet("data/text/corpus.parquet")
    tagged = c[c.tickers != ""].copy()
    llm = LLM(MODELS[name], dtype="bfloat16", max_model_len=1536, gpu_memory_utilization=0.85)
    tok = llm.get_tokenizer()
    sp = SamplingParams(temperature=0, max_tokens=256)
    titles = llm.generate([chat(tok, TR_SYSTEM, t) for t in tagged["title"]], sp)
    has_lead = tagged["lead"] != ""
    leads = llm.generate([chat(tok, TR_SYSTEM, t) for t in tagged.loc[has_lead, "lead"]], sp)
    tagged["title_en"] = [o.outputs[0].text.strip() for o in titles]
    tagged["lead_en"] = ""
    tagged.loc[has_lead, "lead_en"] = [o.outputs[0].text.strip() for o in leads]
    tagged[["url", "title_en", "lead_en"]].to_parquet(f"data/text/translated_{name}.parquet", index=False)
    pairs = tagged.assign(ticker=lambda d: d.tickers.str.split("|")).explode("ticker")
    prompts = [chat(tok, SC_SYSTEM, f"Company: {' / '.join(ALIASES[r.ticker])}\nHeadline: {r.title_en}\n"
                    + (f"Summary: {r.lead_en}\n" if r.lead_en else "") + "Score:") for r in pairs.itertuples()]
    outs = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=4))
    raw = [o.outputs[0].text for o in outs]
    pairs["score"] = [int(m.group()) if (m := re.search(r"-?[0-2]", t)) else 0 for t in raw]
    pairs["parsed"] = [bool(re.search(r"-?[0-2]", t)) for t in raw]
    pairs[["url", "ticker", "score", "parsed"]].to_parquet(f"data/text/llm_{name}_en.parquet", index=False)
    print(f"translated {len(tagged)} items; scored {len(pairs)} pairs, parsed {pairs.parsed.mean()*100:.1f}%")
    print(pairs.score.value_counts().sort_index().to_string())
    for r in tagged.sample(8, random_state=0).itertuples():
        print(f"VI: {r.title}\nEN: {r.title_en}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="qwen25_7b", choices=list(MODELS))
    main(ap.parse_args().name)
