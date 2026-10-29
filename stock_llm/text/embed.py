"""Embed every corpus item (title + lead) with a frozen multilingual encoder. Cached by URL, so
re-running after the corpus grows only embeds new items.
Output: data/text/emb_<name>.npz with `url` and float16 `emb` (L2-normalised).
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

MODELS = {"bge_m3": "BAAI/bge-m3"}


def main(name, corpus="data/text/corpus.parquet", batch=128):
    path = f"data/text/emb_{name}.npz"
    c = pd.read_parquet(corpus)
    done_urls, done_emb = np.array([], dtype=object), np.zeros((0, 0), np.float16)
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        done_urls, done_emb = z["url"], z["emb"]
    todo = c[~c["url"].isin(set(done_urls))]
    print(f"{len(c)} items, {len(todo)} to embed", flush=True)
    if len(todo):
        model = SentenceTransformer(MODELS[name], device="cuda" if torch.cuda.is_available() else "cpu")
        model.half()
        text = (todo["title"] + ". " + todo["lead"]).str.strip(". ").tolist()
        emb = model.encode(text, batch_size=batch, normalize_embeddings=True, show_progress_bar=True,
                           convert_to_numpy=True).astype(np.float16)
        done_urls = np.concatenate([done_urls, todo["url"].to_numpy(dtype=object)])
        done_emb = emb if done_emb.size == 0 else np.concatenate([done_emb, emb])
        np.savez(path, url=done_urls, emb=done_emb)
    print(f"saved {len(done_urls)} embeddings, dim {done_emb.shape[1]}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="bge_m3", choices=list(MODELS))
    main(ap.parse_args().name)
