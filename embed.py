"""Embed corpus + queries of one BEIR dataset through a running llama-server.
Usage: embed.py DATASET QUANT_TAG PORT
Writes emb/{dataset}_{quant}_{side}_{mode}.npy  (side=doc|query, mode=prefix|raw)
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import requests

D, TAG, PORT = sys.argv[1], sys.argv[2], sys.argv[3]
URL = f"http://127.0.0.1:{PORT}/v1/embeddings"
os.makedirs("emb", exist_ok=True)

corpus = pd.read_parquet(f"data/{D}/corpus_corpus.parquet")
queries = pd.read_parquet(f"data/{D}/queries_queries.parquet")
qrels = pd.read_parquet(f"data/{D}/default_test.parquet")
qids = sorted(qrels["query-id"].astype(str).unique())
qmap = dict(zip(queries["_id"].astype(str), queries["text"]))
qtexts = [qmap[q] for q in qids]


def doc_text(t, x, mode):
    t = (t or "").strip()
    if mode == "prefix":
        return f"title: {t or 'none'} | text: {x}"
    return f"{t}. {x}" if t else x


def q_text(q, mode):
    return f"task: search result | query: {q}" if mode == "prefix" else q


def call(batch):
    for i in range(5):
        try:
            r = requests.post(URL, json={"input": batch}, timeout=600)
            r.raise_for_status()
            return [e["embedding"] for e in sorted(r.json()["data"], key=lambda e: e["index"])]
        except Exception as e:  # noqa
            err = e
            time.sleep(2 * (i + 1))
    raise err


def embed(texts, bs=16, workers=8):
    batches = [texts[i:i + bs] for i in range(0, len(texts), bs)]
    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(call, batches))
    return np.array([v for b in out for v in b], dtype=np.float32)


meta = {}
# Docs embedded once with the documented prefix (CPU budget). The prefix test varies
# only the QUERY side: the common bug is query code that forgets the prefix.
for mode in ("prefix", "raw"):
    jobs = {"query": [q_text(q, mode) for q in qtexts]}
    if mode == "prefix":
        jobs["doc"] = [doc_text(t, x, mode) for t, x in zip(corpus["title"], corpus["text"])]
    for side, texts in jobs.items():
        f = f"emb/{D}_{TAG}_{side}_{mode}.npy"
        if os.path.exists(f):
            continue
        t0 = time.time()
        E = embed(texts)
        dt = time.time() - t0
        np.save(f, E)
        meta[f] = {"n": len(texts), "sec": round(dt, 1), "per_item_ms": round(1000 * dt / len(texts), 2), "dim": E.shape[1]}
        print(f, meta[f], flush=True)

json.dump({"qids": qids, "doc_ids": corpus["_id"].astype(str).tolist()}, open(f"emb/{D}_ids.json", "w"))
with open("emb/timing.jsonl", "a") as fh:
    for k, v in meta.items():
        fh.write(json.dumps({"file": k, **v}) + "\n")
