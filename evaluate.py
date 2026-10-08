"""nDCG@10 for every (quant, doc_mode, query_mode, dim) + BM25 baseline.
ArguAna: the query's own id is excluded from results (MTEB convention)."""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

DATASETS = ["scifact", "nfcorpus", "arguana"]
QUANTS = ["Q4", "BF16"]
DIMS = [768, 512, 256, 128]
K = 10


def ndcg(ranked, rel):
    dcg = sum((2 ** rel.get(d, 0) - 1) / np.log2(i + 2) for i, d in enumerate(ranked[:K]))
    ideal = sorted(rel.values(), reverse=True)[:K]
    idcg = sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def norm(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def rank_dense(Q, Dm, doc_ids, qids, ignore_self):
    S = norm(Q) @ norm(Dm).T
    pos = {d: i for i, d in enumerate(doc_ids)}
    if ignore_self:
        for qi, q in enumerate(qids):
            if q in pos:
                S[qi, pos[q]] = -9
    top = np.argsort(-S, axis=1)[:, :K]
    return [[doc_ids[j] for j in row] for row in top]


rows, tops = [], {}
for D in DATASETS:
    ids = json.load(open(f"emb/{D}_ids.json"))
    qids, doc_ids = ids["qids"], ids["doc_ids"]
    qr = pd.read_parquet(f"data/{D}/default_test.parquet")
    qr["query-id"] = qr["query-id"].astype(str)
    qr["corpus-id"] = qr["corpus-id"].astype(str)
    rel = {q: dict(zip(g["corpus-id"], g["score"].astype(int))) for q, g in qr.groupby("query-id")}
    ign = D == "arguana"

    # BM25
    corpus = pd.read_parquet(f"data/{D}/corpus_corpus.parquet")
    queries = pd.read_parquet(f"data/{D}/queries_queries.parquet")
    qmap = dict(zip(queries["_id"].astype(str), queries["text"]))
    tok = lambda s: re.findall(r"\w+", s.lower())
    bm = BM25Okapi([tok(f"{t} {x}") for t, x in zip(corpus["title"], corpus["text"])])
    bmr = []
    for q in qids:
        s = bm.get_scores(tok(qmap[q]))
        if ign and q in doc_ids:
            s[doc_ids.index(q)] = -9
        bmr.append([doc_ids[j] for j in np.argsort(-s)[:K]])
    rows.append({"dataset": D, "system": "BM25", "ndcg10": np.mean([ndcg(r, rel[q]) for r, q in zip(bmr, qids)])})

    for Qt in QUANTS:
        for dm in ("prefix",):
            Dm = np.load(f"emb/{D}_{Qt}_doc_{dm}.npy")
            for qm in ("prefix", "raw"):
                Q = np.load(f"emb/{D}_{Qt}_query_{qm}.npy")
                for dim in DIMS:
                    ranked = rank_dense(Q[:, :dim], Dm[:, :dim], doc_ids, qids, ign)
                    sc = [ndcg(r, rel[q]) for r, q in zip(ranked, qids)]
                    rows.append({"dataset": D, "system": "EG2", "quant": Qt, "doc": dm, "query": qm, "dim": dim,
                                 "ndcg10": float(np.mean(sc))})
                    tops[(D, Qt, dm, qm, dim)] = ranked

# top-10 agreement vs BF16 (same config)
agree = []
for (D, Qt, dm, qm, dim), r in tops.items():
    if Qt == "BF16":
        continue
    ref = tops[(D, "BF16", dm, qm, dim)]
    agree.append({"dataset": D, "quant": Qt, "doc": dm, "query": qm, "dim": dim,
                  "top10_overlap": float(np.mean([len(set(a) & set(b)) / K for a, b in zip(r, ref)])),
                  "top1_same": float(np.mean([a[0] == b[0] for a, b in zip(r, ref)]))})

df = pd.DataFrame(rows)
df.to_csv("results_ndcg.csv", index=False)
pd.DataFrame(agree).to_csv("results_agreement.csv", index=False)
pd.set_option("display.width", 200)
print(df.round(4).to_string(index=False))
