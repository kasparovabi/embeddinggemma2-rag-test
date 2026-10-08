# Three "harmless" settings for EmbeddingGemma 2 in a RAG pipeline

EmbeddingGemma 2 (Google DeepMind, 740M multimodal, 270M text path) shipped this week and the `unsloth/embeddinggemma-2-GGUF` build is trending on Hugging Face. Anyone dropping it into a retrieval pipeline makes three quick decisions:

1. **Quantization.** The 4-bit file is 176 MB, the BF16 file is 558 MB.
2. **Vector size.** The model supports Matryoshka truncation to 512, 256 or 128 dims.
3. **Task prefix.** The model card asks for `task: search result | query: ...` on queries. It is easy to forget, and nothing errors when you do.

This repo measures each one, and all three together, on public retrieval sets with known answers.

## Setup

| | |
|---|---|
| Model files | `unsloth/embeddinggemma-2-GGUF`: `UD-Q4_K_XL` and `BF16` (sha256 in `results/model_sha256.txt`) |
| Runtime | llama.cpp `b11496` server, `--embeddings --pooling mean`, CUDA, RTX 5070 Ti 16 GB |
| Datasets | BEIR via MTEB: SciFact, NFCorpus, ArguAna (test splits) |
| Size | 17,490 documents, 2,029 queries |
| Metric | nDCG@10, mean over the three datasets |
| Baseline | BM25 (k1=1.5, b=0.75), no model |

Documents are always embedded with the documented format `title: {title} | text: {text}`. The prefix test changes only the query side, because that is where the bug usually lives.

## Results (mean nDCG@10 over 3 datasets)

| Setting | nDCG@10 | vs full |
|---|---|---|
| BF16, query prefix, 768d | 0.602 | reference |
| 4-bit, same | 0.599 | -0.4% |
| 256d (3x less storage) | 0.586 | -2.6% |
| 128d (6x less storage) | 0.541 | -10.1% |
| Query prefix forgotten, 768d | 0.582 | -3.3% |
| Prefix forgotten + 128d + 4-bit | 0.472 | -21.6% |
| BM25 | 0.448 | -25.5% |

Per dataset numbers: `results/ndcg10.csv`.

**What it says**

- 4-bit is almost free on average. It is not identical: the top-1 result differs from BF16 on 7% to 22% of queries depending on the dataset (`results/q4_vs_bf16_agreement.csv`).
- A forgotten query prefix costs 3.3% at 768d and 11.1% at 128d. Truncation amplifies it.
- All three shortcuts together leave the model 5% above BM25, down from 34% above.
- On the GPU, 4-bit was not faster. Both files ran at about 9.5 ms per document, with about 4.5 GB of GPU memory in use (`results/timing.jsonl`, `results/memory.jsonl`). The gain is disk and download size.

**A llama.cpp pitfall we hit**

`llama-server` splits `-c` across the `-np` parallel slots. With `-c 8192 -np 8` each request gets 1,024 tokens, and longer documents are rejected with HTTP 400. An indexing script that does not check errors will skip them. The runs here use `-c 65536 -np 8` and logged zero rejections.

## Limits

- Three English BEIR sets. Not multilingual, not multimodal, not your corpus.
- One GPU, one llama.cpp build. CPU timing was not measured end to end.
- 8-bit and 512d sit in between and were left out of the headline table (512d is in the CSV).
- Means over datasets of different sizes, each dataset weighted equally.

## Reproduce

```
pip install numpy pandas pyarrow rank_bm25 requests
# data: MTEB parquet files for scifact, nfcorpus, arguana into data/<name>/
# models: the two GGUF files into models/
powershell -File run_windows_cuda.ps1     # starts llama-server per file, embeds, evaluates
```

`embed.py DATASET TAG PORT` embeds one dataset through a running `llama-server`. `evaluate.py` computes all tables.

## License

Code MIT. Datasets and model belong to their owners under their own licenses.
