"""
Pick the multilingual embedding model for STRATA's strategy retrieval.

Test 1 — cross-lingual alignment (always): FLORES-200 dev parallel sentences.
  For each language X in {hi, te, ja}: embed N sentences in X and in English;
  recall@1 = share of X sentences whose nearest English neighbour is their
  own translation. (Also X->X' between hi/te/ja.) Needed for cross_language
  retrieval and for code-mixed responses to land near their pure-script peers.

Test 2 — response-state clustering (optional, --labeled):
  JSONL rows {"text", "label", "language"} — e.g. target responses labelled
  refusal / partial / full from a STRATA log. Reports leave-one-out 1-NN label
  accuracy within each language and across languages. This mirrors how STRATA
  retrieves: the query is the target's last response.

Backends:
  st:<hf_id>                      sentence-transformers, in-process
  api:<model>@<base_url>          OpenAI-compatible /v1/embeddings (LM Studio, oMLX)
An optional "|<prefix>" is prepended to every text (same prefix for both sides,
since retrieval here is symmetric: response vs response).

  python scripts/embedding_check.py --n 300 --models \
    "st:Qwen/Qwen3-Embedding-4B" "st:Qwen/Qwen3-Embedding-0.6B" \
    "st:google/embeddinggemma-2|task: sentence similarity | query: " \
    "st:intfloat/multilingual-e5-large-instruct|Instruct: Retrieve semantically similar text\nQuery: " \
    "st:BAAI/bge-m3"

Prefix notes (symmetric response-vs-response similarity):
  - EmbeddingGemma 2: model card's symmetric "SentenceSimilarity" template is
    "task: sentence similarity | query: {content}". Text-only use; its vision/audio
    encoders can be skipped to save memory (see the model card's config_kwargs).
  - multilingual-e5-*-instruct: requires an "Instruct: ...\nQuery: " prefix.
  - Qwen3-Embedding / bge-m3: no prefix needed for symmetric use.
"""

import argparse
import json
import os
import sys
import time
import urllib.request

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eval.runs import read_jsonl, run_dir, write_manifest  # noqa: E402
from src.paths import BENCHMARKS_DIR  # noqa: E402


# ---------------------------------------------------------------------------
# math (numpy only)
# ---------------------------------------------------------------------------

def _normalize(m):
    import numpy as np
    m = np.asarray(m, dtype=np.float32)
    return m / np.clip(np.linalg.norm(m, axis=1, keepdims=True), 1e-12, None)


def recall_at_1(src_emb, tgt_emb):
    """src_emb[i] should be nearest to tgt_emb[i] (cosine)."""
    import numpy as np
    s, t = _normalize(src_emb), _normalize(tgt_emb)
    sims = s @ t.T
    return float(np.mean(np.argmax(sims, axis=1) == np.arange(len(s))))


def loo_1nn_accuracy(emb, labels, query_idx=None, pool_idx=None):
    """Leave-one-out 1-NN label accuracy; queries from query_idx, neighbours from pool_idx."""
    import numpy as np
    e = _normalize(emb)
    n = len(labels)
    query_idx = list(range(n)) if query_idx is None else list(query_idx)
    pool_idx = list(range(n)) if pool_idx is None else list(pool_idx)
    if not query_idx or not pool_idx:
        return None
    correct = 0
    for q in query_idx:
        cands = [p for p in pool_idx if p != q]
        if not cands:
            continue
        sims = e[cands] @ e[q]
        nn = cands[int(np.argmax(sims))]
        correct += int(labels[nn] == labels[q])
    return correct / len(query_idx)


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------

class Embedder:
    def __init__(self, spec):
        self.spec = spec
        body, _, self.prefix = spec.partition("|")
        self.prefix = self.prefix.replace("\\n", "\n")
        self.kind, _, rest = body.partition(":")
        if self.kind == "st":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(rest, trust_remote_code=True)
        elif self.kind == "api":
            self.api_model, _, self.base_url = rest.partition("@")
            self.base_url = self.base_url.rstrip("/")
        else:
            raise ValueError(f"unknown backend in {spec}")

    def encode(self, texts, batch=32):
        texts = [self.prefix + t for t in texts]
        if self.kind == "st":
            return self.model.encode(texts, batch_size=batch, normalize_embeddings=True)
        out = []
        for i in range(0, len(texts), batch):
            req = urllib.request.Request(
                f"{self.base_url}/v1/embeddings",
                data=json.dumps({"model": self.api_model, "input": texts[i:i + batch]}).encode(),
                headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
            with urllib.request.urlopen(req, timeout=300) as r:
                data = json.loads(r.read())["data"]
            out.extend(d["embedding"] for d in sorted(data, key=lambda d: d["index"]))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=300, help="FLORES dev sentences per language")
    ap.add_argument("--labeled", default=None, help="optional JSONL {text,label,language}")
    ap.add_argument("--run_name", default="embedding_check")
    args = ap.parse_args()

    rows = read_jsonl(BENCHMARKS_DIR / "flores_dev.jsonl")[: args.n]
    labeled = read_jsonl(args.labeled) if args.labeled else []
    results = {}
    for spec in args.models:
        t0 = time.time()
        emb = Embedder(spec)
        vecs = {lang: emb.encode([r[lang] for r in rows]) for lang in ("en", "hi", "te", "ja")}
        res = {f"{x}->en": round(recall_at_1(vecs[x], vecs["en"]), 4) for x in ("hi", "te", "ja")}
        res.update({f"{a}->{b}": round(recall_at_1(vecs[a], vecs[b]), 4)
                    for a, b in (("hi", "te"), ("te", "hi"), ("hi", "ja"), ("te", "ja"))})
        if labeled:
            e = emb.encode([r["text"] for r in labeled])
            labels = [r["label"] for r in labeled]
            langs = sorted({r["language"] for r in labeled})
            res["loo_1nn_all"] = loo_1nn_accuracy(e, labels)
            for lang in langs:
                idx = [i for i, r in enumerate(labeled) if r["language"] == lang]
                other = [i for i, r in enumerate(labeled) if r["language"] != lang]
                res[f"loo_1nn_within_{lang}"] = loo_1nn_accuracy(e, labels, idx, idx)
                res[f"loo_1nn_cross_{lang}"] = loo_1nn_accuracy(e, labels, idx, other)
        res["seconds"] = round(time.time() - t0, 1)
        results[spec] = res
        print(f"\n{spec}\n  " + "  ".join(f"{k}={v}" for k, v in res.items()), flush=True)
        del emb

    write_manifest(run_dir(args.run_name), {"script": "embedding_check", "args": vars(args),
                                            "results": results}, name="embedding_summary.json")


if __name__ == "__main__":
    main()
