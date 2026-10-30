"""Retrieval: BM25 over the rule book and nearest-neighbour search over past resolved tickets (KB)."""

import gzip
import json
import math
from collections import Counter
from pathlib import Path

from .rules import RuleBook
from .textnorm import stems


class BM25:
    def __init__(self, docs: list[list[str]], k1: float = 1.4, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = docs
        self.N = len(docs)
        self.avgdl = sum(len(d) for d in docs) / max(1, self.N)
        self.tf = [Counter(d) for d in docs]
        df = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        out = []
        for i, tf in enumerate(self.tf):
            dl = len(self.docs[i])
            s = 0.0
            for q in query:
                f = tf.get(q)
                if not f:
                    continue
                s += self.idf.get(q, 0.0) * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
            out.append(s)
        return out


class RuleRetriever:
    def __init__(self, book: RuleBook):
        self.book = book
        self.ids = [r.id for r in book.rules]
        self.bm25 = BM25([stems(r.full_text + " " + " ".join(r.aliases)) for r in book.rules])

    def search(self, query: str, k: int = 5, extra_terms: list[str] | None = None) -> list[tuple[str, float]]:
        q = stems(query)
        if extra_terms:
            q += [t.lower()[:5] for t in extra_terms]
        if not q or not self.ids:
            return []
        sc = self.bm25.scores(q)
        ranked = sorted(zip(self.ids, sc), key=lambda x: -x[1])
        return [(rid, round(s, 3)) for rid, s in ranked[:k] if s > 0]

    def by_hints(self, hints: list[str], k: int = 3) -> list[str]:
        """Rules whose text contains the concept's hint stems (cross-lingual bridge)."""
        if not hints:
            return []
        scored = []
        for r in self.book.rules:
            txt = (r.full_text + " " + " ".join(r.aliases)).lower()
            s = sum(1 for h in hints if h.lower() in txt)
            if s:
                # prefer leaf items ("2.1") over whole sections ("2")
                scored.append((s + (0.5 if "." in r.id else 0.0), r.id))
        scored.sort(key=lambda x: -x[0])
        return [rid for _, rid in scored[:k]]


class KnowledgeBase:
    """Past tickets with admin replies (built by `make train`): similar-case retrieval."""

    def __init__(self, records: list[dict], vectorizer=None, matrix=None):
        self.records = records
        self.vectorizer = vectorizer
        self.matrix = matrix

    @classmethod
    def load(cls, models_dir: Path) -> "KnowledgeBase | None":
        rec_path = models_dir / "kb.jsonl.gz"
        idx_path = models_dir / "kb_index.joblib"
        if not rec_path.exists() or not idx_path.exists():
            return None
        import joblib
        with gzip.open(rec_path, "rt", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]
        idx = joblib.load(idx_path)
        return cls(records, idx["vectorizer"], idx["matrix"])

    def search(self, text: str, kind: str | None = None, k: int = 5, min_sim: float = 0.25) -> list[dict]:
        if not text or self.vectorizer is None:
            return []
        q = self.vectorizer.transform([text])
        sims = (self.matrix @ q.T).toarray().ravel()
        top = sims.argsort()[::-1][: k * 6]
        out = []
        for i in top:
            s = float(sims[i])
            if s < min_sim:
                break
            r = self.records[i]
            if kind and r.get("kind") != kind:
                continue
            out.append({**r, "sim": round(s, 3)})
            if len(out) >= k:
                break
        return out
