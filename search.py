"""Semantic search over OCR text.

Backends
  dense  : multilingual sentence embeddings + FAISS (cross-lingual search,
           e.g. an English query can find a Kannada passage).
  tfidf  : character n-gram TF-IDF. No downloads, works offline, tolerant of OCR
           spelling errors, but only matches the SAME language/script.

SEARCH_BACKEND = auto (default) | dense | tfidf
DENSE_MODEL    = intfloat/multilingual-e5-small (default; ~100 languages incl. Kannada)
"""
import json
import os
import re

import numpy as np

DENSE_MODEL = os.getenv("DENSE_MODEL", "intfloat/multilingual-e5-small")
_SENT_SPLIT = re.compile(r"(?<=[.!?।॥])\s+")  # includes Devanagari danda


def chunk_text(text, size=300):
    """Split page text into passages of roughly `size` characters, on sentence
    boundaries, never across paragraphs. OCR line wraps are joined."""
    chunks = []
    for para in re.split(r"\n\s*\n", text):
        para = " ".join(para.split())
        if not para:
            continue
        buf = ""
        for sent in _SENT_SPLIT.split(para):
            if buf and len(buf) + len(sent) + 1 > size:
                chunks.append(buf)
                buf = sent
            else:
                buf = f"{buf} {sent}".strip()
        if buf:
            chunks.append(buf)
    return chunks


# --------------------------------------------------------------------- backends
class DenseBackend:
    name = "dense"

    def __init__(self, model_name=DENSE_MODEL):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.e5 = "e5" in model_name.lower()  # e5 models need query:/passage: prefixes

    def encode(self, texts, kind="passage"):
        prefix = f"{kind}: " if self.e5 else ""
        vecs = self.model.encode(
            [prefix + t for t in texts],
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype="float32")


def make_backend(preference=None):
    preference = (preference or os.getenv("SEARCH_BACKEND", "auto")).lower()
    if preference in ("auto", "dense"):
        try:
            return DenseBackend()
        except Exception as e:
            if preference == "dense":
                raise
            print(f"[search] dense backend unavailable ({type(e).__name__}); using tfidf fallback")
    return None  # None => tfidf


# ------------------------------------------------------------------------ index
class SearchIndex:
    def __init__(self, backend=None):
        self.backend = backend  # DenseBackend or None (tfidf)
        self.chunks = []        # [{doc_id, doc, page, text}]
        self.emb = None         # float32 (n, dim) for dense
        self._faiss = None
        self._vec = None        # tfidf vectorizer
        self._mat = None

    # -- info
    @property
    def mode(self):
        return "dense" if self.backend else "tfidf"

    def info(self):
        faiss_ok = False
        try:
            import faiss  # noqa: F401
            faiss_ok = True
        except Exception:
            pass
        return {
            "mode": self.mode,
            "model": getattr(self.backend, "model_name", None),
            "faiss": faiss_ok and self.mode == "dense",
            "chunks": len(self.chunks),
        }

    # -- building
    def _rebuild_search_structs(self):
        if not self.chunks:
            self._faiss = self._vec = self._mat = None
            return
        if self.backend:
            try:
                import faiss

                self._faiss = faiss.IndexFlatIP(self.emb.shape[1])
                self._faiss.add(self.emb)
            except Exception:
                self._faiss = None  # numpy dot-product fallback in search()
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
            self._mat = self._vec.fit_transform([c["text"] for c in self.chunks])

    def add_chunks(self, new_chunks):
        if not new_chunks:
            return
        if self.backend:
            new_emb = self.backend.encode([c["text"] for c in new_chunks], "passage")
            self.emb = new_emb if self.emb is None else np.vstack([self.emb, new_emb])
        self.chunks.extend(new_chunks)
        self._rebuild_search_structs()

    def remove_doc(self, doc_id):
        keep = [i for i, c in enumerate(self.chunks) if c["doc_id"] != doc_id]
        self.chunks = [self.chunks[i] for i in keep]
        if self.backend and self.emb is not None:
            self.emb = self.emb[keep] if keep else None
        self._rebuild_search_structs()

    # -- searching
    def search(self, query, k=5):
        query = query.strip()
        if not query or not self.chunks:
            return []
        k = min(k, len(self.chunks))
        if self.backend:
            q = self.backend.encode([query], "query")
            if self._faiss is not None:
                scores, ids = self._faiss.search(q, k)
                pairs = list(zip(ids[0], scores[0]))
            else:
                sc = (self.emb @ q.T).ravel()
                top = np.argsort(-sc)[:k]
                pairs = [(i, sc[i]) for i in top]
        else:
            sc = (self._mat @ self._vec.transform([query]).T).toarray().ravel()
            top = np.argsort(-sc)[:k]
            pairs = [(i, sc[i]) for i in top if sc[i] > 0.03]
        out = []
        for i, s in pairs:
            if i < 0:
                continue
            c = self.chunks[int(i)]
            out.append({"doc_id": c["doc_id"], "doc": c["doc"], "page": c["page"],
                        "text": c["text"], "score": round(float(s), 4)})
        return out

    # -- persistence
    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "chunks.json"), "w", encoding="utf-8") as f:
            json.dump({"mode": self.mode, "model": getattr(self.backend, "model_name", None),
                       "chunks": self.chunks}, f, ensure_ascii=False)
        path = os.path.join(folder, "emb.npy")
        if self.backend and self.emb is not None:
            np.save(path, self.emb)
        elif os.path.exists(path):
            os.remove(path)

    def load(self, folder):
        p = os.path.join(folder, "chunks.json")
        if not os.path.exists(p):
            return False
        with open(p, encoding="utf-8") as f:
            saved = json.load(f)
        self.chunks = saved["chunks"]
        if self.backend:
            e = os.path.join(folder, "emb.npy")
            same_model = saved.get("model") == getattr(self.backend, "model_name", None)
            if same_model and os.path.exists(e):
                self.emb = np.load(e)
            elif self.chunks:  # saved with another backend/model: re-encode
                self.emb = self.backend.encode([c["text"] for c in self.chunks], "passage")
        self._rebuild_search_structs()
        return True
