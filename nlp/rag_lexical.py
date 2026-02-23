import os
import unicodedata
import re
import math
from typing import List, Dict, Any

class BM25Index:
    def __init__(self):
        self.k1 = 1.5
        self.b = 0.75
        self.min_df = 1
        self.N = 0
        self.avgdl = 1.0
        self.postings = {}
        self.df = {}
        self.idf = {}
        self.doc_len = []
        self.ready = False

    def build(self, documents: List[str]):
        self.k1 = float(os.getenv("BM25_K1", "1.5"))
        self.b  = float(os.getenv("BM25_B",  "0.75"))
        self.min_df = int(os.getenv("BM25_MIN_DF", "1"))

        STOP_ES_EN = set((
            "el","la","los","las","de","del","en","para","por","con","a","un","una","unos","unas","y","o",
            "que","es","son","al","lo","como","se","su","sus","más","mas","muy","ya","si","sí","no","the",
            "of","in","on","for","to","and","or","is","are","as","be","by","an","a","at","it","this","that"
        ))
        extra_sw = set((os.getenv("BM25_EXTRA_STOPWORDS","").strip() or "").split(",")) if os.getenv("BM25_EXTRA_STOPWORDS") else set()
        STOP = STOP_ES_EN | {w.strip() for w in extra_sw if w.strip()}

        def _norm(s: str) -> str:
            s = unicodedata.normalize("NFKD", s or "")
            s = "".join(ch for ch in s if not unicodedata.combining(ch))
            return s.lower()

        token_re = re.compile(r"[a-z0-9áéíóúüñ]+", re.IGNORECASE)
        postings, df, doc_len = {}, {}, []
        N = len(documents)

        for i, text in enumerate(documents):
            if not isinstance(text, str) or not text:
                doc_len.append(0)
                continue
            toks = [_norm(t) for t in token_re.findall(text)]
            if os.getenv("BM25_REMOVE_STOPWORDS", "1") == "1":
                toks = [t for t in toks if t not in STOP and len(t) > 1]
            doc_len.append(len(toks))
            if not toks:
                continue
            tf_local = {}
            for t in toks:
                tf_local[t] = tf_local.get(t, 0) + 1
            for t, tf in tf_local.items():
                postings.setdefault(t, {})[i] = tf

        postings_f, df_f = {}, {}
        for term, dct in postings.items():
            dfi = len(dct)
            if dfi >= self.min_df:
                postings_f[term] = dct
                df_f[term] = dfi

        avgdl = (sum(doc_len) / max(1, N))
        idf = {}
        for term, dfi in df_f.items():
            idf[term] = math.log(1.0 + ((N - dfi + 0.5) / (dfi + 0.5)))

        self.N = N
        self.avgdl = float(avgdl)
        self.postings = postings_f
        self.df = df_f
        self.idf = idf
        self.doc_len = doc_len
        self.ready = True

    def search(self, query: str, documents: List[str], doc_names: List[str], limit: int = 100) -> List[Dict[str, Any]]:
        import unicodedata, re

        if not self.ready or self.N != len(documents):
            self.build(documents)
        if not documents:
            return []

        def _norm(s: str) -> str:
            s = unicodedata.normalize("NFKD", s or "")
            s = "".join(ch for ch in s if not unicodedata.combining(ch))
            return s.lower()

        token_re = re.compile(r"[a-z0-9áéíóúüñ]+", re.IGNORECASE)
        toks = [_norm(t) for t in token_re.findall(query or "")]
        if os.getenv("BM25_REMOVE_STOPWORDS", "1") == "1":
            STOP_ES_EN = set((
                "el","la","los","las","de","del","en","para","por","con","a","un","una","unos","unas","y","o",
                "que","es","son","al","lo","como","se","su","sus","más","mas","muy","ya","si","sí","no","the",
                "of","in","on","for","to","and","or","is","are","as","be","by","an","a","at","it","this","that"
            ))
            extra_sw = set((os.getenv("BM25_EXTRA_STOPWORDS","").strip() or "").split(",")) if os.getenv("BM25_EXTRA_STOPWORDS") else set()
            STOP = STOP_ES_EN | {w.strip() for w in extra_sw if w.strip()}
            toks = [t for t in toks if t not in STOP and len(t) > 1]

        postings = self.postings
        idf = self.idf
        doc_len = self.doc_len
        k1 = self.k1
        b  = self.b
        avgdl = self.avgdl

        scores = {}
        for t in toks:
            if t not in postings:
                continue
            t_idf = idf.get(t, 0.0)
            for doc_id, tf in postings[t].items():
                dl = max(1, doc_len[doc_id])
                denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
                part = t_idf * ((tf * (k1 + 1.0)) / denom)
                scores[doc_id] = scores.get(doc_id, 0.0) + part

        if not scores:
            return []

        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max(1, int(limit))]
        results = []
        for rank, (doc_id, sc) in enumerate(ordered, start=1):
            txt = documents[doc_id] if 0 <= doc_id < len(documents) else ""
            dname = doc_names[doc_id] if doc_id < len(doc_names) else ""
            results.append({
                "uid": None,
                "doc_id": doc_id,
                "text": txt or "",
                "doc_name": dname,
                "score": float(sc),
                "rank": rank
            })
        return results
