import os
from typing import List

_SPACY_NLP = None

def _get_nlp():
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy
        model = os.getenv("SPACY_MODEL_ES", "es_core_news_md")
        _SPACY_NLP = spacy.load(model)
    return _SPACY_NLP


def dynamic_fragment_size_impl(total_tokens: int) -> int:
    """
    Se mantiene firma y trazas (print) del original.
    """
    print(f"(fragment_size) total_tokens={total_tokens}", flush=True)
    # Configurable para producción (sin romper backward compatibility):
    # - documentos cortos: chunks pequeños para precisión
    # - documentos medianos/largos: chunks más amplios para capturar secciones completas
    short = int(os.getenv("RAG_CHUNK_TOKENS_SHORT", "90"))
    medium = int(os.getenv("RAG_CHUNK_TOKENS_MEDIUM", "180"))
    large = int(os.getenv("RAG_CHUNK_TOKENS_LARGE", "280"))
    t1 = int(os.getenv("RAG_CHUNK_BREAKPOINT_1", "320"))
    t2 = int(os.getenv("RAG_CHUNK_BREAKPOINT_2", "1800"))

    size = short if total_tokens < t1 else medium if total_tokens < t2 else large
    print(f"(fragment_size) size={size}", flush=True)
    return size


def fragment_text_semantically_impl(
    text: str,
    max_tokens: int = 100,
    overlap_tokens: int = 33
) -> List[str]:
    """
    Segmentación por oraciones con refuerzo semántico en UNA sola pasada de spaCy.
    Conserva las trazas y el comportamiento del método original.
    """
    import re

    min_overlap = int(os.getenv("RAG_CHUNK_MIN_OVERLAP_TOKENS", "40"))
    overlap_ratio = float(os.getenv("RAG_CHUNK_OVERLAP_RATIO", "0.30"))
    overlap_tokens = max(overlap_tokens, min_overlap, int(max_tokens * overlap_ratio))

    print(f"(fragment) Iniciando fragmentación semántica: max_tokens={max_tokens}, overlap={overlap_tokens}", flush=True)

    title_re  = re.compile(r"^\s*([A-ZÁÉÍÓÚÜÑ0-9][A-ZÁÉÍÓÚÜÑ0-9 \-_]{3,}:?|#{1,6}\s+.+)$")
    bullet_re = re.compile(r"^\s*(?:[-*•\u2022]|\d{1,2}\.)\s+")

    lines = [ln.rstrip() for ln in text.splitlines()]
    blocks, buf = [], []
    for ln in lines:
        if title_re.match(ln) and buf:
            blocks.append("\n".join(buf).strip()); buf = [ln]
        else:
            buf.append(ln)
    if buf:
        blocks.append("\n".join(buf).strip())

    def split_block(btxt: str):
        parts, cur = [], []
        for ln in btxt.splitlines():
            if bullet_re.match(ln) or not ln.strip():
                cur.append(ln)
            else:
                if cur:
                    parts.append("\n".join(cur).strip()); cur = []
                parts.append(ln)
        if cur:
            parts.append("\n".join(cur).strip())
        return [p for p in parts if p.strip()]

    segments = []
    segment_headers = []
    active_header = ""
    for b in blocks:
        parts = split_block(b)
        if not parts:
            continue
        first_line = parts[0].splitlines()[0].strip() if parts[0].strip() else ""
        if title_re.match(first_line):
            active_header = first_line
        for p in parts:
            segments.append(p)
            segment_headers.append(active_header)

    # ÚNICO pase spaCy sobre todos los segmentos
    nlp = _get_nlp()
    batch_size = int(os.getenv("NLP_PIPE_BATCH", "32"))
    seg_docs = list(nlp.pipe(segments, batch_size=batch_size))

    # Construye frases y cuenta tokens por sentencia sin más llamadas a nlp()
    sent_texts: List[str] = []
    sent_tokens: List[int] = []
    for seg_header, d in zip(segment_headers, seg_docs):
        for s in d.sents:
            st = s.text.strip()
            if st:
                if seg_header and seg_header.lower() not in st.lower():
                    st = f"{seg_header}\n{st}"
                sent_texts.append(st)
                sent_tokens.append(len(s))

    # Empaquetado con solapamiento
    fragments, current, counts, total = [], [], [], 0
    for st, tk in zip(sent_texts, sent_tokens):
        current.append(st); counts.append(tk); total += tk
        if total >= max_tokens:
            frag = " ".join(current).strip()
            fragments.append(frag)
            # overlap
            overlap, o_counts, total = [], [], 0
            for s, cc in reversed(list(zip(current, counts))):
                if not s.strip():
                    continue
                overlap.insert(0, s); o_counts.insert(0, cc); total += cc
                if total >= overlap_tokens:
                    break
            current, counts = overlap, o_counts
    if current:
        fragments.append(" ".join(current).strip())

    print(f"(fragment) Total fragments generados: {len(fragments)}", flush=True)
    return fragments
