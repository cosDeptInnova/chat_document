import asyncio
import random
import logging
import re

import httpx
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS, exceptions as ddgs_exceptions
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np

# --- Logging configuration ---
logging.basicConfig(
    force=True,
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# --- Initialization of NLP and SBERT ---
try:
    nlp = spacy.load("es_core_news_md")  # spacy Spanish model
except Exception:
    raise Exception("Ejecuta: python -m spacy download es_core_news_md")

sbert_model = SentenceTransformer("intfloat/multilingual-e5-large")

# --- Cache for page contents ---
_content_cache: dict[str, str] = {}

# --- HTTP client with follow redirects ---
_SESSION = httpx.AsyncClient(timeout=30.0, follow_redirects=True)

# --- User-Agent rotation pool ---
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
]
ACCEPT_HEADER = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
LANGUAGE_HEADER = "es-ES,es;q=0.8"

# --- Concurrency control ---
SEM = asyncio.Semaphore(5)

# --- Helper: rotated headers ---
def get_headers() -> dict[str, str]:
    return {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": ACCEPT_HEADER,
        "Accept-Language": LANGUAGE_HEADER,
    }

# --- Helper: safe GET with retry/backoff on 202 ---
async def safe_get(url: str, **kwargs) -> httpx.Response:
    backoff = 1
    for attempt in range(4):
        resp = await _SESSION.get(url, **kwargs)
        if resp.status_code == 202:
            logger.warning(f"Rate-limited (202). Retry in {backoff}s...")
            await asyncio.sleep(backoff)
            backoff *= 2
            continue
        resp.raise_for_status()
        return resp
    logger.error("Exceeded retries due to rate limit (202)")
    resp.raise_for_status()  # will raise the last status

# --- HTML cleaning utility ---

def clean_html(html_text: str) -> str:
    try:
        soup = BeautifulSoup(html_text, "html.parser")
    except Exception as e:
        logger.error(f"Error parsing HTML: {e}")
        text = re.sub(r'<[^>]+>', '', html_text)
        return re.sub(r'\s+', ' ', text)

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return re.sub(r'\s+', ' ', text)

# --- Fetch and clean page content with concurrency limit ---
async def fetch_page_content(url: str) -> str:
    async with SEM:
        try:
            resp = await safe_get(url)
            return resp.text
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return ""

async def get_clean_content(url: str, depth: int = 2) -> str:
    if url in _content_cache:
        return _content_cache[url]

    html = await fetch_page_content(url)
    main_text = clean_html(html)
    aggregated = [main_text]

    if depth > 0:
        soup = BeautifulSoup(html, "html.parser")
        links = [a["href"] for a in soup.find_all("a", href=True) if a["href"].startswith("http")]
        for link in links[:2]:
            aggregated.append(await get_clean_content(link, depth - 1))

    content = "\n".join(aggregated)
    _content_cache[url] = content
    return content

# --- DuckDuckGo search using HTML scraping ---
async def duckduckgo_search(query: str, max_results: int = 6) -> list[dict]:
    search_url = 'https://duckduckgo.com/html/'
    params = {'q': query}

    try:
        resp = await safe_get(
            search_url,
            params=params,
            headers=get_headers(),
            timeout=10.0
        )
    except Exception as e:
        logger.error(f"DuckDuckGo HTML request failed: {e}")
        return []

    soup = BeautifulSoup(resp.text, 'html.parser')
    results = []
    for result in soup.select('div.result')[:max_results]:
        link_tag = result.find('a', class_='result__a')
        snippet_tag = result.find('a', class_='result__snippet') or result.find('div', class_='result__snippet')
        href = link_tag['href'] if link_tag and link_tag.has_attr('href') else None
        snippet = snippet_tag.get_text(strip=True) if snippet_tag else ''
        if href:
            results.append({'href': href, 'snippet': snippet})
    logger.debug(f"Found {len(results)} DuckDuckGo results for '{query}'")
    return results

# --- Preprocessing and ranking pipeline ---
def preprocess_query(query: str) -> str:
    doc = nlp(query)
    return " ".join(token.lemma_ for token in doc if not token.is_stop)

def compute_relevance(q_emb: np.ndarray, c_emb: np.ndarray) -> float:
    dot = np.dot(q_emb, c_emb)
    n1, n2 = np.linalg.norm(q_emb), np.linalg.norm(c_emb)
    return dot / (n1 * n2) if n1 and n2 else 0.0

async def rank_results(query: str, results: list[dict]) -> list[dict]:
    qproc = preprocess_query(query)
    q_emb = sbert_model.encode(qproc)
    for r in results:
        text = f"{r.get('title','')}. {r.get('snippet','')} {r.get('content','')}"
        c_emb = sbert_model.encode(text)
        r["relevance"] = compute_relevance(q_emb, c_emb)
    return sorted(results, key=lambda x: x["relevance"], reverse=True)

# --- Aggregator for search + content + ranking ---
async def aggregate_search(query: str, provider: str = "duckduckgo") -> list[dict]:
    if provider.lower() != "duckduckgo":
        raise ValueError("Sólo 'duckduckgo' soportado.")
    raw = await duckduckgo_search(query)
    tasks = [get_clean_content(r['href'], depth=1) for r in raw]
    contents = await asyncio.gather(*tasks)
    for r, content in zip(raw, contents):
        r['content'] = content
    ranked = await rank_results(query, raw)
    logger.info(f"Returning {len(ranked)} ranked results for '{query}'")
    return ranked
