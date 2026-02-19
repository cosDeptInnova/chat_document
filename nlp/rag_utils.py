from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import importlib
import os
from typing import Optional

def ensure_local_with_transformers(model_id: str, local_dir: str, task: str = "auto") -> str:
    os.makedirs(local_dir, exist_ok=True)

    # Si el fast download está activo pero no hay hf_transfer, lo apagamos
    if os.getenv("HF_HUB_ENABLE_HF_TRANSFER", "") in ("1", "true", "True"):
        try:
            
            importlib.import_module("hf_transfer")
        except Exception:
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

    has_cfg = os.path.exists(os.path.join(local_dir, "config.json"))
    has_tok = any(os.path.exists(os.path.join(local_dir, f)) for f in ("tokenizer.json","vocab.txt","spiece.model"))
    has_wts = any(os.path.exists(os.path.join(local_dir, f)) for f in ("pytorch_model.bin","model.safetensors"))
    if has_cfg and has_tok and has_wts:
        return local_dir

    
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = (AutoModelForSequenceClassification.from_pretrained(model_id)
           if task == "seq-cls" else
           AutoModel.from_pretrained(model_id))
    tok.save_pretrained(local_dir)
    mdl.save_pretrained(local_dir)
    return local_dir

def _safe_dir_name(s: str) -> str:
    return s.replace("/", "--").replace(":", "_")

def _dir_has_hf_model(path: str) -> bool:
        if not path or not os.path.isdir(path):
            return False
        has_cfg = os.path.exists(os.path.join(path, "config.json"))
        has_wts = any(os.path.exists(os.path.join(path, f)) for f in ("model.safetensors", "pytorch_model.bin"))
        has_tok = any(os.path.exists(os.path.join(path, f)) for f in ("tokenizer.json", "spiece.model", "sentencepiece.bpe.model", "vocab.txt"))
        return has_cfg and has_wts and has_tok

def _atomic_write_file(path: str, data: bytes):
        """
        Escritura transaccional de ficheros (pickles/metadata) para evitar corrupción.
        """
        import os, tempfile
        d = os.path.dirname(os.path.abspath(path)) or "."
        with tempfile.NamedTemporaryFile(dir=d, delete=False) as tmp:
            tmp.write(data)
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp_path = tmp.name
        os.replace(tmp_path, path)

def _canon(s: str) -> str:
    return " ".join((s or "").split()).strip().lower()

def _canon_key(s: str) -> str:
    import unicodedata, re
    s = s or ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = re.sub(r"[^a-z0-9]+", "_", s).strip("_")
    return s[:50] or "col"

def _canon_val(s: str) -> str:
    import unicodedata
    s = s or ""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return s.strip().lower()

def _parse_number_es(s: str) -> Optional[float]:
    import re
    try:
        t = str(s).strip()
        if not t:
            return None
        has_comma = "," in t
        has_dot = "." in t
        if has_comma and has_dot:
            if t.rfind(",") > t.rfind("."):
                t2 = t.replace(".", "").replace(",", ".")
            else:
                t2 = t.replace(",", "")
        elif has_comma and not has_dot:
            if re.fullmatch(r"\d{1,3}(,\d{3})+", t):
                t2 = t.replace(",", "")
            else:
                t2 = t.replace(",", ".")
        else:
            t2 = t
        return float(re.sub(r"[^0-9\.\-]", "", t2))
    except Exception:
        return None