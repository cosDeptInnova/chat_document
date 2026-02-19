import os
import json
import pickle
import logging
from typing import Optional

logger = logging.getLogger("cosmos_nlp_v3")  # mantenemos el mismo logger

def load_vector_db_impl(self, index_filepath: Optional[str]):
    """
    Carga pickles (docnames, documents, metas, schema) y asegura la colección.
    Fuerza la reconstrucción diferida del BM25 en la siguiente búsqueda.
    """
    base_path = os.path.splitext(index_filepath or "user_index.pkl")[0]
    dn_path    = f"{base_path}_docnames.pkl"
    docs_path  = f"{base_path}_documents.pkl"
    metas_path = f"{base_path}_metas.pkl"
    meta_json  = f"{base_path}_schema.json"

    try:
        self.doc_names = pickle.load(open(dn_path, 'rb')) if os.path.exists(dn_path) else []
        self.documents = pickle.load(open(docs_path,'rb')) if os.path.exists(docs_path) else []
        if os.path.exists(metas_path):
            self._metas = pickle.load(open(metas_path,'rb'))
        else:
            self._metas = [{"source_path": ""} for _ in self.documents]

        schema = {"version": 2, "documents_len": len(self.documents)}
        try:
            if os.path.exists(meta_json):
                current = json.loads(open(meta_json, "r", encoding="utf-8").read())
                current["documents_len"] = len(self.documents)
                schema = current
            else:
                with self._get_user_lock():
                    self._atomic_write_file(
                        meta_json,
                        json.dumps(schema, ensure_ascii=False, indent=2).encode("utf-8")
                    )
        except Exception as e:
            logger.warning(f"(load_index) No se pudo leer/escribir schema.json: {e}")

        logger.info(
            f"(load_index) docs={len(self.documents)} | names={len(self.doc_names)} | "
            f"metas={len(self._metas)} | schema_v={schema.get('version')}"
        )
    except Exception as e:
        logger.warning(f"(load_index) No se pudieron cargar pickles: {e}. Inicio vacío.")
        self.doc_names, self.documents, self._metas = [], [], []

    self._ensure_collection()
    # Fuerza reconstrucción BM25 en próxima búsqueda
    try:
        self._bm25_index.ready = False
    except Exception:
        pass


def save_doc_names_impl(self, index_filepath: str):
    """
    Guarda doc_names de forma transaccional.
    """
    path = os.path.splitext(index_filepath)[0] + "_docnames.pkl"
    data = pickle.dumps(self.doc_names)
    with self._get_user_lock():
        self._atomic_write_file(path, data)


def save_metas_impl(self, index_filepath: str):
    """
    Guarda metadatos sidecar.
    """
    base = os.path.splitext(index_filepath)[0]
    path_metas = f"{base}_metas.pkl"
    try:
        data = pickle.dumps(self._metas)
        with self._get_user_lock():
            self._atomic_write_file(path_metas, data)
    except Exception as e:
        logger.warning(f"(save_metas) No se pudieron guardar metadatos: {e}")


def save_documents_impl(self, index_filepath: str):
    """
    Guarda documents y schema.json; llama a save_metas_impl.
    """
    base = os.path.splitext(index_filepath)[0]
    path_docs  = f"{base}_documents.pkl"
    schema_json= f"{base}_schema.json"

    with self._get_user_lock():
        self._atomic_write_file(path_docs, pickle.dumps(self.documents))
        save_metas_impl(self, index_filepath)
        try:
            schema = {"version": 2, "documents_len": len(self.documents)}
            self._atomic_write_file(
                schema_json,
                json.dumps(schema, ensure_ascii=False, indent=2).encode("utf-8")
            )
        except Exception as e:
            logger.warning(f"(save_documents) No se pudo escribir schema.json: {e}")
