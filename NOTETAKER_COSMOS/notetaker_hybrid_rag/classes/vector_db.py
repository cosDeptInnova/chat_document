# Clase para Qdrant (Donde crearemos la colección y subiremos los vectores. La función es buscar por significado)

import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import (
    DatetimeRange,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

# Cargamos las variables del archivo .env al inicio
load_dotenv()


class VectorDB:
    def __init__(self):
        # Configuración desde variables de entorno (.env)
        host = os.getenv("QDRANT_HOST")
        port = int(os.getenv("QDRANT_PORT"))
        self.collection_name = os.getenv("QDRANT_COLLECTION")

        local_embed_dir = os.getenv("LOCAL_EMBED_DIR")
        embed_model_id = os.getenv("EMBED_MODEL_ID", "st-5")

        if not local_embed_dir:
            raise ValueError("ERROR CRÍTICO: No se ha definido LOCAL_EMBED_DIR en el archivo .env")

        model_dir = Path(local_embed_dir).expanduser().resolve()
        self.model = self._load_or_download_embedding_model(model_dir=model_dir, model_id=embed_model_id)

        # Detectamos el tamaño del vector del modelo automáticamente
        self.vector_size = self.model.get_sentence_embedding_dimension()
        print(f"Modelo cargado. Dimensión de vectores: {self.vector_size}")

        # 3.- Conectamos con Qdrant
        self.client = QdrantClient(host=host, port=port)
        self._create_collection_if_not_exists()

    def _load_or_download_embedding_model(self, model_dir: Path, model_id: str) -> SentenceTransformer:
        model_dir.mkdir(parents=True, exist_ok=True)
        lock_path = model_dir.with_suffix(".lock")

        def _has_local_model() -> bool:
            return (model_dir / "config.json").exists() and any(model_dir.iterdir())

        for _ in range(120):
            try:
                fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
                break
            except FileExistsError:
                if _has_local_model():
                    print(f"Modelo de embeddings ya disponible en local: {model_dir}")
                    return SentenceTransformer(str(model_dir))
                time.sleep(1.0)
        else:
            raise TimeoutError(f"Timeout esperando lock de descarga del modelo: {lock_path}")

        try:
            if _has_local_model():
                print(f"Cargando modelo de embeddings local: {model_dir}")
                return SentenceTransformer(str(model_dir))

            print(f"Modelo local no encontrado. Descargando '{model_id}' y guardando en {model_dir} ...")
            downloaded = SentenceTransformer(model_id)
            downloaded.save(str(model_dir))
            print("Descarga completada y modelo persistido en disco.")
            return SentenceTransformer(str(model_dir))
        finally:
            try:
                os.unlink(lock_path)
            except FileNotFoundError:
                pass

    def _create_collection_if_not_exists(self):
        """Crea la colección en Qdrant si no existe todavía."""
        try:
            self.client.get_collection(collection_name=self.collection_name)
            print(f"Conectado a la colección: '{self.collection_name}'")
        except Exception:
            print(f"Creando colección '{self.collection_name}' con tamaño {self.vector_size}...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
            )
            print(f"Colección '{self.collection_name}' creada con éxito.")

    def upsert_chunks(self, texts: List[str], payloads: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.model:
            raise Exception("Modelo no cargado.")
        if not texts:
            return []

        print(f"Vectorizando {len(texts)} fragmentos...")
        embeddings = self.model.encode(texts)

        points = []
        chunk_metadata_list = []

        for i, (vector, payload) in enumerate(zip(embeddings, payloads)):
            point_id = str(uuid.uuid4())

            chunk_metadata_list.append({
                "chunk_id": point_id,
                "index": payload.get("chunk_index", i),
            })

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector.tolist(),
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)
        print(f"Subidos {len(texts)} chunks vectorizados a Qdrant.")

        return chunk_metadata_list

    def _build_filter(self, raw_filter: Optional[Dict[str, Any]]) -> Optional[Filter]:
        if not raw_filter or not isinstance(raw_filter, dict):
            return None

        must_conditions: List[FieldCondition] = []
        any_conditions: List[FieldCondition] = []

        for item in raw_filter.get("must", []) or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or "").strip()
            match_value = item.get("match")
            if key and match_value is not None:
                must_conditions.append(FieldCondition(key=key, match=MatchValue(value=match_value)))

        for item in raw_filter.get("must_any", []) or []:
            if not isinstance(item, dict):
                continue
            key = str(item.get("key") or "").strip()
            match_value = item.get("match")
            if key and match_value is not None:
                any_conditions.append(FieldCondition(key=key, match=MatchValue(value=match_value)))

        dt = raw_filter.get("datetime_range")
        if isinstance(dt, dict):
            dt_from = dt.get("from")
            dt_to = dt.get("to")
            if dt_from or dt_to:
                try:
                    must_conditions.append(
                        FieldCondition(
                            key="datetime",
                            range=DatetimeRange(gte=dt_from, lte=dt_to),
                        )
                    )
                except Exception:
                    must_conditions.append(
                        FieldCondition(
                            key="datetime",
                            range=Range(gte=dt_from, lte=dt_to),
                        )
                    )

        if not must_conditions and not any_conditions:
            return None

        q_filter = Filter(must=must_conditions)
        if any_conditions:
            q_filter.should = any_conditions
        return q_filter

    def search(self, query_text: str, limit: int = 5, qdrant_filter: Optional[Dict[str, Any]] = None):
        """Busca los fragmentos más parecidos a una pregunta con prefiltrado opcional."""
        if not self.model:
            return []

        query_vector = self.model.encode(query_text).tolist()
        query_filter = self._build_filter(qdrant_filter)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=max(1, int(limit)),
            query_filter=query_filter,
        )

        return response.points
