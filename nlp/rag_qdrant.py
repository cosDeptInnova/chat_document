import time
import os
import logging
from qdrant_client.http import models as qmodels

logger = logging.getLogger("cosmos_nlp_v3.qdrant")

def create_collection_if_not_exists(
    client,
    collection_name: str,
    dense_dim: int,
    *,
    dense_name: str = None,
    sparse_name: str = None,
    enable_sparse: bool = True,
    sparse_on_disk: bool = False,
    sparse_modifier: str = "",
):
    dense_name = dense_name or os.getenv("QDRANT_DENSE_NAME", "dense")
    sparse_name = sparse_name or os.getenv("QDRANT_SPARSE_NAME", "sparse")

    vectors_cfg = {
        dense_name: qmodels.VectorParams(size=dense_dim, distance=qmodels.Distance.COSINE)
    }

    sparse_cfg = None
    if enable_sparse:
        modifier = None
        if sparse_modifier == "idf":
            try:
                modifier = qmodels.Modifier.IDF
            except Exception:
                logger.warning("(qdrant) Modifier IDF no soportado; sigo sin modifier.")
        sparse_cfg = {
            sparse_name: qmodels.SparseVectorParams(
                index=qmodels.SparseIndexParams(on_disk=sparse_on_disk),
                modifier=modifier
            )
        }

    cols = client.get_collections()
    names = [c.name for c in cols.collections]
    if collection_name in names:
        logger.info(f"(qdrant) Colección '{collection_name}' ya existe; no se recrea.")
        return collection_name

    if sparse_cfg:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_cfg,
            sparse_vectors_config=sparse_cfg
        )
    else:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=vectors_cfg
        )
    logger.info(f"(qdrant) Colección creada: '{collection_name}' (dense={dense_dim}, sparse={enable_sparse})")
    return collection_name


def ensure_collection_compatible(client, collection_name: str, dense_dim: int) -> str:
    """
    Devuelve el nombre de colección que se debe usar.
    Puede ser el mismo o una versión nueva si hay migración.
    """
    dense_name   = os.getenv("QDRANT_DENSE_NAME", "dense")
    sparse_name  = os.getenv("QDRANT_SPARSE_NAME", "sparse")
    enable_sparse = os.getenv("QDRANT_ENABLE_SPARSE", "1") == "1"

    try:
        cols = client.get_collections()
        names = [c.name for c in cols.collections]
        if collection_name not in names:
            logger.info(f"(qdrant) Colección '{collection_name}' no existe. Creando…")
            return create_collection_if_not_exists(
                client,
                collection_name,
                dense_dim,
                dense_name=dense_name,
                sparse_name=sparse_name,
                enable_sparse=enable_sparse,
                sparse_on_disk=(os.getenv("QDRANT_SPARSE_ON_DISK", "0") == "1"),
                sparse_modifier=os.getenv("QDRANT_SPARSE_MODIFIER", "").lower(),
            )

        info = client.get_collection(collection_name)
        params = getattr(getattr(info, "config", None), "params", None)

        current_dense_dim = None
        vectors = getattr(params, "vectors", None)
        if isinstance(vectors, qmodels.VectorParams):
            current_dense_dim = int(vectors.size)
        elif isinstance(vectors, dict) and dense_name in vectors:
            try:
                current_dense_dim = int(vectors[dense_name].size)
            except Exception:
                current_dense_dim = None

        has_sparse = False
        sparse_cfg = getattr(params, "sparse_vectors", None)
        if isinstance(sparse_cfg, dict):
            has_sparse = sparse_name in sparse_cfg
        elif sparse_cfg is not None:
            try:
                has_sparse = bool(sparse_cfg.get(sparse_name))
            except Exception:
                has_sparse = False

        need_migration = (
            current_dense_dim is None or current_dense_dim != dense_dim or
            (enable_sparse and not has_sparse) or
            (not enable_sparse and has_sparse)
        )

        if not need_migration:
            logger.info("(qdrant) Colección compatible. Sin cambios.")
            return collection_name

        suffix = f"v{dense_dim}{'_sp' if enable_sparse else ''}_{int(time.time())}"
        new_name = f"{collection_name}__{suffix}"
        logger.warning(f"(qdrant) Migración no destructiva: creando '{new_name}'.")
        create_collection_if_not_exists(
            client,
            new_name,
            dense_dim,
            dense_name=dense_name,
            sparse_name=sparse_name,
            enable_sparse=enable_sparse,
            sparse_on_disk=(os.getenv("QDRANT_SPARSE_ON_DISK", "0") == "1"),
            sparse_modifier=os.getenv("QDRANT_SPARSE_MODIFIER", "").lower(),
        )
        return new_name

    except Exception as e:
        logger.warning(f"(qdrant) ensure_collection_compatible error: {e}; intento crear colección nueva.")
        return create_collection_if_not_exists(
            client,
            collection_name,
            dense_dim,
            dense_name=dense_name,
            sparse_name=sparse_name,
            enable_sparse=enable_sparse,
            sparse_on_disk=(os.getenv("QDRANT_SPARSE_ON_DISK", "0") == "1"),
            sparse_modifier=os.getenv("QDRANT_SPARSE_MODIFIER", "").lower(),
        )
