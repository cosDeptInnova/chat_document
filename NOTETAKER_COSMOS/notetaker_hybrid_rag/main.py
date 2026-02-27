# Punto de entrada FastAPI para ingestión y consulta híbrida RAG + GraphRAG.

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
from classes.graph_db import GraphDB
from classes.vector_db import VectorDB
from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, HTTPException, Request, Response, status
from jose import JWTError, jwt
from hybrid_crew_orchestrator import HybridRAGCrewOrchestrator
from pydantic import BaseModel, Field
from utils import extract_chunks_and_metadata

# Cargar variables de entorno
load_dotenv()

app = FastAPI(title="NoteTaker Hybrid RAG API")
logger = logging.getLogger("notetaker_hybrid_rag")

# INICIALIZACIÓN DE BASES DE DATOS
try:
    print("Iniciando conexión a Neo4j...")
    neo_client = GraphDB()

    print("Iniciando conexión a Qdrant y cargando modelos...")
    qdrant_client = VectorDB()

    print("Iniciando crew de análisis híbrido...")
    hybrid_crew = HybridRAGCrewOrchestrator()
except Exception as e:
    print(f"Error crítico al iniciar bases de datos o crew: {e}")
    raise e


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    user_id: Optional[str] = Field(default=None)
    limit: int = Field(default=5, ge=1, le=30)
    history: List[Dict[str, Any]] = Field(default_factory=list)


def _resolve_jwt_config() -> Tuple[List[str], List[str]]:
    """
    Obtiene claves/algoritmos JWT con tolerancia a distintos nombres de variables.
    Permite convivir con despliegues legacy (SECRET_KEY) y SSO (COSMOS_SECRET_KEY).
    """
    secret_candidates = [
        os.getenv("SECRET_KEY"),
        os.getenv("COSMOS_SECRET_KEY"),
        os.getenv("JWT_SECRET_KEY"),
        os.getenv("AUTH_SECRET_KEY"),
    ]
    secrets = [s.strip() for s in secret_candidates if isinstance(s, str) and s.strip()]

    algorithm_candidates = [
        os.getenv("ALGORITHM"),
        os.getenv("COSMOS_JWT_ALG"),
        os.getenv("JWT_ALGORITHM"),
    ]
    algorithms = [a.strip() for a in algorithm_candidates if isinstance(a, str) and a.strip()]
    if not algorithms:
        algorithms = ["HS256"]

    return secrets, algorithms


def verify_token(request: Request) -> Dict[str, Any]:
    access_token = request.cookies.get("access_token")
    token_str = None

    if access_token:
        token_str = access_token.replace("Bearer ", "").strip()
    else:
        auth = request.headers.get("Authorization")
        if auth and auth.lower().startswith("bearer "):
            token_str = auth[7:].strip()

    if not token_str:
        raise HTTPException(status_code=403, detail="No autorizado: token ausente.")

    secrets, algorithms = _resolve_jwt_config()
    if not secrets:
        raise HTTPException(
            status_code=500,
            detail=(
                "Configuración inválida: no hay clave JWT. "
                "Define SECRET_KEY, COSMOS_SECRET_KEY o JWT_SECRET_KEY."
            ),
        )

    for secret in secrets:
        try:
            payload = jwt.decode(token_str, secret, algorithms=algorithms)
            user_id = payload.get("user_id")
            if user_id is None:
                raise HTTPException(status_code=403, detail="Token sin user_id.")
            return payload
        except JWTError:
            continue

    raise HTTPException(status_code=403, detail="Token no válido o expirado.")


def _normalize_identity(value: Optional[str]) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "").strip().lower())
    return cleaned


def _authorized_participant_names(token: Dict[str, Any]) -> List[str]:
    names: set[str] = set()
    for key in ("full_name", "name", "username", "preferred_username", "email"):
        raw = token.get(key)
        if isinstance(raw, str) and raw.strip():
            names.add(_normalize_identity(raw))
            if key == "email" and "@" in raw:
                names.add(_normalize_identity(raw.split("@", 1)[0]))
    return [n for n in names if n]


def _can_access_chunk(token: Dict[str, Any], token_names: List[str], payload: Dict[str, Any], graph_context: Dict[str, Any]) -> bool:
    # 1) Control principal por owner del documento/reunión (más robusto para producción)
    token_user_id = str(token.get("user_id") or "").strip()
    payload_user_id = str(payload.get("user_id") or "").strip()
    if token_user_id and payload_user_id and token_user_id == payload_user_id:
        return True

    # 2) Filtro por identidad de participante/invitado para reuniones compartidas
    if not token_names:
        return False

    participant_candidates: List[str] = []
    for key in ("participants", "invited_participants"):
        values = payload.get(key, [])
        if isinstance(values, list):
            participant_candidates.extend([str(v) for v in values if isinstance(v, str)])

    graph_participants = graph_context.get("participants", []) if isinstance(graph_context, dict) else []
    if isinstance(graph_participants, list):
        participant_candidates.extend([str(v) for v in graph_participants if isinstance(v, str)])

    normalized_participants = {_normalize_identity(name) for name in participant_candidates if str(name).strip()}
    return any(name in normalized_participants for name in token_names)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Servidor activo y conectado a DBs"}


@app.post("/ingest")
async def ingest_meeting(
    request: Request,
    data: dict = Body(...),
    response: Response = None,
):
    request_id = request.headers.get("x-request-id") or "n/a"
    meeting_id = data.get("meeting_id") or data.get("charts_id")

    try:

        if not meeting_id:
            raise HTTPException(status_code=400, detail="Faltan datos: meeting_id o charts_id")

        if neo_client.check_meeting_exists(meeting_id):
            print(f"[INGEST][request_id={request_id}] [AVISO] La reunión {meeting_id} ya fue indexada. Abortando.")
            response.status_code = status.HTTP_409_CONFLICT
            return {
                "status": "skipped",
                "message": f"La reunión {meeting_id} ya existe en el sistema.",
            }

        print(f"[INGEST][request_id={request_id}] Procesando NUEVA reunión: {meeting_id}...")

        texts_to_vectorize, payloads = extract_chunks_and_metadata(data)

        if not texts_to_vectorize:
            raise HTTPException(status_code=400, detail="No se encontraron chunks ('toon') para vectorizar.")

        chunk_metadata_ids = qdrant_client.upsert_chunks(texts_to_vectorize, payloads)

        normalized_data = data.get("normalized", [])
        neo_client.create_meeting_node(meeting_id, normalized_data)

        temas = data.get("final_handoff", {}).get("topics", []) or data.get("insights", {}).get("topics", [])
        neo_client.link_topic_to_meeting(meeting_id, temas)

        neo_client.create_chunk_nodes(meeting_id, chunk_metadata_ids)

        return {
            "status": "success",
            "message": f"Reunión {meeting_id} procesada e indexada correctamente.",
            "storage_report": {"qdrant_chunks": len(texts_to_vectorize), "neo4j_topics": len(temas)},
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[INGEST][request_id={request_id}] ERROR PROCESANDO REUNIÓN: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en el pipeline híbrido: {str(e)}")


@app.post("/query")
async def query_meetings(query_data: QueryRequest, token: Dict[str, Any] = Depends(verify_token)):
    """Consulta híbrida con reescritura de query y filtros para Qdrant + explicación final."""
    try:
        user_query = query_data.query.strip()
        if not user_query:
            raise HTTPException(status_code=400, detail="Debes enviar una consulta no vacía")

        token_user_id = str(token.get("user_id"))
        if query_data.user_id and str(query_data.user_id) != token_user_id:
            raise HTTPException(status_code=403, detail="No puedes consultar con un user_id distinto al autenticado.")

        plan = hybrid_crew.plan_query(
            user_query=user_query,
            user_id=token_user_id,
            history=query_data.history,
        )

        normalized_query = str(plan.get("normalized_query") or user_query).strip()
        qdrant_filter = plan.get("qdrant_filter") or {}

        retrieval_hints = plan.get("retrieval_hints") or {}
        limit = int(retrieval_hints.get("limit") or query_data.limit)
        limit = max(1, min(30, limit))

        logger.info("Buscando respuesta para user=%s query='%s'", token_user_id, normalized_query)

        vector_results = qdrant_client.search(query_text=normalized_query, limit=limit, qdrant_filter=qdrant_filter)

        if not vector_results:
            return {
                "status": "success",
                "message": "No se encontró información relevante.",
                "query_plan": plan,
                "context_package": [],
                "assistant_response": {
                    "final_answer": "No encontré reuniones o fragmentos que cumplan los filtros solicitados.",
                    "evidence_summary": [],
                    "quality_assessment": {
                        "coverage": "baja",
                        "consistency": "media",
                        "temporal_alignment": "media",
                    },
                    "follow_up_questions": [
                        "¿Quieres ampliar el rango temporal o quitar algún filtro de reunión/tema?"
                    ],
                    "audit_trail": ["no_vector_results"],
                },
            }

        token_names = _authorized_participant_names(token)
        context_package = []
        for hit in vector_results:
            chunk_id = hit.id
            score = hit.score
            payload = hit.payload or {}

            graph_context = neo_client.get_chunk_context(chunk_id)
            if not _can_access_chunk(token, token_names, payload, graph_context or {}):
                continue

            paquete = {
                "similarity_score": round(float(score or 0.0), 4),
                "text_content": payload.get("chunk_text", ""),
                "metadata": {
                    "meeting_id": payload.get("meeting_id"),
                    "date": payload.get("datetime"),
                    "user_id": payload.get("user_id"),
                    "decisions_made": payload.get("decisions"),
                    "atmosphere": payload.get("atmosphere", {}).get("labels", []),
                },
                "graph_context": graph_context if graph_context else {"status": "sin_contexto_grafo"},
            }
            context_package.append(paquete)

        if not context_package:
            return {
                "status": "success",
                "message": "No se encontraron reuniones autorizadas para este usuario.",
                "query_plan": plan,
                "context_package": [],
                "assistant_response": {
                    "final_answer": "No encontré reuniones en las que figures como participante o invitado para responder esa consulta.",
                    "evidence_summary": [],
                    "quality_assessment": {
                        "coverage": "baja",
                        "consistency": "alta",
                        "temporal_alignment": "media",
                    },
                    "follow_up_questions": [
                        "¿Quieres reformular la pregunta indicando otra reunión en la que participaste?"
                    ],
                    "audit_trail": ["participant_access_denied_or_empty"],
                },
            }

        assistant_response = hybrid_crew.explain_results(
            user_query=user_query,
            normalized_query=normalized_query,
            filter_summary=qdrant_filter,
            combined_results=context_package,
        )

        return {
            "status": "success",
            "query_received": user_query,
            "normalized_query": normalized_query,
            "results_found": len(context_package),
            "query_plan": plan,
            "context_package": context_package,
            "assistant_response": assistant_response,
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"ERROR EN CONSULTA: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en la búsqueda híbrida: {str(e)}")


@app.on_event("shutdown")
def shutdown_event():
    if "neo_client" in globals():
        neo_client.close()
        print("Conexión a Neo4j cerrada.")


if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8001))

    print(f"Arrancando servidor en http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)
