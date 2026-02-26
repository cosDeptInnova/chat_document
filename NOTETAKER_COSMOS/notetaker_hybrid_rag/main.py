# Punto de entrada FastAPI para ingestión y consulta híbrida RAG + GraphRAG.

import os
from typing import Any, Dict, List, Optional

import uvicorn
from classes.graph_db import GraphDB
from classes.vector_db import VectorDB
from dotenv import load_dotenv
from fastapi import Body, FastAPI, HTTPException, Request, Response, status
from hybrid_crew_orchestrator import HybridRAGCrewOrchestrator
from pydantic import BaseModel, Field
from utils import extract_chunks_and_metadata

# Cargar variables de entorno
load_dotenv()

app = FastAPI(title="NoteTaker Hybrid RAG API")

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
    user_id: str = Field(..., min_length=1)
    limit: int = Field(default=5, ge=1, le=30)
    history: List[Dict[str, Any]] = Field(default_factory=list)


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
async def query_meetings(query_data: QueryRequest):
    """Consulta híbrida con reescritura de query y filtros para Qdrant + explicación final."""
    try:
        user_query = query_data.query.strip()
        if not user_query:
            raise HTTPException(status_code=400, detail="Debes enviar una consulta no vacía")

        plan = hybrid_crew.plan_query(
            user_query=user_query,
            user_id=query_data.user_id,
            history=query_data.history,
        )

        normalized_query = str(plan.get("normalized_query") or user_query).strip()
        qdrant_filter = plan.get("qdrant_filter") or {"must": [{"key": "user_id", "match": query_data.user_id}]}

        retrieval_hints = plan.get("retrieval_hints") or {}
        limit = int(retrieval_hints.get("limit") or query_data.limit)
        limit = max(1, min(30, limit))

        print(f"Buscando respuesta para user={query_data.user_id}: '{normalized_query}'")

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

        context_package = []
        for hit in vector_results:
            chunk_id = hit.id
            score = hit.score
            payload = hit.payload or {}

            graph_context = neo_client.get_chunk_context(chunk_id)
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
