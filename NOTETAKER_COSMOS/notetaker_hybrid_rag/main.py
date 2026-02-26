# Aquí usaremos FastAPI, que será el punto de entrada (Recibirrá el JSON)

import os
from dotenv import load_dotenv
from fastapi import FastAPI, Body, HTTPException, Response, status
import uvicorn
from classes.graph_db import GraphDB
from classes.vector_db import VectorDB
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
except Exception as e:
    print(f"Error crítico al iniciar bases de datos: {e}")
    raise e

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Servidor activo y conectado a DBs"}

@app.post("/ingest")
async def ingest_meeting(data: dict = Body(...), response: Response = None):
    try:
        # 1. Identificador único
        meeting_id = data.get("meeting_id") or data.get("charts_id")
        
        if not meeting_id:
            raise HTTPException(status_code=400, detail="Faltan datos: meeting_id o charts_id")

        # 2. CONTROL DE DUPLICADOS (Idempotencia)
        if neo_client.check_meeting_exists(meeting_id):
            print(f"[AVISO] La reunión {meeting_id} ya fue indexada. Abortando.")
            # Cambiamos el status code a 409 (Conflict) para que el que llama sepa que ya existe
            response.status_code = status.HTTP_409_CONFLICT
            return {
                "status": "skipped",
                "message": f"La reunión {meeting_id} ya existe en el sistema."
            }

        print(f"Procesando NUEVA reunión: {meeting_id}...")

        # 3. Extraer textos y metadatos con el nuevo formato TOON
        texts_to_vectorize, payloads = extract_chunks_and_metadata(data)
        
        if not texts_to_vectorize:
            raise HTTPException(status_code=400, detail="No se encontraron chunks ('toon') para vectorizar.")

        # 4. Guardar vectores reales y metadata en Qdrant
        chunk_metadata_ids = qdrant_client.upsert_chunks(texts_to_vectorize, payloads)

        # 5. Guardar grafo en Neo4j
        # Extraemos el bloque 'normalized' para sacar a los participantes
        normalized_data = data.get("normalized", [])
        neo_client.create_meeting_node(meeting_id, normalized_data)
        
        # Temas: Si el final_handoff no tiene topics, intentamos cogerlos de 'insights' (por robustez)
        temas = data.get("final_handoff", {}).get("topics", []) or data.get("insights", {}).get("topics", [])
        neo_client.link_topic_to_meeting(meeting_id, temas)
        
        # 6. Vincular Grafo con Vectores
        neo_client.create_chunk_nodes(meeting_id, chunk_metadata_ids)

        return {
            "status": "success",
            "message": f"Reunión {meeting_id} procesada e indexada correctamente.",
            "storage_report": {
                "qdrant_chunks": len(texts_to_vectorize),
                "neo4j_topics": len(temas)
            }
        }

    except Exception as e:
        print(f"ERROR PROCESANDO REUNIÓN: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en el pipeline híbrido: {str(e)}")
    
# CONSULTA (Devuelve la información conjunta de Qdrant y de Neo4j para que el LLM procese todo y dé una respuesta) ---
@app.post("/query")
async def query_meetings(query_data: dict = Body(...)):
    """
    Endpoint que recibe una pregunta, busca en vectores (Qdrant) 
    y enriquece con contexto del grafo (Neo4j).
    """
    try:
        pregunta = query_data.get("query")
        limit = query_data.get("limit", 5) # Cuántos trozos recuperar por defecto
        
        if not pregunta:
            raise HTTPException(status_code=400, detail="Faltan datos: Debes enviar un campo 'query'")

        print(f"Buscando respuesta para: '{pregunta}'")

        # 1. Búsqueda Semántica en Qdrant
        # Esto nos devuelve los fragmentos de texto más parecidos a la pregunta
        vector_results = qdrant_client.search(query_text=pregunta, limit=limit)
        
        if not vector_results:
            return {
                "status": "success",
                "message": "No se encontró información relevante.",
                "context_package": []
            }

        # 2. Enriquecimiento con Neo4j (GraphRAG)
        context_package = []
        
        for hit in vector_results:
            # hit.id es el UUID de Qdrant, hit.payload tiene la metadata
            chunk_id = hit.id
            score = hit.score # Qué tan parecido es (0.0 a 1.0)
            payload = hit.payload
            
            # Buscamos el contexto general en el grafo
            graph_context = neo_client.get_chunk_context(chunk_id)
            
            # Construimos el "paquete de conocimiento" para el LLM
            paquete = {
                "similarity_score": round(score, 4),
                "text_content": payload.get("chunk_text", ""),
                # Añadimos los datos de Qdrant
                "metadata": {
                    "meeting_id": payload.get("meeting_id"),
                    "date": payload.get("datetime"),
                    "decisions_made": payload.get("decisions"),
                    "atmosphere": payload.get("atmosphere", {}).get("labels", [])
                },
                # Añadimos el contexto global de Neo4j
                "graph_context": graph_context if graph_context else "Contexto global no encontrado"
            }
            context_package.append(paquete)

        # 3. Respuesta a la Plataforma de IA
        return {
            "status": "success",
            "query_received": pregunta,
            "results_found": len(context_package),
            "context_package": context_package
        }

    except Exception as e:
        print(f"ERROR EN CONSULTA: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en la búsqueda híbrida: {str(e)}")
    
# Cerrar la conexión de Neo4j al apagar la aplicación
@app.on_event("shutdown")
def shutdown_event():
    if 'neo_client' in globals():
        neo_client.close()
        print("Conexión a Neo4j cerrada.")

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8001))
    
    print(f"Arrancando servidor en http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)