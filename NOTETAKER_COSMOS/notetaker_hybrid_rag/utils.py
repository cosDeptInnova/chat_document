# Aquí irá la lógica de procesamiento (usaremos el JSON en forrmato toon)

import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime


def _collect_participants(json_data):
    """Intenta extraer participantes/invitados con tolerancia a formatos heterogéneos."""
    out = set()

    def _add_name(value):
        if isinstance(value, str) and value.strip():
            out.add(value.strip())

    # normalized: [{speaker, ...}, ...]
    for row in json_data.get("normalized", []) or []:
        _add_name((row or {}).get("speaker"))

    # insights/final_handoff suelen incluir estructuras de participantes
    for section in (json_data.get("final_handoff", {}), json_data.get("insights", {})):
        if not isinstance(section, dict):
            continue
        for key in ("participants", "attendees", "invitees", "invited_participants"):
            vals = section.get(key, [])
            if not isinstance(vals, list):
                continue
            for item in vals:
                if isinstance(item, str):
                    _add_name(item)
                elif isinstance(item, dict):
                    _add_name(item.get("name"))
                    _add_name(item.get("display_name"))

    return sorted(out)

# Cargamos las variables de entorno para configuración
load_dotenv()

def extract_chunks_and_metadata(json_data):
    """
    Extrae los textos, aplica el chunking (1000 tokens / 200 solape) 
    y construye la metadata requerida para Qdrant con fallback a insights.
    """
    # 1. Variables de la raíz (Si no vienen, ponemos "unknown" por seguridad)
    meeting_id = json_data.get("meeting_id") or json_data.get("charts_id", "unknown")
    doc_id = json_data.get("doc_id", "unknown")
    file_name = json_data.get("file_name", "unknown")
    user_id = json_data.get("user_id", "unknown")
    source_path = json_data.get("source_path", "unknown")
    
    # 2. Variables (Buscamos en final_handoff, y si está vacío, buscamos en insights)
    final_handoff = json_data.get("final_handoff", {})
    insights = json_data.get("insights", {})

    # Usamos 'or' para el fallback. 
    # Si final_handoff.get("topics") es [], pasará a leer insights.get("topics")
    topics = final_handoff.get("topics", []) or insights.get("topics", [])
    decisions = final_handoff.get("decisions", []) or insights.get("decisions", [])
    atmosphere = final_handoff.get("atmosphere", {}) or insights.get("atmosphere", {})
    decisiveness = final_handoff.get("decisiveness", {}) or insights.get("decisiveness", {})

    global_meta = {
        "meeting_id": meeting_id,
        "doc_id": doc_id,
        "file_name": file_name,
        "user_id": user_id,
        "source_path": source_path,
        "datetime": datetime.now().isoformat(),
        "topics": topics,
        "decisions": decisions,
        "atmosphere": atmosphere,
        "decisiveness": decisiveness
    }

    participants = _collect_participants(json_data)
    if participants:
        global_meta["participants"] = participants
        global_meta["invited_participants"] = participants

    # 3. Configuración del Chunker
    chunk_size = int(os.getenv("CHUNK_SIZE", 4000))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 800))
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )

    # 4. Extraer y trocear
    raw_chunks = json_data.get("chunks", [])
    
    texts_to_vectorize = []
    payloads_for_qdrant = []

    for i, chunk_data in enumerate(raw_chunks):
        texto_gigante = chunk_data.get("toon", "")
        
        if not texto_gigante:
            continue
            
        sub_chunks = text_splitter.split_text(texto_gigante)
        
        for j, sub_texto in enumerate(sub_chunks):
            texts_to_vectorize.append(sub_texto)
            
            chunk_payload = global_meta.copy()
            chunk_payload["parent_chunk_index"] = chunk_data.get("chunk_index", i)
            chunk_payload["sub_chunk_index"] = j 
            chunk_payload["chunk_text"] = sub_texto 
            
            payloads_for_qdrant.append(chunk_payload)

    return texts_to_vectorize, payloads_for_qdrant

def clean_transcript(normalized_data):
    """
    Genera un texto plano a partir de la variable 'normalized'.
    """
    full_text = ""
    current_participant = None

    for entry in normalized_data:
        name = entry.get("speaker", "Desconocido")
        text = entry.get("text", "")
        
        if not text: continue

        if name != current_participant:
            full_text += f"\n\n{name}: {text}"
            current_participant = name
        else:
            full_text += f" {text}"
    
    return full_text.strip()
