from typing import Optional, Dict, Any
import json

def build_assistant_system_instructions(flow: Optional[str]) -> str:
    """
    Instrucciones de sistema para el asistente principal.

    Mejora clave:
    - Prioridad dinámica: si la pregunta menciona adjunto/archivo subido/este PDF,
      entonces la fuente prioritaria son los ARCHIVOS EN VUELO, no RAG.
    - Mantiene coherencia empresa/ubicación/tipo de activo.
    - Permite conteos fiables sobre inventarios en RAG o en adjuntos.
    """
    flow_norm = (flow or "").upper()

    if flow_norm == "R":
        flow_msg = (
            "Modo actual: R (respuesta rápida apoyada en el documento más relevante).\n"
            "Sé conciso y prioriza el fragmento principal.\n"
        )
    else:
        flow_msg = (
            "Modo actual: C (COSMOS inteligente).\n"
            "Combina varias fuentes cuando aporten valor.\n"
        )

    base = (
        "Eres el asistente COSMOS.\n"
        "\n"
        "=== Prioridad de fuentes (DINÁMICA) ===\n"
        "1) Si la pregunta del usuario hace referencia explícita a un adjunto o archivo recién subido\n"
        "   (por ejemplo: 'archivo adjunto', 'en el adjunto', 'este PDF', 'te lo acabo de subir', 'documento subido'),\n"
        "   ENTONCES prioriza los ARCHIVOS EN VUELO como fuente #1.\n"
        "2) Si NO hay referencia explícita a adjunto, entonces prioriza RAG (documentos internos indexados) como fuente #1.\n"
        "3) La segunda fuente será la que no haya sido #1 (RAG o archivos en vuelo).\n"
        "4) El historial de conversación solo se usa como apoyo cuando NO contradice la pregunta actual.\n"
        "\n"
        "Responde SIEMPRE en español, claro y estructurado (párrafos breves, listas cuando convenga).\n"
        "\n"
        "=== Coherencia entre pregunta y resultados ===\n"
        "- Antes de usar un fragmento RAG o un adjunto, valida coherencia con:\n"
        "  · EMPRESA mencionada\n"
        "  · UBICACIÓN mencionada\n"
        "  · TIPO de activo/equipo\n"
        "- Si los fragmentos hablan de otra empresa/ubicación/contexto, NO uses sus números ni detalles.\n"
        "\n"
        "=== Cómo interpretar resultados RAG y adjuntos ===\n"
        "- 'Resultados RAG' pueden incluir excels (filas con '|') o texto (PDF/Word/procedimientos/contratos).\n"
        "- Los archivos en vuelo pueden incluir texto extraído de PDFs/Word/Excel o OCR.\n"
        "- Para inventarios tabulares:\n"
        "  · Cada fila = 1 unidad salvo que exista un campo explícito de cantidad.\n"
        "\n"
        "=== Reglas para preguntas de conteo ===\n"
        "- Localiza filas/frases que cumplan filtros (empresa/ubicación/tipo).\n"
        "- Suma filas o cantidades explícitas.\n"
        "- Si no hay ninguna evidencia relevante, indícalo y explica qué faltaría.\n"
        "\n"
        "=== Estilo de respuesta ===\n"
        "- Empieza por la conclusión.\n"
        "- Explica brevemente en qué evidencia te basas (sin volcar texto completo).\n"
        "- No menciones etiquetas internas (p.ej. '[CHAT_ID ...]') ni detalles de orquestación.\n"
    )

    return flow_msg + base

def build_planner_prompt(
    flow: Optional[str],
    history_str: str,
    user_prompt: str,
    tools_block: Optional[str] = None,
) -> str:
    """
    Prompt del planner (JSON estricto).

    Mejora clave:
    - Needs_files se activa SIEMPRE ante señales claras de adjunto/subida reciente.
    - Needs_rag se prioriza cuando el usuario menciona un documento por nombre (X.pdf) sin indicar adjunto reciente,
      o cuando pide explícitamente 'documentos internos', 'inventario corporativo', etc.
    """
    flow_norm = (flow or "").upper()
    flow_hint = "El usuario está en modo C (COSMOS inteligente).\n" if flow_norm != "R" else (
        "El usuario está en modo R (respuesta rápida basada en un documento principal).\n"
    )

    if not history_str:
        history_str = "No hay historial previo relevante."

    if tools_block is None:
        tools_block = (
            "No se ha proporcionado una lista explícita de herramientas. "
            "Si consideras que para resolver la consulta hace falta RAG, archivos en vuelo "
            "o información externa, márcalo en needs_rag / needs_files / needs_web."
        )

    system_planner = (
        "Eres el PLANIFICADOR de consultas de la plataforma COSMOS.\n"
        "Tu única responsabilidad es analizar la pregunta del usuario y devolver un PLAN en formato JSON.\n"
        "NO expliques nada al usuario, NO respondas a la pregunta directamente.\n\n"
        f"{flow_hint}"
        "Tu salida DEBE ser SIEMPRE un JSON VÁLIDO con exactamente estas claves:\n\n"
        "{\n"
        '  "normalized_question": "...",\n'
        '  "intent": "conteo" | "consulta" | "explicacion" | "otro",\n'
        '  "needs_rag": true | false,\n'
        '  "needs_web": true | false,\n'
        '  "needs_files": true | false,\n'
        '  "needs_history": true | false,\n'
        '  "rag_query": "...",\n'
        '  "filters": {\n'
        '    "ubicacion": "..." | null,\n'
        '    "tipo_activo": "PORTATIL" | "IMPRESORA" | "SERVIDOR" | null,\n'
        '    "empresa": "..." | null,\n'
        '    "documento": "..." | null\n'
        "  }\n"
        "}\n\n"
        "Reglas de intent:\n"
        "- conteo: cantidades (¿cuántos, número de...).\n"
        "- consulta: dato concreto / validación / 'qué pone'.\n"
        "- explicacion: explicar un texto/norma.\n"
        "- otro: resto.\n\n"
        "Reglas para needs_files (ADJUNTOS EN VUELO):\n"
        "- needs_files=true SI el usuario indica explícitamente adjunto/subida reciente.\n"
        "  Señales: 'archivo adjunto', 'en el adjunto', 'te lo acabo de subir', 'este PDF', 'documento subido',\n"
        "  'archivo que he subido', 'en el archivo que adjunté'.\n"
        "- Si el usuario solo menciona un NOMBRE de archivo (ej: 'pliego_aena.pdf') pero NO dice que sea adjunto reciente,\n"
        "  NO asumas onFly: needs_files=false y resuélvelo por RAG.\n\n"
        "Reglas para needs_rag (DOCUMENTACIÓN INTERNA):\n"
        "- needs_rag=true cuando la respuesta dependa de documentación corporativa indexada.\n"
        "- Si el usuario menciona un documento por nombre (X.pdf, Y.xlsx...) y pide resumen/qué trata/qué pone,\n"
        "  entonces needs_rag=true y filters.documento debe contener ese nombre.\n"
        "- Si el usuario pregunta por un adjunto ('archivo adjunto') sin nombre, por defecto:\n"
        "  needs_files=true y needs_rag=false (salvo que también pida explícitamente buscar en documentación interna).\n\n"
        "needs_history:\n"
        "- true si la pregunta depende de contexto previo ('y allí?', 'como antes', 'lo mismo que dijiste').\n\n"
        "filters.documento:\n"
        "- Si aparece un nombre de archivo con extensión, ponlo aquí.\n"
        "- Si no hay nombre, null.\n\n"
        "IMPORTANTE:\n"
        "- NO añadas texto fuera del JSON.\n"
        "- NO incluyas fences ```.\n"
    )

    return (
        f"{system_planner}\n\n"
        f"=== Historial reciente ===\n{history_str}\n\n"
        f"=== Herramientas disponibles en el ecosistema COSMOS ===\n{tools_block}\n\n"
        f"=== Pregunta actual del usuario ===\n{user_prompt}\n\n"
        "Devuelve ÚNICAMENTE el JSON con el plan.\n"
    )

def build_rag_router_prompt(
    flow: Optional[str],
    history_str: str,
    user_prompt: str,
    planner_plan: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Router IA para decidir si usar RAG.

    Mejora clave:
    - Sesgo fuerte a NO usar RAG cuando la pregunta es sobre un adjunto/subida reciente.
    - Solo activa RAG si el usuario pide explícitamente documentación interna o si la pregunta no es resoluble con adjuntos/historial.
    """
    flow_norm = (flow or "C").upper()
    if not history_str:
        history_str = "No hay historial previo relevante."

    plan_dump = "No disponible."
    if planner_plan:
        try:
            plan_dump = json.dumps(planner_plan, ensure_ascii=False, indent=2)
        except Exception:
            plan_dump = str(planner_plan)

    return (
        "Eres el ROUTER de uso de RAG de COSMOS.\n"
        "Tu única tarea es decidir si para responder la pregunta actual hace falta consultar documentación interna indexada (RAG).\n"
        "NO respondas al usuario final. Devuelve SOLO un JSON válido y NADA MÁS.\n\n"
        "Formato de salida obligatorio:\n"
        "{\n"
        '  "use_rag": true | false,\n'
        '  "reason": "breve explicación"\n'
        "}\n\n"
        "Criterio principal:\n"
        "- Si la pregunta menciona adjunto/subida reciente ('archivo adjunto', 'este PDF', 'te lo acabo de subir'),\n"
        "  entonces use_rag=false POR DEFECTO.\n"
        "- Solo pon use_rag=true en ese caso si el usuario además pide explícitamente buscar en documentación interna\n"
        "  o comparar con políticas/inventarios corporativos.\n"
        "- Si no hay referencia a adjunto, use_rag=true cuando dependa de datos internos.\n\n"
        "Reglas:\n"
        "- Sé conservador: si depende de documentación interna no aportada, use_rag=true.\n"
        "- No incluyas texto fuera del JSON.\n\n"
        f"Flow: {flow_norm}\n\n"
        "=== Historial reciente ===\n"
        f"{history_str}\n\n"
        "=== Plan del planner (referencia interna) ===\n"
        f"{plan_dump}\n\n"
        "=== Pregunta actual del usuario ===\n"
        f"{user_prompt}\n\n"
        "Devuelve ÚNICAMENTE el JSON.\n"
    )
