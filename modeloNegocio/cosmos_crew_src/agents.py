# cosmos_crewai/agents.py

import logging
from typing import Optional

from crewai import Agent, LLM

logger = logging.getLogger(__name__)


def _describe_llm(llm: LLM) -> str:
    """
    Devuelve una descripción segura del LLM para logs y auditoría.

    No asume atributos concretos (usa getattr defensivo) para no
    romper si CrewAI cambia la implementación interna.
    """
    try:
        model = getattr(llm, "model", None) or getattr(llm, "model_name", None)
    except Exception:
        model = None

    base_url = None
    try:
        base_url = getattr(llm, "base_url", None)
    except Exception:
        base_url = None

    if base_url is None:
        # Algunos wrappers llevan el cliente HTTP dentro
        try:
            client = getattr(llm, "client", None)
            if client is not None:
                base_url = getattr(client, "base_url", None)
        except Exception:
            base_url = None

    try:
        temperature = getattr(llm, "temperature", None)
    except Exception:
        temperature = None

    return f"LLM(model={model!r}, base_url={base_url!r}, temperature={temperature!r})"


def create_assistant_agent(
    llm: LLM,
    verbose: bool = False,
) -> Agent:
    """
    Crea el agente principal de COSMOS que genera la respuesta final al usuario.

    Este agente:
    - Recibe ya “cocinado” el contexto (historial, RAG, archivos en vuelo) desde el orquestador.
    - Aplica las instrucciones de sistema definidas en `build_assistant_system_instructions`.
    - NO llama directamente a herramientas externas; toda la orquestación (RAG, MCP, etc.)
      la hace `CosmosCrewOrchestrator`.

    Mantener:
    - allow_delegation=False para evitar que cree subtareas autónomas.
    """
    logger.info(
        "[CrewAgents] Creando agente 'Asistente principal de COSMOS' con %s",
        _describe_llm(llm),
    )

    agent = Agent(
        role="Asistente principal de COSMOS",
        goal=(
            "Responder a las preguntas del usuario combinando conversación previa, "
            "resultados de búsqueda RAG y archivos cargados en vuelo, manteniendo "
            "siempre la seguridad de datos y el contexto empresarial."
        ),
        backstory=(
            "Eres el asistente corporativo de la plataforma COSMOS. Conoces la "
            "estructura de colecciones (usuario, departamentos) y respetas siempre "
            "los permisos y la confidencialidad. Tu prioridad es ser riguroso, claro "
            "y transparente sobre de dónde salen los datos (RAG, historial, archivos)."
        ),
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )

    logger.debug(
        "[CrewAgents] Agente creado: role=%r, verbose=%s, allow_delegation=%s",
        agent.role,
        agent.verbose,
        agent.allow_delegation,
    )
    return agent


def create_planner_agent(
    llm: LLM,
    verbose: bool = False,
) -> Agent:
    """
    Crea el agente PLANIFICADOR de consultas.

    Responsabilidad:
    - NO responde al usuario final.
    - Analiza la pregunta + historial y devuelve SIEMPRE un JSON con:
      * normalized_question
      * intent (conteo, consulta, explicacion, otro)
      * needs_rag, needs_web
      * rag_query
      * filters (ubicacion, tipo_activo, empresa, ...)

    La validación de que el output es JSON la hace el orquestador (`_plan_query`),
    que loga y aplica fallback si algo viene malformado.
    """
    logger.info(
        "[CrewAgents] Creando agente 'Planificador de consultas COSMOS' con %s",
        _describe_llm(llm),
    )

    agent = Agent(
        role="Planificador de consultas COSMOS",
        goal=(
            "Analizar la pregunta del usuario y devolver un plan JSON con la intención, "
            "si necesita RAG o búsqueda en internet, filtros clave (ubicación, tipo de "
            "activo, empresa, etc.) y una versión normalizada de la pregunta. "
            "NUNCA debes responder directamente al usuario final."
        ),
        backstory=(
            "Eres un analista experto en entender consultas de usuarios corporativos en COSMOS. "
            "Lees el historial reciente y la nueva pregunta, y devuelves SIEMPRE un JSON válido "
            "con un plan de cómo resolverla, para que otros agentes ejecuten la búsqueda y la "
            "explicación. Eres estricto con el formato JSON y no añades texto extra."
        ),
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )

    logger.debug(
        "[CrewAgents] Agente planner creado: role=%r, verbose=%s",
        agent.role,
        agent.verbose,
    )
    return agent


def create_rag_analyst_agent(
    llm: LLM,
    verbose: bool = False,
) -> Agent:
    """
    Crea el agente analista de documentos internos (RAG), usado como 'TOON'.

    Responsabilidad:
    - Recibir los fragmentos RAG BRUTOS (excels, PDFs, contratos, etc.).
    - Destilar SOLO la información relevante para la pregunta.
    - Devolver un resumen estructurado y compacto (no una respuesta final).

    Este agente se usa en `_summarize_rag` cuando el contexto es grande,
    para ahorrar tokens y hacer la capa de conocimiento más auditable.
    """
    logger.info(
        "[CrewAgents] Creando agente 'Analista de documentos RAG COSMOS' con %s",
        _describe_llm(llm),
    )

    agent = Agent(
        role="Analista de documentos internos COSMOS (RAG)",
        goal=(
            "Leer los resultados RAG brutos (filas de Excel, texto de PDFs, "
            "contratos, correos, etc.) y extraer SOLO la información relevante "
            "para la pregunta actual, generando un resumen estructurado y compacto. "
            "NO debes responder al usuario final."
        ),
        backstory=(
            "Eres especialista en inventarios, contratos y documentación corporativa. "
            "Sabes leer filas de Excel, interpretar metadatos y encontrar el dato clave "
            "para responder preguntas de conteo, consulta o explicación. Tu salida va "
            "dirigida a otros agentes de COSMOS, no al usuario final."
        ),
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )

    logger.debug(
        "[CrewAgents] Agente RAG analyst creado: role=%r, verbose=%s",
        agent.role,
        agent.verbose,
    )
    return agent


def create_files_analyst_agent(
    llm: LLM,
    verbose: bool = False,
) -> Agent:
    """
    Crea el agente analista de archivos en vuelo (no indexados).

    Responsabilidad:
    - Leer el contenido de los archivos recién subidos (PDF, Word, Excel, emails, etc.).
    - Resumir solo los puntos relevantes para la pregunta actual.
    - NO responder al usuario final, solo producir contexto destilado.

    Se utiliza en `_summarize_ephemeral_files` para integrar archivos
    ad-hoc sin reventar el contexto del modelo.
    """
    logger.info(
        "[CrewAgents] Creando agente 'Analista de archivos en vuelo COSMOS' con %s",
        _describe_llm(llm),
    )

    agent = Agent(
        role="Analista de archivos en vuelo COSMOS",
        goal=(
            "Leer y resumir los archivos que el usuario acaba de subir, extrayendo "
            "los puntos clave relevantes para la pregunta actual. No debes generar "
            "charla general ni responder directamente al usuario final."
        ),
        backstory=(
            "Te especializas en PDFs, Word, correos y excels que el usuario sube puntualmente. "
            "Tu misión es destilar SOLO lo útil para la pregunta, de forma muy concisa y "
            "estructurada, para que el asistente principal pueda usarlo como contexto."
        ),
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )

    logger.debug(
        "[CrewAgents] Agente files analyst creado: role=%r, verbose=%s",
        agent.role,
        agent.verbose,
    )
    return agent

def create_rag_router_agent(
    llm: LLM,
    verbose: bool = False,
) -> Agent:
    """
    Agente Router que decide si se debe usar RAG para responder.

    Output ESTRICTO:
    { "use_rag": true|false, "reason": "..." }
    """
    try:
        model = getattr(llm, "model", None) or getattr(llm, "model_name", None)
        base_url = getattr(llm, "base_url", None)
        temperature = getattr(llm, "temperature", None)
        llm_desc = f"LLM(model={model!r}, base_url={base_url!r}, temperature={temperature!r})"
    except Exception:
        llm_desc = "LLM(unknown)"

    logger.info("[CrewAgents] Creando agente 'Router de uso de RAG COSMOS' con %s", llm_desc)

    return Agent(
        role="Router de uso de RAG COSMOS",
        goal=(
            "Decidir si para responder la pregunta actual hace falta consultar documentación interna "
            "indexada (RAG) o si puede resolverse con historial/adjuntos/contexto presente."
        ),
        backstory=(
            "Eres un router experto. Minimiza llamadas a RAG cuando no aporten valor. "
            "Evita RAG cuando la pregunta sea sobre un adjunto/subida reciente, "
            "salvo que se pida explícitamente documentación interna."
        ),
        verbose=verbose,
        allow_delegation=False,
        llm=llm,
    )

