from crewai import Agent, LLM  # type: ignore[import]


def create_query_rewriter_agent(llm: LLM, verbose: bool = False) -> Agent:
    return Agent(
        role="Especialista en preparación de consultas para Qdrant",
        goal=(
            "Normalizar la consulta del usuario y generar un payload de filtros seguro, "
            "determinista y útil para prefiltrado en Qdrant antes de similitud coseno."
        ),
        backstory=(
            "Eres un ingeniero de búsqueda empresarial. Traducir lenguaje natural a filtros "
            "estructurados sin perder intención ni introducir supuestos no justificados."
        ),
        llm=llm,
        tools=[],
        allow_delegation=False,
        verbose=verbose,
    )


def create_result_evaluator_agent(llm: LLM, verbose: bool = False) -> Agent:
    return Agent(
        role="Evaluador de evidencia RAG/GraphRAG",
        goal=(
            "Valorar cobertura, consistencia temporal y calidad de los resultados combinados "
            "RAG + GraphRAG, y producir una respuesta final trazable para el usuario."
        ),
        backstory=(
            "Eres analista senior de conocimiento corporativo. Priorizas precisión, "
            "explicabilidad y transparencia de límites de evidencia."
        ),
        llm=llm,
        tools=[],
        allow_delegation=False,
        verbose=verbose,
    )
