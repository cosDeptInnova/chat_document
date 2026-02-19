# legal_crew_src/agents.py

from crewai import Agent, LLM  # type: ignore[import]

def create_legal_planner_agent(llm: LLM, verbose: bool = False) -> Agent:
    return Agent(
        role="Planificador de investigación legal",
        goal=(
            "Convertir la petición en un plan de búsqueda legal: "
            "jurisdicción, términos clave, fuentes prioritarias y 1 a 4 consultas atómicas."
        ),
        backstory=(
            "Eres un investigador legal senior. "
            "Estructuras la búsqueda por jurisdicciones, vigencia (normativa actual) "
            "y jerarquía de fuentes (BOE/Diarios oficiales, EUR-Lex, reguladores, tribunales). "
            "Tu objetivo es maximizar precisión con el mínimo ruido."
        ),
        llm=llm,
        tools=[],  # tools se ejecutan SOLO desde el orquestador (control MCP)
        allow_delegation=False,
        verbose=verbose,
    )

def create_legal_analyst_agent(llm: LLM, verbose: bool = False) -> Agent:
    return Agent(
        role="Analista de evidencia legal",
        goal=(
            "Evaluar si las fuentes cubren el núcleo de la cuestión legal con evidencia suficiente "
            "y decidir si hacen falta búsquedas adicionales (más específicas)."
        ),
        backstory=(
            "Eres exigente con la evidencia: priorizas fuentes primarias, "
            "validas jurisdicción y vigencia, y evitas conclusiones no soportadas. "
            "Si falta una pieza clave, propones nuevas consultas concretas."
        ),
        llm=llm,
        tools=[],
        allow_delegation=False,
        verbose=verbose,
    )

def create_legal_writer_agent(llm: LLM, verbose: bool = False) -> Agent:
    return Agent(
        role="Redactor de memo legal",
        goal=(
            "Redactar una nota/memo legal claro y accionable para abogados corporativos, "
            "basado únicamente en fuentes, con citas [n] y riesgos."
        ),
        backstory=(
            "Eres un abogado-redactor. Entregas un memo utilizable: "
            "resumen ejecutivo, cuestiones, análisis con citas, riesgos y próximos pasos. "
            "No inventas; si falta evidencia, lo declaras."
        ),
        llm=llm,
        tools=[],
        allow_delegation=False,
        verbose=verbose,
    )

def create_legal_reviewer_agent(llm: LLM, verbose: bool = False) -> Agent:
    return Agent(
        role="Revisor de precisión y citación legal",
        goal=(
            "Verificar que el memo final no contiene overclaiming, que las citas [n] son coherentes "
            "y que se respetan jurisdicción, vigencia y límites de evidencia."
        ),
        backstory=(
            "Eres un revisor técnico. Corriges afirmaciones no soportadas, "
            "matizas conclusiones, y mejoras trazabilidad (citas) sin volver el texto inútil."
        ),
        llm=llm,
        tools=[],
        allow_delegation=False,
        verbose=verbose,
    )
