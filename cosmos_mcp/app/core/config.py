# app/core/config.py

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Configuración de entorno para el microservicio Llama MCP."""

    PROJECT_NAME: str = "Cosmos Llama MCP"
    API_V1_PREFIX: str = "/api/v1"

    # URL base del servidor llama.cpp (OpenAI-compatible REAL)
    # 👉 Aquí debe apuntar al puerto donde corre llama.cpp, NO al MCP.
    # Ejemplo típico: http://127.0.0.1:8002/v1
    LLAMA_SERVER_BASE_URL: str = "http://127.0.0.1:8002/v1"

    # API key para llama.cpp (si tu servidor la exige; si no, déjalo en None)
    LLAMA_SERVER_API_KEY: str | None = None

    # Nombre de modelo tal y como lo espera llama.cpp
    # (ajusta este valor al "model" que acepte tu llama-server)
    LLAMA_MODEL: str = "Llama3_8B_Cosmos"

    # Timeout de las peticiones a llama.cpp (segundos)
    LLAMA_REQUEST_TIMEOUT: float = 120.0

    # --- Configuración del microservicio NLP / RAG usado por rag_search_tool ---
    # URL base del microservicio NLP que expone /search (RAG)
    NLP_APP_BASE_URL: str = "http://127.0.0.1:5000"
    # Timeout de las peticiones al servicio NLP (segundos)
    NLP_REQUEST_TIMEOUT: float = 180.0

    # --- Gateway MCP (expuesto por este microservicio) ---
    # Base pública donde otros micros (p. ej., web_search) llamarán a las tools MCP.
    # Si montas el gateway en FastAPI bajo /api/v1/mcp, la URL típica será:
    #   http://<host>:8090/api/v1/mcp
    MCP_GATEWAY_BASE_URL: str = Field(
        default="http://localhost:8090/api/v1/mcp",
        description="Base URL pública del Gateway MCP expuesto por cosmos_mcp",
    )
    # Token Bearer para proteger el Gateway MCP (debe coincidir con el que configures
    # en el cliente que consuma el gateway, p. ej. el microservicio web_search).
    MCP_GATEWAY_TOKEN: str = Field(
        default="",
        description="Token Bearer que protege el Gateway MCP",
    )
    # (Opcional) límites defensivos para el gateway
    MCP_GATEWAY_RATE_LIMIT_RPM: int = Field(
        default=60,
        description="Límite de peticiones por minuto aceptadas por el Gateway MCP",
    )

    # Logging
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,   # permite MCP_GATEWAY_TOKEN o mcp_gateway_token en .env
        extra="ignore",         # evita errores si hay variables extra en .env
    )


settings = Settings()
