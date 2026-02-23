# app/core/logging_config.py

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Dict

from .config import settings


class JsonFormatter(logging.Formatter):
    """
    Formateador de logs en JSON, apto para ingestión por SIEM / ELK / etc.

    Incluye:
    - timestamp
    - nivel
    - logger
    - servicio
    - mensaje
    - módulo / fichero / línea
    - campos extra típicos: request_id, user_id, conversation_id, etc.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_record: Dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "service": getattr(settings, "SERVICE_NAME", "cosmos-service"),
            "message": record.getMessage(),
            "module": record.module,
            "filename": record.filename,
            "lineno": record.lineno,
        }

        # Campos extra útiles para trazabilidad de peticiones / auditoría
        extra_keys = (
            "request_id",
            "user_id",
            "conversation_id",
            "conversation_uuid",
            "client_ip",
            "method",
            "path",
            "endpoint",
        )
        for key in extra_keys:
            value = getattr(record, key, None)
            if value is not None:
                log_record[key] = value

        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(log_record, ensure_ascii=False)


def _get_log_level() -> int:
    """
    Obtiene el nivel de log a partir de settings.LOG_LEVEL (por defecto INFO).
    """
    level_name = getattr(settings, "LOG_LEVEL", "INFO")
    if not isinstance(level_name, str):
        return logging.INFO

    level = getattr(logging, level_name.upper(), logging.INFO)
    if not isinstance(level, int):
        return logging.INFO
    return level


def _get_log_format() -> str:
    """
    Devuelve el formato de log deseado:

    - "json"  → logs estructurados en JSON.
    - "plain" → logs de texto plano (más sencillos de leer en local).

    Se lee de settings.LOG_FORMAT, por defecto "plain".
    """
    fmt = getattr(settings, "LOG_FORMAT", "plain")
    if not isinstance(fmt, str):
        return "plain"
    fmt = fmt.lower().strip()
    return fmt if fmt in {"json", "plain"} else "plain"


def configure_logging(service_name: str | None = None) -> None:
    """
    Configura logging básico para el microservicio COSMOS.

    Características:
    - Usa nivel de log desde settings.LOG_LEVEL (INFO por defecto).
    - Permite elegir formato JSON / texto plano vía settings.LOG_FORMAT.
    - Unifica los logs de uvicorn/fastapi con el logger raíz.
    - Diseñado para entornos empresariales (auditoría / SIEM / ISO 27001).

    Parámetros
    ----------
    service_name : str | None
        Nombre del servicio a registrar en los logs (por ejemplo "cosmos-mcp",
        "cosmos-web", "cosmos-ocr"). Si no se indica, se intentará usar
        settings.SERVICE_NAME, y si no existe, "cosmos-service".
    """
    # Evitar reconfigurar múltiples veces si se llama desde varios módulos
    if getattr(configure_logging, "_configured", False):
        return

    lvl = _get_log_level()
    log_format = _get_log_format()
    svc_name = service_name or getattr(settings, "SERVICE_NAME", "cosmos-service")

    root_logger = logging.getLogger()
    root_logger.setLevel(lvl)

    # Limpiar handlers previos (incluidos los que pueda añadir uvicorn por defecto)
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)

    if log_format == "json":
        formatter: logging.Formatter = JsonFormatter()
    else:
        # Formato plano clásico, pero incluyendo el nombre del servicio
        fmt_str = (
            "%(asctime)s [%(levelname)s] "
            f"[{svc_name}] "
            "%(name)s - %(message)s"
        )
        formatter = logging.Formatter(fmt=fmt_str)

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)

    # Hacer que los loggers de uvicorn/fastapi usen el logger raíz
    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        lg = logging.getLogger(logger_name)
        lg.setLevel(lvl)
        lg.handlers = []
        lg.propagate = True

    # Marcar como configurado para no duplicar handlers en llamadas posteriores
    configure_logging._configured = True  # type: ignore[attr-defined]

    # Log de arranque del servicio
    root_logger.info(
        "Logging configurado",
        extra={"service": svc_name, "log_format": log_format, "log_level": lvl},
    )
