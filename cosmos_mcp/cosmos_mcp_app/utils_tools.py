# cosmos_mcp_app/utils_tools.py

from __future__ import annotations

import os
import time
import json
import logging
from typing import Dict, Tuple, Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Config general de seguridad entre microservicios (tools -> NLP)
# ---------------------------------------------------------------------
#
# Modos soportados:
#   - none    : no añade auth interna (solo user access_token si viene)
#   - static  : envía un token estático por header (legacy)
#   - oauth2  : obtiene JWT vía client-credentials (recomendado)
#
# Variables:
#   NLP_S2S_AUTH_MODE=static|oauth2|none
#
# Legacy/static:
#   NLP_INTERNAL_TOKEN=<valor>
#   NLP_INTERNAL_TOKEN_HEADER=X-Internal-Token (o el que uses)
#
# OAuth2 client-credentials:
#   NLP_S2S_TOKEN_URL=<token endpoint>
#   NLP_S2S_CLIENT_ID=<client id>
#   NLP_S2S_CLIENT_SECRET=<client secret>
#
#   Opcionales:
#   NLP_S2S_SCOPE=<scope(s) separados por espacio>   (Azure v2 recomendado)
#   NLP_S2S_RESOURCE=<resource>                      (Azure v1 legacy)
#   NLP_S2S_AUDIENCE=<audience>                      (otros IdP)
#   NLP_S2S_EXTRA_FORM='{"param":"value"}'
#
# Header para mandar el JWT interno al NLP:
#   NLP_S2S_TOKEN_HEADER=Authorization
#   NLP_S2S_TOKEN_PREFIX=Bearer
#
# ---------------------------------------------------------------------

NLP_S2S_AUTH_MODE: str = os.getenv("NLP_S2S_AUTH_MODE", "static").strip().lower()

# Legacy/static
NLP_INTERNAL_TOKEN: str = os.getenv("NLP_INTERNAL_TOKEN", "").strip()
NLP_INTERNAL_TOKEN_HEADER: str = (
    os.getenv("NLP_INTERNAL_TOKEN_HEADER", "X-Internal-Token").strip()
    or "X-Internal-Token"
)

# OAuth2
NLP_S2S_TOKEN_URL: str = os.getenv("NLP_S2S_TOKEN_URL", "").strip()
NLP_S2S_CLIENT_ID: str = os.getenv("NLP_S2S_CLIENT_ID", "").strip()
NLP_S2S_CLIENT_SECRET: str = os.getenv("NLP_S2S_CLIENT_SECRET", "").strip()
NLP_S2S_SCOPE: str = os.getenv("NLP_S2S_SCOPE", "").strip()
NLP_S2S_RESOURCE: str = os.getenv("NLP_S2S_RESOURCE", "").strip()
NLP_S2S_AUDIENCE: str = os.getenv("NLP_S2S_AUDIENCE", "").strip()

NLP_S2S_EXTRA_FORM_RAW: str = os.getenv("NLP_S2S_EXTRA_FORM", "").strip()

# Header estándar recomendado para JWT interno hacia NLP
NLP_S2S_TOKEN_HEADER: str = os.getenv("NLP_S2S_TOKEN_HEADER", "Authorization").strip() or "Authorization"
NLP_S2S_TOKEN_PREFIX: str = os.getenv("NLP_S2S_TOKEN_PREFIX", "Bearer").strip() or "Bearer"

# Timeouts
try:
    NLP_S2S_TOKEN_TIMEOUT = float(os.getenv("NLP_S2S_TOKEN_TIMEOUT", "10"))
except Exception:
    NLP_S2S_TOKEN_TIMEOUT = 10.0


# ---------------------------------------------------------------------
# Cache simple en memoria del token S2S
# ---------------------------------------------------------------------

_service_token_cache: Dict[str, object] = {
    "access_token": None,
    "expires_at": 0.0,
    "mode": None,
}


def _now() -> float:
    return time.time()


def _normalize_bearer(token: str) -> str:
    t = token.strip()
    if not t:
        return ""
    if t.lower().startswith("bearer "):
        return t[7:].strip()
    return t


def _format_bearer(token: str) -> str:
    t = _normalize_bearer(token)
    if not t:
        return ""
    return f"{NLP_S2S_TOKEN_PREFIX} {t}".strip()


def _parse_extra_form() -> Dict[str, str]:
    if not NLP_S2S_EXTRA_FORM_RAW:
        return {}
    try:
        obj = json.loads(NLP_S2S_EXTRA_FORM_RAW)
        if isinstance(obj, dict):
            return {str(k): str(v) for k, v in obj.items()}
    except Exception:
        logger.warning("[S2S] NLP_S2S_EXTRA_FORM no es JSON válido.")
    return {}


async def _fetch_oauth2_client_credentials_token() -> Optional[Tuple[str, float]]:
    """
    Pide un token al IdP con client-credentials.

    Devuelve:
      (access_token, expires_at_epoch)
    """
    if not NLP_S2S_TOKEN_URL or not NLP_S2S_CLIENT_ID or not NLP_S2S_CLIENT_SECRET:
        logger.warning(
            "[S2S] Faltan variables para oauth2: "
            "NLP_S2S_TOKEN_URL / NLP_S2S_CLIENT_ID / NLP_S2S_CLIENT_SECRET"
        )
        return None

    form: Dict[str, str] = {
        "grant_type": "client_credentials",
        "client_id": NLP_S2S_CLIENT_ID,
        "client_secret": NLP_S2S_CLIENT_SECRET,
    }

    # Azure v2 recomendado: scope
    if NLP_S2S_SCOPE:
        form["scope"] = NLP_S2S_SCOPE

    # Azure v1 legacy: resource
    if NLP_S2S_RESOURCE and "scope" not in form:
        form["resource"] = NLP_S2S_RESOURCE

    # Otros IdP: audience custom
    if NLP_S2S_AUDIENCE:
        # algunos IdP aceptan "audience", otros "resource"
        # no lo forzamos si ya hay resource definido
        if "resource" not in form:
            form["audience"] = NLP_S2S_AUDIENCE

    # extras opcionales
    form.update(_parse_extra_form())

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    try:
        async with httpx.AsyncClient(timeout=NLP_S2S_TOKEN_TIMEOUT) as client:
            resp = await client.post(NLP_S2S_TOKEN_URL, data=form, headers=headers)
        resp.raise_for_status()
        data = resp.json() or {}
    except Exception as e:
        logger.warning("[S2S] Error obteniendo token oauth2: %s", e)
        return None

    token = data.get("access_token")
    if not isinstance(token, str) or not token.strip():
        logger.warning("[S2S] Respuesta oauth2 sin access_token.")
        return None

    # expiración
    expires_in = data.get("expires_in")
    try:
        expires_in_s = float(expires_in) if expires_in is not None else 300.0
    except Exception:
        expires_in_s = 300.0

    # margen de seguridad para refresco
    expires_at = _now() + max(30.0, expires_in_s - 20.0)

    return token.strip(), expires_at


async def get_service_token() -> Optional[str]:
    """
    Devuelve el token interno S2S para tools -> NLP según el modo.

    - none   -> None
    - static -> NLP_INTERNAL_TOKEN (o NLP_S2S_STATIC_TOKEN si quieres extender)
    - oauth2 -> token client-credentials cacheado
    """
    mode = (NLP_S2S_AUTH_MODE or "static").lower().strip()

    if mode == "none":
        return None

    if mode == "static":
        if NLP_INTERNAL_TOKEN:
            return NLP_INTERNAL_TOKEN
        # compat si alguien usa esta variable alternativa
        alt = os.getenv("NLP_S2S_STATIC_TOKEN", "").strip()
        return alt or None

    if mode != "oauth2":
        logger.warning("[S2S] NLP_S2S_AUTH_MODE desconocido=%s. Usando static.", mode)
        if NLP_INTERNAL_TOKEN:
            return NLP_INTERNAL_TOKEN
        return None

    # oauth2 cache
    cached_token = _service_token_cache.get("access_token")
    expires_at = float(_service_token_cache.get("expires_at") or 0.0)

    if isinstance(cached_token, str) and cached_token and _now() < expires_at:
        return cached_token

    fetched = await _fetch_oauth2_client_credentials_token()
    if not fetched:
        # fallback defensivo a static si existiera
        if NLP_INTERNAL_TOKEN:
            logger.warning("[S2S] Fallback a NLP_INTERNAL_TOKEN por fallo oauth2.")
            return NLP_INTERNAL_TOKEN
        return None

    token, new_expires_at = fetched
    _service_token_cache["access_token"] = token
    _service_token_cache["expires_at"] = new_expires_at
    _service_token_cache["mode"] = "oauth2"

    return token


async def build_nlp_auth(
    *,
    access_token: Optional[str] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Construye cookies y headers para llamadas al microservicio NLP.

    Capas soportadas:
    1) access_token de usuario → cookie "access_token=Bearer <token>"
       (compatibilidad con endpoints actuales del NLP).
    2) S2S interno tools -> NLP:
       - static  -> header configurable (legacy)
       - oauth2  -> Authorization: Bearer <service-jwt> (por defecto)
    """
    cookies: Dict[str, str] = {}
    headers: Dict[str, str] = {}

    # 1) Usuario -> NLP (cookie)
    if access_token:
        user_tok = _normalize_bearer(access_token)
        if user_tok:
            cookies["access_token"] = f"Bearer {user_tok}"

    # 2) Service-to-service -> NLP
    mode = (NLP_S2S_AUTH_MODE or "static").lower().strip()
    service_tok = await get_service_token()

    if service_tok:
        if mode == "static":
            # legacy header
            headers[NLP_INTERNAL_TOKEN_HEADER] = service_tok
        else:
            # oauth2 (o futuro) -> Authorization Bearer
            headers[NLP_S2S_TOKEN_HEADER] = _format_bearer(service_tok)

    # Accept explícito por robustez
    headers.setdefault("Accept", "application/json")

    return cookies, headers
