# cosmos_mcp_app/user_jwt.py

from __future__ import annotations

import os
import time
import logging
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Header donde llega el JWT del usuario (lo envía tu MCPClient)
USER_AUTH_HEADER = os.getenv("MCP_USER_AUTH_HEADER", "X-User-Authorization").strip() or "X-User-Authorization"

# Si es 1, exige JWT de usuario para llamar tools (NO recomendado al inicio)
MCP_REQUIRE_USER_JWT = os.getenv("MCP_REQUIRE_USER_JWT", "0").strip() == "1"

# Algoritmos permitidos (por defecto HS256,RS256)
MCP_USER_JWT_ALGOS = [a.strip() for a in os.getenv("MCP_USER_JWT_ALGOS", "HS256,RS256").split(",") if a.strip()]

# Para HS256
MCP_USER_JWT_SECRET = os.getenv("MCP_USER_JWT_SECRET", "").strip()

# Para RS256 (PEM)
MCP_USER_JWT_PUBLIC_KEY = os.getenv("MCP_USER_JWT_PUBLIC_KEY", "").strip()

# Opcionales
MCP_USER_JWT_AUDIENCE = os.getenv("MCP_USER_JWT_AUDIENCE", "").strip() or None
MCP_USER_JWT_ISSUER = os.getenv("MCP_USER_JWT_ISSUER", "").strip() or None

# Claim que usas como user_id (puede ser "sub", "user_id", etc.)
MCP_USER_ID_CLAIM = os.getenv("MCP_USER_ID_CLAIM", "user_id").strip() or "user_id"


def _normalize_bearer(token: str) -> str:
    t = (token or "").strip()
    if not t:
        return ""
    if t.lower().startswith("bearer "):
        t = t[7:].strip()
    return t


def _jwt_verify_key_and_alg() -> Tuple[Optional[str], Optional[str]]:
    """
    Devuelve (key, mode) donde mode ∈ {"hs","rs"}.
    Si no hay key, devuelve (None,None).
    """
    # Prefer RS si hay public key
    if MCP_USER_JWT_PUBLIC_KEY:
        return MCP_USER_JWT_PUBLIC_KEY, "rs"
    if MCP_USER_JWT_SECRET:
        return MCP_USER_JWT_SECRET, "hs"
    return None, None


def _safe_int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def decode_and_validate_user_jwt(token: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Valida y decodifica JWT.
    Devuelve (claims, error). Si error != None, claims será None.

    - No lanza excepciones (para middleware robusto).
    - Valida exp/nbf/iat cuando existan.
    - Valida iss/aud si están configurados.
    """
    tok = _normalize_bearer(token)
    if not tok:
        return None, "missing_token"

    key, mode = _jwt_verify_key_and_alg()
    if not key or not mode:
        # No hay config JWT: no podemos validar
        return None, "jwt_not_configured"

    try:
        import jwt  # pyjwt
    except Exception:
        return None, "pyjwt_not_installed"

    options = {
        "verify_signature": True,
        "verify_exp": True,
        "verify_nbf": True,
        "verify_iat": False,  # iat no siempre fiable
        "require": [],        # no exigimos claims por defecto
    }

    try:
        kwargs: Dict[str, Any] = {
            "algorithms": MCP_USER_JWT_ALGOS,
            "options": options,
        }
        if MCP_USER_JWT_AUDIENCE:
            kwargs["audience"] = MCP_USER_JWT_AUDIENCE
        if MCP_USER_JWT_ISSUER:
            kwargs["issuer"] = MCP_USER_JWT_ISSUER

        claims = jwt.decode(tok, key, **kwargs)

        # Validaciones defensivas adicionales:
        now = int(time.time())

        exp = _safe_int(claims.get("exp"))
        if exp is not None and exp < now:
            return None, "token_expired"

        nbf = _safe_int(claims.get("nbf"))
        if nbf is not None and nbf > now:
            return None, "token_not_yet_valid"

        # user_id
        uid = claims.get(MCP_USER_ID_CLAIM)
        if uid is None:
            # fallback típico: sub
            uid = claims.get("sub")

        if uid is None:
            return None, "missing_user_id_claim"

        # Normaliza user_id a string/int simple
        try:
            if isinstance(uid, str) and uid.isdigit():
                uid_norm: Any = int(uid)
            else:
                uid_norm = uid
        except Exception:
            uid_norm = uid

        claims["_user_id"] = uid_norm
        return claims, None

    except Exception as e:
        logger.info("[MCP_USER_JWT] JWT decode error: %s", e)
        return None, "invalid_token"
