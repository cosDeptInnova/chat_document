# cosmos_mcp_app/mcp_auth.py

from __future__ import annotations

import os
import logging
from typing import Iterable, Optional, Set, Dict, Any, Tuple
import time

logger = logging.getLogger(__name__)

# Puedes controlar si el gateway exige auth o no.
# Default seguro: ON
MCP_REQUIRE_AUTH: bool = os.getenv("MCP_REQUIRE_AUTH", "1").strip() != "0"

DEFAULT_EXEMPT_PATHS = {
    "/health",
    "/healthz",
    "/ready",
    "/readyz",
    "/live",
    "/livez",
}

_TOKEN_ENV_VARS = (
    "MCP_SERVICE_TOKEN",
    "CHATDOC_MCP_SERVICE_TOKEN",
    "MCP_AUTH_TOKEN",
)

# -----------------------------
# NUEVO: User JWT configuration
# -----------------------------
MCP_ENABLE_USER_JWT: bool = os.getenv("MCP_ENABLE_USER_JWT", "1").strip() != "0"
MCP_USER_AUTH_HEADER: str = os.getenv("MCP_USER_AUTH_HEADER", "X-User-Authorization").strip() or "X-User-Authorization"

# Env para verificar JWT (elige UNO de estos enfoques)
MCP_USER_JWT_SECRET: str = (os.getenv("MCP_USER_JWT_SECRET", "") or "").strip()
MCP_USER_JWT_ALGORITHMS: str = (os.getenv("MCP_USER_JWT_ALGORITHMS", "HS256") or "HS256").strip()
MCP_USER_JWT_ISSUER: str = (os.getenv("MCP_USER_JWT_ISSUER", "") or "").strip()
MCP_USER_JWT_AUDIENCE: str = (os.getenv("MCP_USER_JWT_AUDIENCE", "") or "").strip()

# Alternativa JWKS (RS256/ES256)
MCP_USER_JWKS_URL: str = (os.getenv("MCP_USER_JWKS_URL", "") or "").strip()
MCP_USER_JWT_REQUIRE: bool = os.getenv("MCP_USER_JWT_REQUIRE", "0").strip() == "1"  # si true y falta/invalid => 401

# Claim mapping
MCP_USER_ID_CLAIM: str = (os.getenv("MCP_USER_ID_CLAIM", "user_id") or "user_id").strip()
MCP_USER_SUB_FALLBACK: bool = os.getenv("MCP_USER_SUB_FALLBACK", "1").strip() != "0"


def _normalize_bearer(token: str) -> str:
    t = (token or "").strip()
    if not t:
        return ""
    if t.lower().startswith("bearer "):
        t = t[7:].strip()
    return t


def _split_tokens(raw: str) -> Iterable[str]:
    if not raw:
        return []
    raw = raw.replace("\n", ",")
    parts = [p.strip() for p in raw.split(",")]
    return [p for p in parts if p]


def get_expected_gateway_tokens() -> Set[str]:
    tokens: Set[str] = set()
    for env_name in _TOKEN_ENV_VARS:
        raw = os.getenv(env_name, "").strip()
        if not raw:
            continue
        for part in _split_tokens(raw):
            tok = _normalize_bearer(part)
            if tok:
                tokens.add(tok)
    return tokens


def is_gateway_auth_configured() -> bool:
    return bool(get_expected_gateway_tokens())


def validate_request_authorization_header(auth_header: Optional[str]) -> bool:
    expected = get_expected_gateway_tokens()

    if not expected:
        return not MCP_REQUIRE_AUTH

    incoming = _normalize_bearer(auth_header or "")
    if not incoming:
        return False

    return incoming in expected


# ---------------------------------------------------------------------
# NUEVO: JWT decode/verify (best-effort; NO rompe si no config)
# ---------------------------------------------------------------------
_JWKS_CACHE: Dict[str, Any] = {"fetched_at": 0.0, "jwks": None}


def _jwt_lib_available() -> bool:
    try:
        import jwt  # noqa
        return True
    except Exception:
        return False


def _should_verify_user_jwt() -> bool:
    # verificamos si está habilitado y hay config suficiente
    if not MCP_ENABLE_USER_JWT:
        return False
    if MCP_USER_JWKS_URL:
        return True
    if MCP_USER_JWT_SECRET:
        return True
    return False


async def _fetch_jwks() -> Optional[Dict[str, Any]]:
    """
    Carga JWKS con caché simple (TTL configurable).
    Best-effort: si falla, devuelve None.
    """
    if not MCP_USER_JWKS_URL:
        return None

    ttl = 300
    try:
        ttl = int(os.getenv("MCP_USER_JWKS_CACHE_TTL", "300"))
    except Exception:
        ttl = 300

    now = time.time()
    cached = _JWKS_CACHE.get("jwks")
    fetched_at = float(_JWKS_CACHE.get("fetched_at") or 0.0)
    if cached and (now - fetched_at) < ttl:
        return cached

    try:
        import httpx  # type: ignore
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(MCP_USER_JWKS_URL)
            r.raise_for_status()
            jwks = r.json()
            if isinstance(jwks, dict) and jwks.get("keys"):
                _JWKS_CACHE["jwks"] = jwks
                _JWKS_CACHE["fetched_at"] = now
                return jwks
    except Exception as e:
        logger.warning("[MCP_AUTH] No se pudo cargar JWKS: %s", e)

    return None


def _extract_user_id_from_claims(claims: Dict[str, Any]) -> Optional[str]:
    if not isinstance(claims, dict):
        return None

    val = claims.get(MCP_USER_ID_CLAIM)
    if val is not None and str(val).strip():
        return str(val).strip()

    if MCP_USER_SUB_FALLBACK:
        sub = claims.get("sub")
        if sub is not None and str(sub).strip():
            return str(sub).strip()

    return None


async def decode_and_validate_user_jwt(user_auth_header: Optional[str]) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Devuelve (claims, error_code).
    - Si no viene header => (None, None)
    - Si no hay lib/config => (None, None) (best-effort, no rompe)
    - Si requiere JWT (MCP_USER_JWT_REQUIRE=1) y falla => (None, "invalid_jwt")
    """
    raw = _normalize_bearer(user_auth_header or "")
    if not raw:
        return None, None

    if not _jwt_lib_available():
        # No rompemos: si quieres exigirlo, activa MCP_USER_JWT_REQUIRE=1 y asegúrate de instalar PyJWT.
        return None, "jwt_lib_missing" if MCP_USER_JWT_REQUIRE else None

    if not _should_verify_user_jwt():
        # No hay config de verificación; no rompemos.
        return None, "jwt_verify_not_configured" if MCP_USER_JWT_REQUIRE else None

    try:
        import jwt  # type: ignore

        algos = [a.strip() for a in (MCP_USER_JWT_ALGORITHMS or "HS256").split(",") if a.strip()]
        options = {"verify_signature": True, "verify_exp": True}

        kwargs: Dict[str, Any] = {"algorithms": algos, "options": options}

        if MCP_USER_JWT_ISSUER:
            kwargs["issuer"] = MCP_USER_JWT_ISSUER
        if MCP_USER_JWT_AUDIENCE:
            kwargs["audience"] = MCP_USER_JWT_AUDIENCE

        if MCP_USER_JWKS_URL:
            jwks = await _fetch_jwks()
            if not jwks:
                return None, "jwks_unavailable" if MCP_USER_JWT_REQUIRE else None

            unverified_header = jwt.get_unverified_header(raw)
            kid = (unverified_header or {}).get("kid")
            if not kid:
                return None, "kid_missing"

            keys = jwks.get("keys") or []
            jwk = next((k for k in keys if k.get("kid") == kid), None)
            if not jwk:
                return None, "kid_not_found"

            key_obj = jwt.algorithms.RSAAlgorithm.from_jwk(jwk)  # type: ignore[attr-defined]
            claims = jwt.decode(raw, key=key_obj, **kwargs)
            if isinstance(claims, dict):
                return claims, None
            return None, "invalid_claims"

        # HS* secret
        secret = MCP_USER_JWT_SECRET
        if not secret:
            return None, "missing_secret"

        claims = jwt.decode(raw, key=secret, **kwargs)
        if isinstance(claims, dict):
            return claims, None
        return None, "invalid_claims"

    except Exception as e:
        logger.info("[MCP_AUTH] JWT inválido: %s", e)
        return None, "invalid_jwt"


# ---------------------------------------------------------------------
# Middleware instalable en FastAPI/Starlette
# ---------------------------------------------------------------------
def install_mcp_auth_middleware(mcp_server, *, exempt_paths: Optional[Set[str]] = None) -> None:
    """
    Middleware global sobre el app subyacente del FastMCP.

    Mantiene S2S (Authorization) igual que hoy.
    Añade (opcional) JWT usuario en header configurable (X-User-Authorization por defecto),
    y expone user_id en request.state.user_id para rate-limit per-user en tools.

    No rompe:
    - Si no envías JWT usuario => sigue funcionando.
    - Si no configuras verificación JWT => best-effort y no rompe (salvo MCP_USER_JWT_REQUIRE=1).
    """
    exempt = set(exempt_paths or set()) | DEFAULT_EXEMPT_PATHS

    app = getattr(mcp_server, "app", None)
    if app is None:
        logger.warning("[MCP_AUTH] FastMCP no expone .app; no se pudo instalar middleware global.")
        return

    try:
        from starlette.middleware.base import BaseHTTPMiddleware
        from starlette.requests import Request
        from starlette.responses import JSONResponse

        class MCPServiceAuthMiddleware(BaseHTTPMiddleware):
            async def dispatch(self, request: Request, call_next):
                # Exentos
                path = request.url.path or ""
                if path in exempt:
                    return await call_next(request)

                # -----------------------
                # 1) Validación S2S (como hoy)
                # -----------------------
                if MCP_REQUIRE_AUTH or is_gateway_auth_configured():
                    auth_header = request.headers.get("Authorization")
                    ok = validate_request_authorization_header(auth_header)
                    if not ok:
                        return JSONResponse(
                            status_code=401,
                            content={"error": "Unauthorized", "detail": "MCP gateway token inválido o ausente."},
                        )

                # -----------------------
                # 2) JWT usuario (opcional) para user_id
                # -----------------------
                request.state.user_id = None
                request.state.user_claims = None
                request.state.user_jwt_error = None

                if MCP_ENABLE_USER_JWT:
                    user_hdr = request.headers.get(MCP_USER_AUTH_HEADER)
                    claims, err = await decode_and_validate_user_jwt(user_hdr)

                    if claims:
                        request.state.user_claims = claims
                        request.state.user_id = _extract_user_id_from_claims(claims)
                    elif err:
                        request.state.user_jwt_error = err
                        if MCP_USER_JWT_REQUIRE:
                            return JSONResponse(
                                status_code=401,
                                content={"error": "Unauthorized", "detail": "User JWT inválido o ausente.", "code": err},
                            )

                return await call_next(request)

        app.add_middleware(MCPServiceAuthMiddleware)
        logger.info(
            "[MCP_AUTH] Middleware instalado. require_s2s=%s user_jwt_enabled=%s user_jwt_require=%s",
            MCP_REQUIRE_AUTH or is_gateway_auth_configured(),
            MCP_ENABLE_USER_JWT,
            MCP_USER_JWT_REQUIRE,
        )
    except Exception as e:
        logger.exception("[MCP_AUTH] Error instalando middleware global: %s", e)

