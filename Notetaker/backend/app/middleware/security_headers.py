"""Middleware para añadir headers de seguridad HTTP."""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import logging

logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware que añade headers de seguridad HTTP a todas las respuestas."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Strict-Transport-Security (HSTS)
        # Fuerza a los navegadores a usar HTTPS durante 1 año
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        
        # Content-Security-Policy (CSP)
        # Política de seguridad de contenido básica
        # Permite recursos del mismo origen, scripts inline necesarios, y conexiones a APIs externas
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'",  # unsafe-inline y unsafe-eval para compatibilidad
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: https:",
            "font-src 'self' data:",
            "connect-src 'self' https:",
            "object-src 'none'",  # Bloquea plugins como Flash
            "base-uri 'self'",  # Restringe la etiqueta <base>
            "frame-src 'self' https:",  # Permite iframes del mismo origen y HTTPS
            "form-action 'self'",
            "frame-ancestors 'self'",
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)
        
        # X-Content-Type-Options
        # Previene MIME type sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # X-Frame-Options
        # Previene clickjacking (redundante con CSP frame-ancestors pero útil para compatibilidad)
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        
        # X-XSS-Protection (legacy pero útil para navegadores antiguos)
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # Referrer-Policy
        # Controla qué información de referrer se envía
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions-Policy (anteriormente Feature-Policy)
        # Desactiva características que no se necesitan
        response.headers["Permissions-Policy"] = (
            "geolocation=(), "
            "microphone=(), "
            "camera=(), "
            "payment=(), "
            "usb=()"
        )
        
        return response
