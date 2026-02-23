"""Endpoints de seguridad y gestión de certificados SSL."""
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Optional, List
from app.database import get_db
from app.api.dependencies.auth import get_current_admin_user
from app.models.user import User
from app.services.ssl_service import SSLService
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/security", tags=["security"])

ssl_service = SSLService()


@router.get("/ssl/certificate")
async def get_ssl_certificate_summary(
    current_user: User = Depends(get_current_admin_user)
):
    """Obtiene el resumen del certificado SSL actual."""
    try:
        summary = ssl_service.get_certificate_summary()
        if summary is None:
            return JSONResponse(
                status_code=404,
                content={"message": "No hay certificado SSL configurado"}
            )
        return summary
    except Exception as e:
        logger.error(f"Error obteniendo resumen de certificado: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener certificado: {str(e)}")


@router.post("/ssl/import")
async def import_ssl_certificate(
    certificate: UploadFile = File(..., description="Archivo del certificado SSL (.crt, .cer, .pem)"),
    private_key: Optional[UploadFile] = File(None, description="Archivo de clave privada (.key)"),
    keystore_password: Optional[str] = Form(None, description="Contraseña del keystore si es necesario"),
    intermediate_mode: str = Form("manual", description="Modo de certificados intermedios: 'manual' o 'automatic'"),
    intermediate_certificates: Optional[List[UploadFile]] = File(None, description="Certificados intermedios (solo si modo manual)"),
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db)
):
    """Importa un certificado SSL."""
    try:
        # Leer certificado
        cert_data = await certificate.read()
        cert_filename = certificate.filename or ""
        cert_ext = cert_filename.lower().split('.')[-1] if '.' in cert_filename else ''
        
        # Validar formato del certificado
        if not ssl_service.validate_certificate_format(cert_data, f'.{cert_ext}'):
            raise HTTPException(
                status_code=400,
                detail=f"Formato de certificado no válido. Formatos soportados: .crt, .cer, .pem"
            )
        
        # Leer clave privada si se proporciona
        key_data = None
        if private_key:
            key_data = await private_key.read()
            logger.info("Clave privada recibida")
        elif cert_ext in ['crt', 'cer', 'pem']:
            # Si es .crt, .cer o .pem, esperamos que se proporcione la clave
            raise HTTPException(
                status_code=400,
                detail="Se requiere la clave privada para certificados .crt/.cer/.pem"
            )
        
        # Procesar certificados intermedios
        intermediate_certs_data = []
        
        if intermediate_mode == "automatic":
            # Intentar detectar automáticamente
            logger.info("Intentando detectar certificados intermedios automáticamente...")
            detected = ssl_service.detect_intermediate_certificates(cert_data)
            intermediate_certs_data.extend(detected)
            if not detected:
                logger.warning("No se pudieron detectar certificados intermedios automáticamente")
        elif intermediate_mode == "manual" and intermediate_certificates:
            # Leer certificados intermedios proporcionados manualmente
            for inter_cert in intermediate_certificates:
                inter_data = await inter_cert.read()
                intermediate_certs_data.append(inter_data)
                logger.info(f"Certificado intermedio recibido: {inter_cert.filename}")
        
        # Guardar certificado
        result = ssl_service.save_certificate(
            cert_data=cert_data,
            key_data=key_data,
            password=keystore_password,
            intermediate_certs=intermediate_certs_data if intermediate_certs_data else None,
            imported_by=current_user.email
        )
        
        # Intentar aplicar a nginx (preparar archivos)
        apply_result = ssl_service.apply_to_nginx()
        
        return {
            "success": True,
            "message": "Certificado importado exitosamente",
            "certificate_info": result["certificate_info"],
            "nginx_instructions": apply_result.get("instructions"),
            "note": "El certificado se ha guardado. Debe configurar nginx manualmente o reiniciar el servicio web."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error importando certificado SSL: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error al importar certificado: {str(e)}")


@router.get("/ssl/chain")
async def get_ssl_certificate_chain(
    current_user: User = Depends(get_current_admin_user)
):
    """Obtiene la cadena completa de certificados SSL."""
    try:
        summary = ssl_service.get_certificate_summary()
        if summary is None:
            return JSONResponse(
                status_code=404,
                content={"message": "No hay certificado SSL configurado"}
            )
        
        return {
            "certificate": summary["certificate"],
            "intermediate_chain": summary.get("chain", []),
            "imported_by": summary.get("imported_by"),
            "imported_at": summary.get("imported_at")
        }
    except Exception as e:
        logger.error(f"Error obteniendo cadena de certificados: {e}")
        raise HTTPException(status_code=500, detail=f"Error al obtener cadena: {str(e)}")
