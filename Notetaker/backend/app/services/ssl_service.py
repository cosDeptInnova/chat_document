"""Servicio para gestionar certificados SSL."""
import os
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import tempfile
import shutil
from cryptography import x509
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.serialization import Encoding
import requests

logger = logging.getLogger(__name__)


class SSLService:
    """Servicio para gestionar certificados SSL."""

    def _get_signature_algorithm_name(self, oid) -> str:
        """Obtiene el nombre legible del algoritmo de firma."""
        try:
            # Intentar obtener el nombre del OID
            if hasattr(oid, '_name'):
                return oid._name
            # Mapeo común de algoritmos
            oid_str = str(oid)
            if 'sha256WithRSAEncryption' in oid_str or '1.2.840.113549.1.1.11' in oid_str:
                return 'SHA256withRSA'
            elif 'sha1WithRSAEncryption' in oid_str or '1.2.840.113549.1.1.5' in oid_str:
                return 'SHA1withRSA'
            elif 'ecdsa-with-SHA256' in oid_str:
                return 'SHA256withECDSA'
            else:
                return str(oid)
        except Exception:
            return str(oid)
    
    def __init__(self):
        # Directorio donde se guardan los certificados SSL
        self.ssl_dir = Path(os.getenv("SSL_CERT_DIR", "./ssl_certs"))
        self.ssl_dir.mkdir(parents=True, exist_ok=True)
        
        # Rutas de los archivos de certificado
        self.cert_file = self.ssl_dir / "certificate.crt"
        self.key_file = self.ssl_dir / "private.key"
        self.chain_file = self.ssl_dir / "chain.crt"
        self.fullchain_file = self.ssl_dir / "fullchain.crt"
        
        # Información del último importador
        self.metadata_file = self.ssl_dir / "metadata.json"
        
    def parse_certificate(self, cert_data: bytes) -> Dict:
        """Parsea un certificado y extrae información."""
        try:
            cert = x509.load_pem_x509_certificate(cert_data, default_backend())
            
            # Extraer información del certificado
            subject = cert.subject
            issuer = cert.issuer
            
            # Obtener nombres alternativos del sujeto (SAN)
            san_names = []
            try:
                san_ext = cert.extensions.get_extension_for_oid(
                    x509.oid.ExtensionOID.SUBJECT_ALTERNATIVE_NAME
                )
                san_names = [name.value for name in san_ext.value]
            except x509.ExtensionNotFound:
                pass
            
            # Obtener nombre común (CN)
            cn = None
            for attr in subject:
                if attr.oid == x509.oid.NameOID.COMMON_NAME:
                    cn = attr.value
                    break
            
            return {
                "subject_name": cn or str(subject),
                "subject_alternative_names": san_names,
                "issuer_name": str(issuer),
                "valid_from": cert.not_valid_before.isoformat(),
                "valid_to": cert.not_valid_after.isoformat(),
                "signature_algorithm": self._get_signature_algorithm_name(cert.signature_algorithm_oid),
                "key_algorithm": cert.public_key().__class__.__name__.replace("PublicKey", ""),
                "key_size": cert.public_key().key_size if hasattr(cert.public_key(), 'key_size') else None,
            }
        except Exception as e:
            logger.error(f"Error parseando certificado: {e}")
            raise ValueError(f"Error al parsear certificado: {str(e)}")
    
    def parse_certificate_chain(self, cert_data: bytes) -> List[Dict]:
        """Parsea una cadena de certificados."""
        certificates = []
        
        try:
            # Intentar parsear múltiples certificados PEM
            cert_text = cert_data.decode('utf-8')
            cert_pem_blocks = cert_text.split('-----BEGIN CERTIFICATE-----')
            
            for block in cert_pem_blocks:
                if not block.strip():
                    continue
                
                try:
                    cert_text_block = '-----BEGIN CERTIFICATE-----' + block.strip()
                    cert_bytes = cert_text_block.encode('utf-8')
                    cert_info = self.parse_certificate(cert_bytes)
                    certificates.append(cert_info)
                except Exception as e:
                    logger.warning(f"Error parseando certificado en cadena: {e}")
                    continue
        except Exception as e:
            logger.error(f"Error decodificando cadena de certificados: {e}")
        
        return certificates
    
    def detect_intermediate_certificates(self, cert_data: bytes) -> List[bytes]:
        """
        Descarga certificados intermedios usando la extensión AIA (Authority Information Access)
        del certificado. Muchas CAs incluyen una URL caIssuers que apunta al certificado del emisor.
        """
        intermediate_certs: List[bytes] = []
        
        try:
            cert = x509.load_pem_x509_certificate(cert_data, default_backend())
            logger.info(f"Certificado emitido por: {cert.issuer}")
            
            # Obtener extensión Authority Information Access (AIA)
            try:
                aia_ext = cert.extensions.get_extension_for_oid(
                    x509.oid.ExtensionOID.AUTHORITY_INFORMATION_ACCESS
                )
            except x509.ExtensionNotFound:
                logger.info(
                    "El certificado no tiene extension AIA; no se pueden obtener intermedios automaticamente."
                )
                return []
            
            # Extraer URLs caIssuers
            ca_issuer_urls: List[str] = []
            for desc in aia_ext.value:
                if desc.access_method == x509.oid.AuthorityInformationAccessOID.CA_ISSUERS:
                    if isinstance(desc.access_location, x509.UniformResourceIdentifier):
                        ca_issuer_urls.append(desc.access_location.value)
            
            if not ca_issuer_urls:
                logger.info("AIA no contiene URLs caIssuers.")
                return []
            
            # Descargar certificado(s) desde cada URL (evitar duplicados por contenido)
            seen_der: set = set()
            
            for url in ca_issuer_urls:
                try:
                    resp = requests.get(url, timeout=15)
                    resp.raise_for_status()
                    data = resp.content
                    
                    # Puede venir en PEM (texto) o DER (binario)
                    if b"-----BEGIN CERTIFICATE-----" in data:
                        # PEM: puede haber varios bloques
                        parts = data.split(b"-----BEGIN CERTIFICATE-----")
                        for part in parts:
                            part = part.strip()
                            if not part:
                                continue
                            pem_block = b"-----BEGIN CERTIFICATE-----" + part
                            if pem_block not in seen_der:
                                try:
                                    x509.load_pem_x509_certificate(pem_block, default_backend())
                                    intermediate_certs.append(pem_block)
                                    seen_der.add(pem_block)
                                except Exception as e:
                                    logger.warning(f"Certificado PEM invalido desde {url}: {e}")
                    else:
                        # DER
                        try:
                            cert_der = x509.load_der_x509_certificate(data, default_backend())
                            der_bytes = data
                            if der_bytes in seen_der:
                                continue
                            seen_der.add(der_bytes)
                            pem_bytes = cert_der.public_bytes(Encoding.PEM)
                            intermediate_certs.append(pem_bytes)
                        except Exception:
                            # Algunos servidores devuelven PEM con Content-Type application/pkix-cert
                            try:
                                cert_pem = x509.load_pem_x509_certificate(data, default_backend())
                                pem_bytes = cert_pem.public_bytes(Encoding.PEM)
                                if pem_bytes not in seen_der:
                                    seen_der.add(pem_bytes)
                                    intermediate_certs.append(pem_bytes)
                            except Exception as e:
                                logger.warning(f"No se pudo parsear certificado desde {url}: {e}")
                                
                except requests.RequestException as e:
                    logger.warning(f"Error descargando desde {url}: {e}")
                except Exception as e:
                    logger.warning(f"Error procesando certificado desde {url}: {e}")
            
            if intermediate_certs:
                logger.info(f"Se descargaron {len(intermediate_certs)} certificado(s) intermedio(s) desde AIA.")
            else:
                logger.info(
                    "No se pudieron descargar certificados intermedios desde las URLs AIA."
                )
                
        except Exception as e:
            logger.error(f"Error detectando certificados intermedios: {e}", exc_info=True)
        
        return intermediate_certs
    
    def save_certificate(
        self,
        cert_data: bytes,
        key_data: Optional[bytes] = None,
        password: Optional[str] = None,
        intermediate_certs: Optional[List[bytes]] = None,
        imported_by: Optional[str] = None
    ) -> Dict:
        """Guarda un certificado SSL y sus componentes."""
        try:
            # Guardar certificado principal
            self.cert_file.write_bytes(cert_data)
            logger.info(f"Certificado guardado en {self.cert_file}")
            
            # Guardar clave privada si se proporciona
            if key_data:
                self.key_file.write_bytes(key_data)
                # Asegurar permisos restrictivos en la clave privada
                if os.name != 'nt':  # No Windows
                    os.chmod(self.key_file, 0o600)
                logger.info(f"Clave privada guardada en {self.key_file}")
            
            # Guardar certificados intermedios
            if intermediate_certs:
                chain_data = b'\n'.join(intermediate_certs)
                self.chain_file.write_bytes(chain_data)
                logger.info(f"Cadena de certificados guardada en {self.chain_file}")
                
                # Crear fullchain (certificado + cadena)
                fullchain_data = cert_data + b'\n' + chain_data
                self.fullchain_file.write_bytes(fullchain_data)
                logger.info(f"Fullchain guardado en {self.fullchain_file}")
            else:
                # Si no hay intermedios, fullchain es solo el certificado
                self.fullchain_file.write_bytes(cert_data)
            
            # Guardar metadata
            import json
            metadata = {
                "imported_by": imported_by or "Unknown",
                "imported_at": datetime.now().isoformat(),
                "certificate_info": self.parse_certificate(cert_data)
            }
            self.metadata_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            
            return {
                "success": True,
                "message": "Certificado guardado exitosamente",
                "certificate_info": metadata["certificate_info"]
            }
            
        except Exception as e:
            logger.error(f"Error guardando certificado: {e}")
            raise ValueError(f"Error al guardar certificado: {str(e)}")
    
    def get_certificate_summary(self) -> Optional[Dict]:
        """Obtiene un resumen del certificado actual."""
        if not self.cert_file.exists():
            return None
        
        try:
            cert_data = self.cert_file.read_bytes()
            cert_info = self.parse_certificate(cert_data)
            
            # Cargar metadata si existe
            metadata = {}
            if self.metadata_file.exists():
                import json
                metadata = json.loads(self.metadata_file.read_text(encoding='utf-8'))
            
            # Parsear cadena de certificados si existe
            chain_info = []
            if self.chain_file.exists():
                chain_data = self.chain_file.read_bytes()
                chain_info = self.parse_certificate_chain(chain_data)
            
            return {
                "certificate": cert_info,
                "chain": chain_info,
                "imported_by": metadata.get("imported_by"),
                "imported_at": metadata.get("imported_at"),
                "has_private_key": self.key_file.exists(),
            }
        except Exception as e:
            logger.error(f"Error obteniendo resumen de certificado: {e}")
            return None
    
    def apply_to_nginx(self, nginx_config_path: Optional[str] = None) -> Dict:
        """Aplica el certificado a nginx (requiere configuración manual o script)."""
        # Esta función prepara los archivos pero no modifica nginx directamente
        # por seguridad. El administrador debe configurar nginx manualmente o
        # usar un script externo con permisos adecuados.
        
        if not self.cert_file.exists():
            raise ValueError("No hay certificado para aplicar")
        
        if not self.key_file.exists():
            raise ValueError("No hay clave privada para aplicar")
        
        # Preparar instrucciones para el administrador
        instructions = {
            "certificate_path": str(self.cert_file.absolute()),
            "key_path": str(self.key_file.absolute()),
            "chain_path": str(self.chain_file.absolute()) if self.chain_file.exists() else None,
            "fullchain_path": str(self.fullchain_file.absolute()),
            "nginx_config_example": f"""
# Ejemplo de configuración para nginx:
server {{
    listen 443 ssl http2;
    server_name notetaker.cosgs.com;
    
    ssl_certificate {self.fullchain_file.absolute()};
    ssl_certificate_key {self.key_file.absolute()};
    
    # Configuración SSL recomendada
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    # ... resto de configuración ...
}}
"""
        }
        
        logger.info("Certificado preparado para aplicar a nginx")
        logger.info(f"Rutas: cert={self.cert_file}, key={self.key_file}")
        
        return {
            "success": True,
            "message": "Certificado preparado. Configure nginx manualmente o use el script de aplicación.",
            "instructions": instructions
        }
    
    def validate_certificate_format(self, cert_data: bytes, cert_type: str) -> bool:
        """Valida el formato de un certificado."""
        try:
            if cert_type in ['.crt', '.cer', '.pem']:
                # Formato PEM
                x509.load_pem_x509_certificate(cert_data, default_backend())
                return True
            elif cert_type == '.p7b':
                # PKCS#7 - requiere procesamiento adicional
                # Por ahora, intentamos parsearlo como PEM
                try:
                    x509.load_pem_x509_certificate(cert_data, default_backend())
                    return True
                except:
                    # Podría ser DER, pero por simplicidad retornamos False
                    return False
            else:
                # .pfx, .keystore, .jks requieren procesamiento especial
                return False
        except Exception as e:
            logger.error(f"Error validando formato de certificado: {e}")
            return False
