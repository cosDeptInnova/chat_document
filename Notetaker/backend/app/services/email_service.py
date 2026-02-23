"""Servicio de envío de emails usando Microsoft Graph API."""
import requests
import msal
import base64
import json
import logging
from typing import List, Union, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
from app.config import settings

logger = logging.getLogger(__name__)


class GraphTool:
    """
    Clase para enviar correos electrónicos usando Microsoft Graph API.
    Adaptada para usar configuración desde settings.
    """
    
    GRAPH_ENDPOINT = "https://graph.microsoft.com/v1.0"
    SCOPES = ["https://graph.microsoft.com/.default"]
    
    def __init__(self):
        """Inicializa GraphTool con configuración desde settings."""
        self.tenant_id = settings.graph_tenant_id or ""
        self.client_id = settings.graph_client_id or ""
        self.client_secret = settings.graph_client_secret or ""
        self.user_email = settings.graph_user_email or ""
        
        if not all([self.tenant_id, self.client_id, self.client_secret, self.user_email]):
            logger.warning("⚠️ Configuración de Microsoft Graph API incompleta. El envío de emails no funcionará.")
            self.app = None
            self.access_token = None
            return
        
        try:
            self.app = msal.ConfidentialClientApplication(
                client_id=self.client_id,
                client_credential=self.client_secret,
                authority=f"https://login.microsoftonline.com/{self.tenant_id}"
            )
            
            self.access_token = self._obtener_token()
        except Exception as e:
            logger.error(f"Error inicializando GraphTool: {e}")
            self.app = None
            self.access_token = None
    
    def _obtener_token(self) -> str:
        """Obtiene token de acceso de Microsoft Graph API."""
        if not self.app:
            raise Exception("GraphTool no está configurado correctamente")
        
        try:
            result = self.app.acquire_token_for_client(scopes=self.SCOPES)
            
            if "access_token" in result:
                logger.debug("✅ Token de Microsoft Graph obtenido exitosamente")
                return result["access_token"]
            else:
                error = result.get("error_description", "Error desconocido")
                raise Exception(f"Error al obtener token: {error}")
        except Exception as e:
            logger.error(f"Error en autenticación Microsoft Graph: {str(e)}")
            raise
    
    def _construir_mensaje(self,
                          destinatarios: Union[str, List[str]],
                          asunto: str,
                          cuerpo: str,
                          cc: Optional[Union[str, List[str]]] = None,
                          cco: Optional[Union[str, List[str]]] = None,
                          es_html: bool = False,
                          adjuntos: Optional[List[str]] = None,
                          from_name: Optional[str] = None) -> dict:
        """Construye el payload del mensaje para Microsoft Graph API."""
        if isinstance(destinatarios, str):
            destinatarios = [destinatarios]
        
        mensaje = {
            "message": {
                "subject": asunto,
                "body": {
                    "contentType": "HTML" if es_html else "Text",
                    "content": cuerpo
                },
                "toRecipients": [
                    {"emailAddress": {"address": email}} 
                    for email in destinatarios
                ]
            },
            "saveToSentItems": True
        }
        
        # Configurar nombre del remitente si se proporciona
        if from_name and self.user_email:
            mensaje["message"]["from"] = {
                "emailAddress": {
                    "address": self.user_email,
                    "name": from_name
                }
            }
        
        if cc:
            if isinstance(cc, str):
                cc = [cc]
            mensaje["message"]["ccRecipients"] = [
                {"emailAddress": {"address": email}} for email in cc
            ]
        
        if cco:
            if isinstance(cco, str):
                cco = [cco]
            mensaje["message"]["bccRecipients"] = [
                {"emailAddress": {"address": email}} for email in cco
            ]
        
        if adjuntos:
            mensaje["message"]["attachments"] = []
            for archivo_path in adjuntos:
                ruta = Path(archivo_path)
                if ruta.exists():
                    with open(ruta, 'rb') as f:
                        contenido = f.read()
                        contenido_base64 = base64.b64encode(contenido).decode('utf-8')
                    
                    adjunto = {
                        "@odata.type": "#microsoft.graph.fileAttachment",
                        "name": ruta.name,
                        "contentBytes": contenido_base64
                    }
                    mensaje["message"]["attachments"].append(adjunto)
                else:
                    logger.warning(f"Archivo no encontrado: {archivo_path}")
        
        return mensaje
    
    def enviar_correo(self,
                     destinatarios: Union[str, List[str]],
                     asunto: str,
                     cuerpo: str,
                     cc: Optional[Union[str, List[str]]] = None,
                     cco: Optional[Union[str, List[str]]] = None,
                     adjuntos: Optional[List[str]] = None,
                     es_html: bool = False,
                     from_name: Optional[str] = None) -> bool:
        """Envía un correo electrónico usando Microsoft Graph API."""
        if not self.app or not self.access_token:
            logger.error("GraphTool no está configurado o no tiene token de acceso")
            return False
        
        try:
            payload = self._construir_mensaje(
                destinatarios=destinatarios,
                asunto=asunto,
                cuerpo=cuerpo,
                cc=cc,
                cco=cco,
                es_html=es_html,
                adjuntos=adjuntos,
                from_name=from_name
            )
            
            if self.user_email:
                endpoint = f"{self.GRAPH_ENDPOINT}/users/{self.user_email}/sendMail"
            else:
                endpoint = f"{self.GRAPH_ENDPOINT}/me/sendMail"
            
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                endpoint,
                headers=headers,
                data=json.dumps(payload)
            )
            
            if response.status_code == 202:
                logger.info(f"✅ Correo enviado exitosamente a {destinatarios}")
                return True
            else:
                logger.error(f"❌ Error al enviar correo: {response.status_code} - {response.text}")
                # Intentar renovar token si es error 401
                if response.status_code == 401:
                    try:
                        self.renovar_token()
                        # Reintentar una vez
                        response = requests.post(
                            endpoint,
                            headers={"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"},
                            data=json.dumps(payload)
                        )
                        if response.status_code == 202:
                            logger.info(f"✅ Correo enviado exitosamente a {destinatarios} (después de renovar token)")
                            return True
                    except Exception as e:
                        logger.error(f"Error al renovar token: {e}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error al enviar correo: {str(e)}")
            return False
    
    def renovar_token(self):
        """Renueva el token de acceso."""
        self.access_token = self._obtener_token()


# Instancia global del servicio de email
_email_service: Optional[GraphTool] = None


def get_email_service() -> GraphTool:
    """Obtiene la instancia del servicio de email (singleton)."""
    global _email_service
    if _email_service is None:
        _email_service = GraphTool()
    return _email_service


def send_password_reset_email(email: str, reset_token: str, reset_url: str) -> bool:
    """
    Envía un email con el link para resetear la contraseña.
    
    Args:
        email: Email del destinatario
        reset_token: Token de reset (para logging, no se incluye en el email)
        reset_url: URL completa con el token para resetear la contraseña
        
    Returns:
        True si el email se envió exitosamente, False en caso contrario
    """
    email_service = get_email_service()
    
    if not email_service.app:
        logger.error("No se puede enviar email: servicio de email no configurado")
        return False
    
    # Construir HTML del email
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
            }}
            .container {{
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background-color: #0066FF;
                color: white;
                padding: 20px;
                text-align: center;
                border-radius: 5px 5px 0 0;
            }}
            .content {{
                background-color: #f9f9f9;
                padding: 30px;
                border-radius: 0 0 5px 5px;
            }}
            .button {{
                display: inline-block;
                padding: 12px 30px;
                background-color: #0066FF;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                margin: 20px 0;
            }}
            .footer {{
                margin-top: 20px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                font-size: 12px;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Cosmos NoteTaker</h1>
            </div>
            <div class="content">
                <h2>Recuperación de Contraseña</h2>
                <p>Hemos recibido una solicitud para restablecer tu contraseña.</p>
                <p>Haz clic en el siguiente botón para crear una nueva contraseña:</p>
                <p style="text-align: center;">
                    <a href="{reset_url}" class="button">Restablecer Contraseña</a>
                </p>
                <p>O copia y pega este enlace en tu navegador:</p>
                <p style="word-break: break-all; color: #0066FF;">{reset_url}</p>
                <p><strong>Este enlace expirará en 1 hora.</strong></p>
                <p>Si no solicitaste este cambio, puedes ignorar este correo de forma segura.</p>
                <div class="footer">
                    <p>Este es un correo automático, por favor no respondas.</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    asunto = "Recuperación de Contraseña - Cosmos NoteTaker"
    
    try:
        success = email_service.enviar_correo(
            destinatarios=email,
            asunto=asunto,
            cuerpo=html_body,
            es_html=True,
            from_name="Cosmos Notetaker"
        )
        
        if success:
            logger.info(f"✅ Email de recuperación enviado a {email}")
        else:
            logger.error(f"❌ Error al enviar email de recuperación a {email}")
        
        return success
    except Exception as e:
        logger.error(f"Error al enviar email de recuperación: {e}")
        return False


def send_alert_email(
    alert_type: str,
    title: str,
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Envía un email de alerta a los administradores configurados en ADMIN_EMAILS.
    
    Args:
        alert_type: Tipo de alerta (ej: "timeout", "celery_queue", "db_pool", "worker_down")
        title: Título de la alerta
        message: Mensaje principal de la alerta
        details: Diccionario con detalles adicionales (opcional)
        
    Returns:
        True si el email se envió exitosamente, False en caso contrario
    """
    email_service = get_email_service()
    
    if not email_service.app:
        logger.error("No se puede enviar email de alerta: servicio de email no configurado")
        return False
    
    # Obtener emails de administradores desde settings
    admin_emails_str = settings.admin_emails or ""
    if not admin_emails_str:
        logger.warning("No se configuraron ADMIN_EMAILS en .env, no se puede enviar alerta")
        return False
    
    # Parsear lista de emails (separados por coma)
    admin_emails = [email.strip() for email in admin_emails_str.split(",") if email.strip()]
    
    if not admin_emails:
        logger.warning("Lista de ADMIN_EMAILS vacía, no se puede enviar alerta")
        return False
    
    # Construir HTML del email de alerta
    details_html = ""
    if details:
        details_html = '<div style="margin-top: 20px; padding: 15px; background-color: #f3f4f6; border-radius: 5px;">'
        details_html += '<h3 style="margin-top: 0; color: #111827;">Detalles:</h3>'
        details_html += '<ul style="margin: 0; padding-left: 20px; color: #374151;">'
        for key, value in details.items():
            details_html += f'<li><strong>{key}:</strong> {value}</li>'
        details_html += '</ul>'
        details_html += '</div>'
    
    # Iconos según tipo de alerta
    alert_icons = {
        "timeout": "⏱️",
        "celery_queue": "📋",
        "db_pool": "💾",
        "worker_down": "⚠️",
        "beat_down": "⏰",
        "error": "❌",
        "warning": "⚠️"
    }
    icon = alert_icons.get(alert_type, "⚠️")
    
    html_body = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
            }}
            .container {{
                max-width: 600px;
                margin: 0 auto;
                padding: 20px;
            }}
            .header {{
                background-color: #dc2626;
                color: white;
                padding: 20px;
                text-align: center;
                border-radius: 5px 5px 0 0;
            }}
            .content {{
                background-color: #f9f9f9;
                padding: 30px;
                border-radius: 0 0 5px 5px;
            }}
            .alert-type {{
                display: inline-block;
                padding: 5px 15px;
                background-color: #fee2e2;
                color: #991b1b;
                border-radius: 3px;
                font-size: 12px;
                font-weight: bold;
                text-transform: uppercase;
                margin-bottom: 15px;
            }}
            .footer {{
                margin-top: 20px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                font-size: 12px;
                color: #666;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>{icon} Alerta del Sistema - Cosmos NoteTaker</h1>
            </div>
            <div class="content">
                <div class="alert-type">{alert_type}</div>
                <h2>{title}</h2>
                <p>{message}</p>
                {details_html}
                <div class="footer">
                    <p>Este es un correo automático del sistema de monitoreo.</p>
                    <p>Fecha: {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    asunto = f"[ALERTA] {title} - Cosmos NoteTaker"
    
    try:
        success = email_service.enviar_correo(
            destinatarios=admin_emails,
            asunto=asunto,
            cuerpo=html_body,
            es_html=True,
            from_name="Cosmos Notetaker - Sistema de Monitoreo"
        )
        
        if success:
            logger.info(f"✅ Email de alerta '{alert_type}' enviado a administradores: {admin_emails}")
        else:
            logger.error(f"❌ Error al enviar email de alerta '{alert_type}' a administradores")
        
        return success
    except Exception as e:
        logger.error(f"Error al enviar email de alerta: {e}")
        return False


def generate_meeting_summary_email_html(
    meeting_title: Optional[str],
    meeting_date: datetime,
    meeting_duration: Optional[str],
    participants: List[str],
    summary_text: Optional[str],
    insights: Optional[Dict[str, Any]],
    organizer_name: Optional[str] = None,
    meeting_id: Optional[str] = None,
    logo_base64: Optional[str] = None,
    cosmos_logo_base64: Optional[str] = None,
    frontend_url: Optional[str] = None
) -> str:
    """
    Genera el HTML del email con el resumen de la reunión.
    
    Args:
        meeting_title: Título de la reunión
        meeting_date: Fecha y hora de la reunión
        meeting_duration: Duración de la reunión (opcional)
        participants: Lista de nombres/emails de participantes
        summary_text: Texto del resumen de la reunión
        insights: Diccionario con insights de la reunión (topics, highlights, etc.)
        organizer_name: Nombre del organizador (opcional)
        logo_base64: Logo en base64 (opcional)
        frontend_url: URL del frontend para el logo (opcional)
        
    Returns:
        HTML del email como string
    """
    # Formatear fecha
    days = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
    months = ['enero', 'febrero', 'marzo', 'abril', 'mayo', 'junio', 
              'julio', 'agosto', 'septiembre', 'octubre', 'noviembre', 'diciembre']
    
    date_str = meeting_date.strftime(f"%d de {months[meeting_date.month - 1]} de %Y · %H:%M")
    day_name = days[meeting_date.weekday()]
    full_date_str = f"{day_name} {date_str}"
    
    # Logos - usar base64 incrustado, tamaño pequeño como título, uno encima del otro
    # Usar width y max-width para forzar el tamaño (algunos clientes de email ignoran solo height)
    # Tamaño doble: 240x48
    logos_html = ""
    if cosmos_logo_base64 or logo_base64:
        logos_html = '<div style="text-align: center; margin-bottom: 24px;">'
        if cosmos_logo_base64:
            logos_html += f'<div style="margin-bottom: 8px;"><img src="data:image/png;base64,{cosmos_logo_base64}" alt="Cosmos" width="240" height="48" style="max-width: 240px; width: 240px; height: auto; max-height: 48px; display: block; margin-left: auto; margin-right: auto;"></div>'
        if logo_base64:
            logos_html += f'<div><img src="data:image/png;base64,{logo_base64}" alt="Notetaker" width="240" height="48" style="max-width: 240px; width: 240px; height: auto; max-height: 48px; display: block; margin-left: auto; margin-right: auto;"></div>'
        logos_html += '</div>'
    elif frontend_url:
        # Fallback: usar URL pública solo si no hay base64
        logos_html = '<div style="text-align: center; margin-bottom: 24px;">'
        cosmos_logo_url = f"{frontend_url}/cosmos-logo.png"
        logo_url = f"{frontend_url}/notetaker-light.png"
        logos_html += f'<div style="margin-bottom: 8px;"><img src="{cosmos_logo_url}" alt="Cosmos" width="240" height="48" style="max-width: 240px; width: 240px; height: auto; max-height: 48px; display: block; margin-left: auto; margin-right: auto;"></div>'
        logos_html += f'<div><img src="{logo_url}" alt="Notetaker" width="240" height="48" style="max-width: 240px; width: 240px; height: auto; max-height: 48px; display: block; margin-left: auto; margin-right: auto;"></div>'
        logos_html += '</div>'
    else:
        logos_html = '<h1 style="color: #6366f1; font-size: 18px; margin-bottom: 24px; text-align: center;">Notetaker</h1>'
    
    # Botón "Ver reunión" - más grande y a la derecha (como en MeetGeek)
    # Usar tabla HTML para mejor compatibilidad con clientes de email
    view_meeting_button = ""
    if meeting_id:
        # Usar BACKEND_PUBLIC_URL o FRONTEND_URL si está configurado, sino usar la URL interna
        base_url = settings.backend_public_url or settings.frontend_url or "http://172.29.14.14:5173"
        meeting_url = f"{base_url}/meetings/{meeting_id}?redirect=true"
        view_meeting_button = f'''
        <table border="0" cellspacing="0" cellpadding="0" style="margin: 0; padding: 0;">
            <tr>
                <td align="right" style="padding: 0;">
                    <a href="{meeting_url}" style="display: inline-block; padding: 16px 32px; background-color: #6366f1; color: #ffffff; text-decoration: none; border-radius: 6px; font-weight: 600; font-size: 16px; white-space: nowrap; line-height: 1.2;">
                        Ver Reunión
                    </a>
                </td>
            </tr>
        </table>
        '''
    
    # Participantes como lista
    participants_html = ""
    if participants:
        participants_html = '<div style="margin-top: 12px;">'
        participants_html += '<ul style="margin: 0; padding-left: 20px; color: #374151; font-size: 14px; line-height: 1.8;">'
        for participant in participants:
            # Escapar HTML para seguridad
            participant_escaped = participant.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            participants_html += f'<li style="margin-bottom: 4px;">{participant_escaped}</li>'
        participants_html += '</ul>'
        participants_html += '</div>'
    
    # Resumen
    summary_html = ""
    if summary_text:
        summary_html = f'''
        <div style="margin-top: 24px;">
            <table width="100%" border="0" cellspacing="0" cellpadding="0" style="margin-bottom: 12px;">
                <tr>
                    <td align="left" valign="middle" style="padding: 0;">
                        <h2 style="font-size: 18px; font-weight: 600; color: #111827; margin: 0;">Resumen de la Reunión</h2>
                    </td>
                    <td align="right" valign="middle" style="padding: 0; padding-left: 16px;">
                        {view_meeting_button}
                    </td>
                </tr>
            </table>
            <p style="font-style: italic; color: #6b7280; font-size: 13px; margin-bottom: 12px;">
                *Los resúmenes de IA pueden contener errores. Considera verificar la información importante.
            </p>
            <p style="color: #374151; line-height: 1.6; margin: 0;">
                {summary_text}
            </p>
        </div>
        '''
    
    # Topics & Highlights
    topics_html = ""
    if insights:
        topics = insights.get("topics", [])
        decisions = insights.get("decisions", [])
        action_items = insights.get("action_items", [])
        
        # Combinar todos los highlights
        highlights = []
        
        # Añadir topics como highlights
        for topic in topics[:5]:  # Limitar a 5 topics
            if isinstance(topic, dict):
                # Si es un objeto con name y weight, extraer solo el name
                topic_name = topic.get("name", "")
                if topic_name:
                    highlights.append({"text": topic_name, "timestamp": None})
            elif isinstance(topic, str):
                # Si es un string, usarlo directamente
                highlights.append({"text": topic, "timestamp": None})
        
        # Añadir decisions como highlights
        for decision in decisions[:3]:  # Limitar a 3 decisions
            highlights.append({"text": f"Decisión: {decision}", "timestamp": None})
        
        # Añadir action items como highlights
        for item in action_items[:5]:  # Limitar a 5 action items
            if isinstance(item, dict):
                task = item.get("task", "")
                owner = item.get("owner", "")
                if task:
                    highlight_text = task
                    if owner:
                        highlight_text += f" (Responsable: {owner})"
                    highlights.append({"text": highlight_text, "timestamp": None})
            elif isinstance(item, str):
                highlights.append({"text": item, "timestamp": None})
        
        if highlights:
            topics_html = '<div style="margin-top: 32px;">'
            topics_html += '<table width="100%" border="0" cellspacing="0" cellpadding="0" style="margin-bottom: 0;">'
            topics_html += '<tr>'
            topics_html += '<td align="left" valign="middle" style="padding: 0;"><h2 style="font-size: 18px; font-weight: 600; color: #111827; margin: 0 0 16px 0;">Temas y Destacados</h2></td>'
            topics_html += f'<td align="right" valign="middle" style="padding: 0; padding-left: 16px;">{view_meeting_button}</td>'
            topics_html += '</tr>'
            topics_html += '</table>'
            
            # Contenedor para la lista con margen superior para separar del título
            topics_html += '<div style="margin-top: 16px;">'
            
            # Agrupar por categorías si es posible
            current_category = None
            for highlight in highlights[:10]:  # Limitar a 10 highlights
                text = highlight.get("text", "")
                timestamp = highlight.get("timestamp")
                
                if text:
                    timestamp_str = f" ({timestamp})" if timestamp else ""
                    topics_html += f'''
                    <div style="margin-bottom: 12px; padding-left: 16px; border-left: 3px solid #6366f1;">
                        <p style="color: #374151; line-height: 1.6; margin: 0;">
                            {text}{timestamp_str}
                        </p>
                    </div>
                    '''
            
            topics_html += '</div>'  # Cerrar contenedor de lista
            
            topics_html += '</div>'
    
    # Construir HTML completo
    html = f'''
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Resumen de Reunión</title>
    </head>
    <body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f9fafb;">
        <div style="max-width: 600px; margin: 0 auto; background-color: #ffffff; padding: 40px;">
            <!-- Header con logos -->
            <div style="text-align: center; margin-bottom: 32px;">
                {logos_html}
            </div>
            
            <!-- Mensaje de compartido -->
            {f'<p style="color: #6b7280; font-size: 14px; margin-bottom: 24px; text-align: center;">{organizer_name} compartió las notas de la reunión contigo.</p>' if organizer_name else ''}
            
            <!-- Información de la reunión -->
            <div style="background-color: #f3f4f6; border-radius: 8px; padding: 24px; margin-bottom: 32px;">
                <div style="color: #6b7280; font-size: 14px; margin-bottom: 8px;">
                    {full_date_str}
                </div>
                <h1 style="font-size: 24px; font-weight: 700; color: #111827; margin: 0 0 16px 0;">
                    {meeting_title or "Reunión sin título"}
                </h1>
                {participants_html}
            </div>
            
            {summary_html}
            
            {topics_html}
            
            <!-- Footer -->
            <div style="margin-top: 48px; padding-top: 24px; border-top: 1px solid #e5e7eb; text-align: center;">
                <p style="color: #9ca3af; font-size: 12px; margin: 0;">
                    Este correo fue generado automáticamente por Notetaker
                </p>
            </div>
        </div>
    </body>
    </html>
    '''
    
    return html


def send_meeting_summary_email(
    recipients: List[str],
    subject: str,
    meeting_title: Optional[str],
    meeting_date: datetime,
    meeting_duration: Optional[str],
    participants: List[str],
    summary_text: Optional[str],
    insights: Optional[Dict[str, Any]],
    organizer_name: Optional[str] = None,
    meeting_id: Optional[str] = None,
    cc: Optional[List[str]] = None,
    logo_base64: Optional[str] = None,
    cosmos_logo_base64: Optional[str] = None,
    frontend_url: Optional[str] = None
) -> bool:
    """
    Envía un email con el resumen de una reunión.
    
    Args:
        recipients: Lista de emails destinatarios
        subject: Asunto del email
        meeting_title: Título de la reunión
        meeting_date: Fecha y hora de la reunión
        meeting_duration: Duración de la reunión (opcional)
        participants: Lista de nombres/emails de participantes
        summary_text: Texto del resumen de la reunión
        insights: Diccionario con insights de la reunión
        organizer_name: Nombre del organizador (opcional)
        cc: Lista de emails en copia (opcional)
        logo_base64: Logo en base64 (opcional)
        frontend_url: URL del frontend para el logo (opcional)
        
    Returns:
        True si el email se envió exitosamente, False en caso contrario
    """
    email_service = get_email_service()
    
    if not email_service.app:
        logger.error("No se puede enviar email: servicio de email no configurado")
        return False
    
    # Generar HTML del email
    html_body = generate_meeting_summary_email_html(
        meeting_title=meeting_title,
        meeting_date=meeting_date,
        meeting_duration=meeting_duration,
        participants=participants,
        summary_text=summary_text,
        insights=insights,
        organizer_name=organizer_name,
        meeting_id=meeting_id,
        logo_base64=logo_base64,
        cosmos_logo_base64=cosmos_logo_base64,
        frontend_url=frontend_url
    )
    
    try:
        success = email_service.enviar_correo(
            destinatarios=recipients,
            asunto=subject,
            cuerpo=html_body,
            cc=cc,
            es_html=True,
            from_name="Cosmos Notetaker"
        )
        
        if success:
            logger.info(f"✅ Email de resumen de reunión enviado a {recipients}")
        else:
            logger.error(f"❌ Error al enviar email de resumen de reunión a {recipients}")
        
        return success
    except Exception as e:
        logger.error(f"Error al enviar email de resumen de reunión: {e}")
        return False

