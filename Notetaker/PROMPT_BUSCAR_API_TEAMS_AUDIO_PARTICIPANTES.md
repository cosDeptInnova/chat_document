# Prompt para Buscar Cómo Acceder al Audio y Participantes de Teams vía API

## Prompt Completo para IA

```
Necesito ayuda para integrar la obtención de audio y participantes de reuniones de Microsoft Teams usando la Microsoft Graph API en mi proyecto Python. Te proporciono el contexto completo:

## Contexto del Proyecto

**Descripción del Sistema:**
- Tengo un bot automatizado que se une a reuniones de Microsoft Teams usando Playwright (automatización de navegador)
- El bot se autentica con una cuenta de Teams específica (email y contraseña)
- El bot entra físicamente a las reuniones como un participante normal
- Después de que el bot sale de la reunión, necesito obtener:
  1. El audio de la reunión (grabación)
  2. La lista de participantes con sus nombres y emails

**Configuración Actual de Azure AD:**
- Ya tengo una aplicación registrada en Azure AD
- La aplicación tiene el permiso `OnlineMeetingRecording.Read.All` configurado y aprobado
- Uso autenticación OAuth 2.0 con Client Credentials Flow (aplicación, no usuario delegado)
- Tengo Client ID y Client Secret configurados
- El bot usa la cuenta: `cosmos_bot@cosgs.com` (ya tiene permisos aceptados para solicitar audios)

**Stack Tecnológico:**
- Python 3.x
- FastAPI (backend)
- Microsoft Graph API (v1.0)
- Biblioteca `msal` (Microsoft Authentication Library) para autenticación
- Biblioteca `httpx` para llamadas HTTP asíncronas
- El bot usa Playwright para automatizar el navegador y unirse a reuniones

**Flujo Actual:**
1. El bot se une a una reunión de Teams usando Playwright
2. El bot captura audio del navegador mientras está en la reunión (fallback)
3. Cuando el bot sale de la reunión, necesito:
   - Intentar obtener el audio oficial de la reunión desde Graph API (opción principal)
   - Obtener la lista de participantes desde Graph API
   - Si Graph API no devuelve audio, usar el audio capturado del navegador

## Lo que Necesito Saber

**1. Obtener Audio de la Reunión:**
- ¿Cuál es el endpoint exacto de Microsoft Graph API para obtener el audio/grabación de una reunión de Teams?
- ¿Cómo identifico una reunión específica? (tengo el `onlineMeetingId` y el `meetingId` interno)
- ¿Qué formato devuelve el audio? (MP4, MP3, WAV, etc.)
- ¿Hay algún tiempo de espera después de que termina la reunión antes de que la grabación esté disponible?
- ¿Cómo descargo el contenido del audio? ¿Es un stream directo o necesito hacer múltiples requests?
- ¿Necesito algún permiso adicional además de `OnlineMeetingRecording.Read.All`?

**2. Obtener Lista de Participantes:**
- ¿Cuál es el endpoint de Microsoft Graph API para obtener los participantes de una reunión?
- ¿Qué información incluye cada participante? (nombre, email, hora de entrada/salida, etc.)
- ¿Puedo obtener participantes incluso si la reunión ya terminó?
- ¿Necesito el `onlineMeetingId` o el `meetingId` para obtener participantes?
- ¿Hay alguna diferencia entre obtener participantes de una reunión en curso vs una reunión terminada?

**3. Autenticación y Permisos:**
- Con Client Credentials Flow, ¿puedo acceder a las grabaciones de reuniones donde el bot (`cosmos_bot@cosgs.com`) participó?
- ¿El permiso `OnlineMeetingRecording.Read.All` es suficiente o necesito permisos adicionales?
- ¿Necesito usar un flujo de autenticación diferente (delegado vs aplicación)?
- ¿Cómo obtengo el token de acceso correctamente con `msal` usando Client Credentials Flow?

**4. Identificación de Reuniones:**
- Tengo el `onlineMeetingId` extraído de la URL de Teams (ejemplo: `19:meeting_xxx@thread.v2`)
- También tengo un `meetingId` interno del sistema (UUID)
- ¿Cuál debo usar en las llamadas a Graph API?
- ¿Cómo extraigo correctamente el `onlineMeetingId` de diferentes formatos de URL de Teams?

**5. Mejores Prácticas y Consideraciones:**
- ¿Hay límites de rate limiting que deba considerar?
- ¿Cuánto tiempo tarda normalmente una grabación en estar disponible después de que termina la reunión?
- ¿Hay algún webhook o notificación cuando la grabación está lista?
- ¿Cómo manejo errores comunes (grabación no disponible, reunión sin grabación, etc.)?

**6. Ejemplos de Código:**
- ¿Puedes proporcionar ejemplos de código Python usando `httpx` y `msal` para:
  - Autenticarse con Client Credentials Flow
  - Obtener la lista de grabaciones disponibles para una reunión
  - Descargar el contenido de audio de una grabación
  - Obtener la lista de participantes de una reunión

## Información Adicional

**Estructura de Datos que Necesito:**
- Audio: Archivo de audio (preferiblemente WAV o MP3) guardado localmente
- Participantes: Lista de diccionarios con estructura:
  ```python
  [
      {
          "name": "Nombre Completo",
          "email": "email@ejemplo.com",
          "joined_at": "2026-01-28T10:00:00Z",  # opcional
          "left_at": "2026-01-28T11:00:00Z"    # opcional
      },
      ...
  ]
  ```

**Restricciones:**
- No puedo usar SDKs propietarios de Microsoft (solo Graph API REST)
- Debe funcionar con autenticación de aplicación (Client Credentials), no delegado de usuario
- El bot ya está en la reunión, así que no necesito crear reuniones, solo leer datos de reuniones existentes

Por favor, proporciona información detallada sobre los endpoints específicos, ejemplos de código Python, y cualquier consideración importante para implementar esta funcionalidad.
```

## Información Adicional para el Prompt

Si necesitas más contexto específico, puedes agregar:

### Endpoints que ya estoy usando (para referencia):
- Base URL: `https://graph.microsoft.com/v1.0`
- Para autenticación: `https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token`
- Scopes necesarios: `["https://graph.microsoft.com/.default"]`

### Código Actual (para contexto):
```python
# Autenticación con msal
from msal import ConfidentialClientApplication

app = ConfidentialClientApplication(
    client_id=client_id,
    client_credential=client_secret,
    authority=f"https://login.microsoftonline.com/{tenant_id}"
)

token_result = app.acquire_token_for_client(scopes=["https://graph.microsoft.com/.default"])
access_token = token_result["access_token"]
```

### Estructura de Archivos del Proyecto:
- `app/services/teams_bot.py` - Lógica del bot que se une a reuniones
- `app/services/teams_recording_service.py` - Servicio para interactuar con Graph API
- `app/services/cosmos_service.py` - Servicio para enviar audio a procesamiento
- `app/tasks/meeting_tasks.py` - Tareas Celery que orquestan el flujo

### Preguntas Específicas Adicionales:
1. ¿Hay alguna diferencia entre obtener grabaciones de reuniones programadas vs reuniones ad-hoc?
2. ¿Puedo obtener el audio en tiempo real mientras la reunión está en curso, o solo después de que termina?
3. ¿Cómo manejo el caso donde una reunión no tiene grabación habilitada?
4. ¿Hay alguna forma de verificar si una reunión tiene grabación disponible antes de intentar descargarla?

---

## Cómo Usar Este Prompt

1. Copia el contenido del prompt completo (desde "Necesito ayuda..." hasta "...esta funcionalidad.")
2. Pégalo en tu herramienta de IA preferida (ChatGPT, Claude, etc.)
3. La IA debería proporcionarte:
   - Endpoints específicos de Graph API
   - Ejemplos de código Python
   - Información sobre permisos y autenticación
   - Mejores prácticas y consideraciones

## Notas Importantes

- Este prompt está diseñado para obtener información técnica específica sobre Microsoft Graph API
- Incluye todo el contexto necesario para que la IA entienda tu situación específica
- Puedes modificar las preguntas según tus necesidades específicas
- Si obtienes información útil, considera actualizar este documento con los hallazgos
