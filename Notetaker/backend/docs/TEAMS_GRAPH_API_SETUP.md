# Configuración de Teams Bot con Microsoft Graph API

Este documento explica cómo configurar el bot de Teams para usar Microsoft Graph API en lugar de Playwright.

## Requisitos Previos

1. **Aplicación registrada en Azure AD**
   - Debes tener una aplicación registrada en Azure Portal
   - Necesitas el `Client ID` y `Client Secret`

2. **Permisos de Graph API requeridos**
   - `Calls.JoinGroupCall.All` (o `Calls.JoinGroupCallAsGuest.All`)
   - `Calls.Initiate.All`
   - `Calls.AccessMedia.All`
   - `OnlineMeetings.ReadWrite.All`
   - `OnlineMeetingTranscript.Read.All`
   - `OnlineMeetingRecording.Read.All`
   - `OnlineMeetingTranscript.Read.Chat` (RSC, si app-only)
   - `OnlineMeetingRecording.Read.Chat` (RSC, si app-only)

3. **Bot registrado en Microsoft Teams**
   - El bot debe estar registrado en Teams con `supportsCalling` y `supportsVideo` habilitados
   - Necesitas configurar un webhook URL para recibir notificaciones

## Configuración

### Variables de Entorno

Añade las siguientes variables a tu archivo `.env`:

```env
# Microsoft Graph API para el bot
OUTLOOK_BOT_CLIENT_ID=tu-client-id
OUTLOOK_BOT_CLIENT_SECRET=tu-client-secret
OUTLOOK_BOT_REDIRECT_URI=https://tu-dominio.com/api/integrations/oauth/callback/outlook-bot

# O usa las variables generales si no tienes configuración específica del bot
OUTLOOK_CLIENT_ID=tu-client-id
OUTLOOK_CLIENT_SECRET=tu-client-secret

# Tenant ID (opcional, puede ser "common" para multi-tenant)
OUTLOOK_TENANT_ID=tu-tenant-id

# URL pública del backend para webhooks (requerido para suscripciones)
BACKEND_PUBLIC_URL=https://tu-dominio.com

# Graph API general (para otros servicios)
GRAPH_TENANT_ID=tu-tenant-id
GRAPH_CLIENT_ID=tu-client-id
GRAPH_CLIENT_SECRET=tu-client-secret
```

### Uso del Servicio

#### Modo Graph API (Recomendado)

```python
from app.services.teams_bot import TeamsBotService

# Crear bot con Graph API habilitado
bot = TeamsBotService(use_graph_api=True)

# Unirse a una reunión
await bot.join_meeting(
    meeting_url="https://teams.microsoft.com/l/meetup-join/...",
    meeting_id="meeting-123",
    bot_display_name="Notetaker Bot",
    access_token=access_token,  # Opcional si ya tienes token
    refresh_token=refresh_token,  # Opcional
    client_id=client_id,  # Opcional, usa config si no se proporciona
    client_secret=client_secret,  # Opcional
)

# Salir de la reunión
await bot.leave_meeting(
    reason="manual",
    access_token=access_token,  # Opcional
    refresh_token=refresh_token,  # Opcional
    client_id=client_id,  # Opcional
    client_secret=client_secret,  # Opcional
)
```

#### Modo Playwright (Legacy)

```python
# Crear bot con Playwright (modo legacy)
bot = TeamsBotService(use_graph_api=False)

# El resto del código funciona igual
await bot.join_meeting(
    meeting_url="https://teams.microsoft.com/l/meetup-join/...",
    meeting_id="meeting-123",
    bot_display_name="Notetaker Bot",
)
```

## Obtener Transcripciones y Grabaciones

Después de que el bot se una a la reunión y esta termine, puedes obtener las transcripciones y grabaciones:

```python
from app.services.teams_recording_service import TeamsRecordingService

recording_service = TeamsRecordingService()

# Obtener onlineMeetingId desde la URL
meeting_id = recording_service.get_online_meeting_id_from_url(meeting_url)

# Listar transcripciones disponibles
transcripts = await recording_service.list_meeting_transcripts(
    access_token=access_token,
    refresh_token=refresh_token,
    client_id=client_id,
    client_secret=client_secret,
    meeting_id=meeting_id,
    user_id=user_id,  # Opcional, ID del organizador
)

# Descargar transcripción (VTT)
if transcripts:
    transcript_id = transcripts[0]["id"]
    await recording_service.download_transcript_content(
        access_token=access_token,
        refresh_token=refresh_token,
        client_id=client_id,
        client_secret=client_secret,
        meeting_id=meeting_id,
        transcript_id=transcript_id,
        output_path=Path("./transcript.vtt"),
    )

# Descargar metadata de transcripción (con speakers)
metadata_text = await recording_service.download_transcript_metadata(
    access_token=access_token,
    refresh_token=refresh_token,
    client_id=client_id,
    client_secret=client_secret,
    meeting_id=meeting_id,
    transcript_id=transcript_id,
)

# Parsear metadata
parsed_segments = recording_service.parse_metadata_vtt(metadata_text)
for segment in parsed_segments:
    print(f"{segment['speakerName']}: {segment['spokenText']}")
```

## Suscripciones a Notificaciones

Para recibir notificaciones cuando las transcripciones/grabaciones estén disponibles:

```python
from app.services.teams_subscription_service import TeamsSubscriptionService

subscription_service = TeamsSubscriptionService()

# Suscribirse a transcripciones y grabaciones
subscriptions = await subscription_service.subscribe_to_meeting(
    meeting_id=online_meeting_id,
    notification_url="https://tu-dominio.com/api/webhooks/teams/notifications",
    expiration_hours=3,
    access_token=access_token,
    refresh_token=refresh_token,
    client_id=client_id,
    client_secret=client_secret,
)
```

## Endpoints de Webhook

Necesitas crear endpoints en tu aplicación para recibir:

1. **Notificaciones de llamadas** (`/api/webhooks/teams/call`)
   - Recibe notificaciones cuando hay llamadas entrantes
   - Debes responder con `answer` para aceptar la llamada

2. **Notificaciones de transcripciones/grabaciones** (`/api/webhooks/teams/notifications`)
   - Recibe notificaciones cuando las transcripciones/grabaciones están disponibles
   - Debes validar el token de verificación de Microsoft

## Limitaciones

1. **Solo reuniones programadas**: Algunos endpoints solo funcionan con reuniones calendarizadas
2. **Disponibilidad**: Las transcripciones/grabaciones solo están disponibles DESPUÉS de que termine la reunión
3. **Permisos**: Requiere admin consent para los permisos de aplicación
4. **Webhooks**: Requiere una URL pública HTTPS para recibir notificaciones

## Migración desde Playwright

Para migrar de Playwright a Graph API:

1. Configura las variables de entorno necesarias
2. Cambia `TeamsBotService(use_graph_api=False)` a `TeamsBotService(use_graph_api=True)`
3. Asegúrate de tener los permisos de Graph API configurados
4. Configura los webhooks necesarios
5. Actualiza el código para manejar las transcripciones/grabaciones después de la reunión

## Troubleshooting

### Error: "Se requiere OUTLOOK_BOT_CLIENT_ID y OUTLOOK_BOT_CLIENT_SECRET"
- Asegúrate de configurar estas variables en `.env`
- O configura `OUTLOOK_CLIENT_ID` y `OUTLOOK_CLIENT_SECRET` como fallback

### Error: "No se pudo extraer información de la URL"
- Verifica que la URL de Teams sea válida
- Asegúrate de que la reunión esté programada (no ad-hoc)

### Error: "Error HTTP uniéndose a la reunión: 403"
- Verifica que los permisos de Graph API estén configurados correctamente
- Asegúrate de que el admin haya dado consentimiento a la aplicación

### Las transcripciones no están disponibles
- Las transcripciones solo están disponibles después de que termine la reunión
- Puede haber un delay de varios minutos
- Asegúrate de que la grabación/transcripción esté habilitada en la reunión
