# Uso de VEXA como teams_bot para reuniones y transcripciones

Este documento describe cómo usar VEXA (instalado en red local) para meter un bot en reuniones de Microsoft Teams y obtener transcripciones, sustituyendo el enfoque anterior (Playwright, Recall.ai o API de Teams).

## Resumen de endpoints

| Uso | Método | URL base | Autenticación |
|-----|--------|----------|---------------|
| Crear bot (unir a reunión) | POST | API Gateway (8056) | `X-API-Key` |
| Obtener transcripción (REST) | GET | API Gateway (8056) | `X-API-Key` |
| Transcripción en tiempo real | WebSocket | API Gateway (8056) | `X-API-Key` |
| Admin (usuarios/tokens) | - | Admin API (8057) | `X-Admin-API-Token` o según doc VEXA |

Con tu instalación:

- **IP VEXA:** `172.29.14.10`
- **API Gateway (User API):** `http://172.29.14.10:8056` — bots y transcripciones
- **Admin API:** `http://172.29.14.10:8057` — gestión de usuarios y tokens
- **Transcription Collector:** `http://172.29.14.10:8123` — uso interno de VEXA

La API que usa notetaker2.0 es la **User API (8056)** con header `X-API-Key`. El token de admin (`ADMIN_API_TOKEN`) sirve para la Admin API; en muchos despliegues self-hosted puedes crear un usuario y obtener un API key para usar como `X-API-Key` en la User API (ver guía self-hosted de VEXA si la tienes).

---

## 1. Meter el bot en una reunión de Teams

**Endpoint:** `POST /bots`  
**URL completa:** `http://172.29.14.10:8056/bots`  
**Headers:**

- `X-API-Key: <VEXA_API_KEY>`
- `Content-Type: application/json`

**Body (Teams):**

```json
{
  "platform": "teams",
  "native_meeting_id": "<ID_NUMERICO_REUNION>",
  "passcode": "<CODIGO_DE_ACCESO_REUNION>"
}
```

- `native_meeting_id`: ID numérico de la reunión de Teams (solo dígitos).
- `passcode`: Código de acceso/PIN de la reunión si la reunión lo requiere (si no hay, se puede omitir o enviar vacío según documentación VEXA).

**Ejemplo con curl:**

```bash
curl -X POST "http://172.29.14.10:8056/bots" \
  -H "X-API-Key: TU_VEXA_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"platform\": \"teams\", \"native_meeting_id\": \"9387167464734\", \"passcode\": \"qxJanYOcdjN4d6UlGa\"}"
```

La respuesta incluirá información del bot creado (por ejemplo `id` de reunión/bot). Ese bot se une a la reunión; hay que admitir el bot en la reunión si hay sala de espera.

---

## 2. Obtener transcripción (REST)

**Endpoint:** `GET /transcripts/{platform}/{native_id}`  
**URL completa (ejemplo):** `http://172.29.14.10:8056/transcripts/teams/9387167464734`  
**Query opcional:** `meeting_id=<id_interno>` si VEXA devuelve un `meeting_id` y lo necesitas para discriminar.

**Headers:**

- `X-API-Key: <VEXA_API_KEY>`

**Ejemplo con curl:**

```bash
curl -X GET "http://172.29.14.10:8056/transcripts/teams/9387167464734" \
  -H "X-API-Key: TU_VEXA_API_KEY"
```

**Ejemplo de respuesta (según doc VEXA):**

```json
{
  "segments": [
    {
      "text": "Hello everyone",
      "speaker": "John",
      "absolute_start_time": "2025-01-15T10:30:00Z",
      "absolute_end_time": "2025-01-15T10:30:03Z"
    }
  ]
}
```

Puedes llamar a este endpoint periódicamente mientras la reunión está en curso o una vez finalizada para obtener la transcripción acumulada.

---

## 3. Transcripción en tiempo real (WebSocket)

Para recibir transcripciones en vivo sin hacer polling a REST:

- **URL WebSocket:** `ws://172.29.14.10:8056/ws`
- **Autenticación:** header `X-API-Key: <VEXA_API_KEY>` (según implementación del cliente WebSocket).

Tras conectar:

1. **Bootstrap inicial:** obtener última transcripción vía REST (sección 2) para no perder segmentos ya emitidos.
2. **Suscribirse a la reunión:** enviar por WebSocket:

```json
{
  "action": "subscribe",
  "meetings": [
    {
      "platform": "teams",
      "native_id": "9387167464734"
    }
  ]
}
```

3. El servidor enviará mensajes de tipo `transcript.mutable` con nuevos segmentos y `meeting.status` con estados (`requested`, `joining`, `awaiting_admission`, `connecting`, `active`, `stopping`, `completed`, `failed`).

Detalle completo del protocolo y formatos en la documentación de VEXA (websocket.md).

---

## 4. Configuración en notetaker2.0

En `.env` (o variables de entorno) se usan:

- `VEXA_API_BASE_URL`: base de la User API, ej. `http://172.29.14.10:8056`
- `VEXA_API_KEY`: API key para `X-API-Key` (User API). Si en tu VEXA self-hosted solo tienes `ADMIN_API_TOKEN`, en algunos despliegues se usa el mismo valor aquí; si no, hay que crear un usuario y token desde la Admin API (puerto 8057) y usar ese token como `VEXA_API_KEY`.

Opcional, solo si vas a usar Admin API desde notetaker2.0:

- `VEXA_ADMIN_API_URL`: `http://172.29.14.10:8057`
- `VEXA_ADMIN_API_TOKEN`: valor de `ADMIN_API_TOKEN` de tu `.env` de VEXA

El servicio `vexa_service.py` usa `VEXA_API_BASE_URL` y `VEXA_API_KEY` para:

1. Llamar a `POST /bots` (meter bot en reunión Teams).
2. Llamar a `GET /transcripts/teams/{native_id}` (obtener transcripciones).

---

## 5. Obtención de `native_meeting_id` y `passcode` desde Teams

- El **native_meeting_id** es el ID numérico de la reunión (solo dígitos). Puede aparecer en la URL de la reunión o en la convocatoria de Teams (por ejemplo en “ID de reunión”).
- El **passcode** es el PIN/código de acceso de la reunión, si existe.

Si en notetaker2.0 solo tienes `meeting_url` de Teams, tendrás que parsear la URL o usar la API de Graph/Calendario donde tengas el `onlineMeeting` con `id` y `joinInformation.password` (passcode) para rellenar `native_meeting_id` y `passcode` en las llamadas anteriores.

---

## Referencia rápida de URLs con tu IP

| Acción | URL |
|--------|-----|
| Crear bot (Teams) | `POST http://172.29.14.10:8056/bots` |
| Transcripción REST | `GET http://172.29.14.10:8056/transcripts/teams/{native_id}` |
| WebSocket (tiempo real) | `ws://172.29.14.10:8056/ws` |
| Docs API (si está habilitado) | `http://172.29.14.10:8056/docs` |
| Admin API (gestión) | `http://172.29.14.10:8057` |

Documentación oficial VEXA: deployment.md, README, websocket.md (en tu carpeta de Descargas).
