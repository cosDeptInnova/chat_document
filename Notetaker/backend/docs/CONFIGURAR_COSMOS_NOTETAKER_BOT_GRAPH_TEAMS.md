# Configuración exacta: Cosmos Notetaker Bot para Graph API y Teams (llamadas y transcripciones)

Esta guía detalla cómo configurar la aplicación **Cosmos Notetaker Bot** que ya tienes en Azure AD (creada según [GUIA_INTEGRAR_OUTLOOK_CALENDAR_BOT.md](../../GUIA_INTEGRAR_OUTLOOK_CALENDAR_BOT.md)) para:

1. Añadir los permisos necesarios para unirse a reuniones y usar transcripciones/grabaciones vía Graph API.
2. Registrar el bot en Teams con `supportsCalling` y `supportsVideo`.
3. Configurar la URL de webhook para notificaciones de llamadas y de transcripciones/grabaciones.

---

## Parte 1: Permisos en Azure AD (app Cosmos Notetaker Bot)

La app **Cosmos Notetaker Bot** es la que usas como **Aplicación del Bot** (`OUTLOOK_BOT_CLIENT_ID` / `OUTLOOK_BOT_CLIENT_SECRET`). Según la guía original ya tiene:

- `User.Read` (Delegado)
- `OnlineMeetingRecording.Read.All` (Delegado, con consentimiento de administrador)

Debes **añadir** los siguientes permisos en esa misma app.

### 1.1 Tabla de permisos a añadir

| Permiso | Tipo | Requiere consentimiento de administrador | Para qué se usa |
|--------|------|------------------------------------------|-----------------|
| `Calls.JoinGroupCall.All` | **Aplicación** | **Sí** | Unirse a reuniones de Teams como bot (llamadas de grupo). |
| `Calls.JoinGroupCallAsGuest.All` | **Aplicación** | **Sí** | Alternativa: unirse como invitado si no está en el tenant. |
| `Calls.Initiate.All` | **Aplicación** | **Sí** | Crear/iniciar llamadas (necesario para POST /communications/calls). |
| `Calls.AccessMedia.All` | **Aplicación** | **Sí** | Acceso a flujos de audio/video en la llamada (grabación en tiempo real). |
| `OnlineMeetings.ReadWrite.All` | **Aplicación** | **Sí** | Leer/escribir reuniones en línea (coordenadas de reunión para unirse). |
| `OnlineMeetingTranscript.Read.All` | **Delegado** | **Sí** | Leer transcripciones de reuniones (post-reunión) en nombre del usuario. |
| `OnlineMeetingRecording.Read.All` | *(ya lo tienes)* | **Sí** | Leer grabaciones (post-reunión). |

**Resumen:**

- **Tipo Aplicación** (Application): la app actúa por sí misma (sin usuario firmado). Todos los de Communications (Calls.*, OnlineMeetings.ReadWrite.All) son de aplicación y **requieren admin**.
- **Tipo Delegado** (Delegated): la app actúa en nombre del usuario/bot que autorizó. `OnlineMeetingTranscript.Read.All` y `OnlineMeetingRecording.Read.All` son delegados y **requieren admin** para poder consentirlos en la organización.

### 1.2 Pasos exactos en Azure Portal (Cosmos Notetaker Bot)

1. Ve a [Azure Portal](https://portal.azure.com/) → **Microsoft Entra ID** (o Azure Active Directory) → **Registros de aplicaciones**.
2. Abre la aplicación **Cosmos Notetaker Bot** (la del Bot, no la de usuarios).
3. Menú **Permisos de API** → **+ Añadir un permiso**.
4. Elige **Microsoft Graph**.
5. **Tipo de permiso** (importante):
   - Para permisos de **Aplicación**: en la pestaña que aparece, selecciona **Permisos de aplicación** (no "Permisos delegados"). Busca y marca los que tengan tipo "Aplicación" en la tabla anterior.
   - Para permisos **Delegados**: selecciona **Permisos delegados** y marca `OnlineMeetingTranscript.Read.All` (el de grabaciones ya lo tienes).
6. Añade en bloques para no equivocarte:
   - Primera petición: **Permisos de aplicación** → marcar `Calls.JoinGroupCall.All`, `Calls.Initiate.All`, `Calls.AccessMedia.All`, `OnlineMeetings.ReadWrite.All` (y si quieres `Calls.JoinGroupCallAsGuest.All`) → **Añadir permisos**.
   - Segunda petición: **Permisos delegados** → marcar `OnlineMeetingTranscript.Read.All` → **Añadir permisos**.

**Permisos de aplicación (Application):**

- `Calls.JoinGroupCall.All`
- `Calls.JoinGroupCallAsGuest.All` (opcional si solo unes al bot dentro de tu tenant)
- `Calls.Initiate.All`
- `Calls.AccessMedia.All`
- `OnlineMeetings.ReadWrite.All`

**Permisos delegados (Delegated):**

- `OnlineMeetingTranscript.Read.All` (añadir si no está; `OnlineMeetingRecording.Read.All` ya lo tienes).

7. En **Permisos de API**, pulsa **Conceder consentimiento de administrador para [tu organización]**.
   - Sin este paso, los permisos que requieren admin no estarán realmente concedidos y las llamadas/transcripciones fallarán.
8. Comprueba que en la tabla de permisos todos aparecen como **Concedido para [organización]** (icono verde).

### 1.3 No cambiar lo que ya tienes

- **URI de redirección**: mantener el que ya usas para OAuth del bot (por ejemplo `.../api/integrations/oauth/callback/outlook`).
- **Certificados y secretos**: seguir usando el mismo Client Secret que ya tienes como `OUTLOOK_BOT_CLIENT_SECRET` (o crear uno nuevo si lo necesitas y actualizar `.env`).

---

## Parte 2: Registrar el bot en Teams (supportsCalling y supportsVideo)

Para que Teams reconozca tu app como bot que puede unirse a **llamadas y reuniones**, la app de Teams debe declarar un bot con `supportsCalling` y `supportsVideo` en true. Eso se hace en el **manifest** de la app de Teams.

### 2.1 Opción A: Crear/editar el manifest manualmente (recomendado para control total)

1. Crea o edita el archivo del manifest de Teams (por ejemplo `manifest.json` en la raíz del proyecto o en una carpeta `teams/`).

2. Esquema recomendado (para que el editor valide bien):

   ```json
   {
     "$schema": "https://developer.microsoft.com/json-schemas/teams/v1.16/MicrosoftTeams.schema.json",
     "manifestVersion": "1.16",
     "version": "1.0.0",
     "id": "<TU_APP_ID_DE_TEAMS_O_MISMO_QUE_OUTLOOK_BOT_CLIENT_ID>",
     "packageName": "com.cosmos.notetakerbot",
     "developer": {
       "name": "Tu organización",
       "websiteUrl": "https://tu-dominio.com",
       "privacyUrl": "https://tu-dominio.com/privacy",
       "termsOfUseUrl": "https://tu-dominio.com/terms"
     },
     "name": {
       "short": "Cosmos Notetaker Bot",
       "full": "Cosmos Notetaker Bot - Transcripciones y grabaciones"
     },
     "description": {
       "short": "Bot para reuniones y transcripciones",
       "full": "Bot que se une a reuniones de Teams para transcripciones y grabaciones vía Graph API."
     },
     "icons": {
       "outline": "outline.png",
       "color": "color.png"
     },
     "accentColor": "#FFFFFF",
     "bots": [
       {
         "botId": "<OUTLOOK_BOT_CLIENT_ID>",
         "scopes": [
           "team",
           "personal",
           "groupChat"
         ],
         "supportsFiles": false,
         "isNotificationOnly": false,
         "commandLists": [],
         "supportsCalling": true,
         "supportsVideo": true
       }
     ],
     "validDomains": [
       "*.ngrok-free.app",
       "*.ngrok.io",
       "tu-dominio.com"
     ]
   }
   ```

3. Sustituye:
   - `botId`: por el **Application (client) ID** de la app **Cosmos Notetaker Bot** en Azure (el mismo que `OUTLOOK_BOT_CLIENT_ID`).
   - `id` (arriba del manifest): puede ser el mismo que `botId` o un GUID que identifique la app de Teams.
   - `validDomains`: por los dominios donde esté alojado tu backend (ngrok, producción, etc.).

4. Iconos: en la misma carpeta que el manifest, coloca `outline.png` (20x20) y `color.png` (96x96). Si no los tienes, puedes usar placeholders y luego sustituirlos.

5. Empaquetado:
   - Comprime en un `.zip` el `manifest.json` y los dos iconos (sin carpeta intermedia: el ZIP debe contener directamente `manifest.json`, `outline.png`, `color.png`).

6. Instalación en Teams:
   - **Teams (cliente)** → **Apps** → **Administrar las aplicaciones** (o **Cargar una aplicación personalizada**) → **Cargar una aplicación personalizada** → selecciona el `.zip`.
   - O bien en [Teams Admin Center](https://admin.teams.microsoft.com/) → **Teams apps** → **Manage apps** → **Upload** (según política de tu tenant).

Con esto el bot queda registrado en Teams con **supportsCalling** y **supportsVideo** habilitados.

### 2.2 Opción B: Teams Developer Portal

1. Entra en [Teams Developer Portal](https://dev.teams.microsoft.com/).
2. Inicia sesión con la cuenta del tenant donde quieres instalar el bot.
3. **Apps** → **+ Nueva aplicación** (o importar una existente si ya tienes manifest).
4. En la configuración del bot:
   - **Bot ID**: el mismo **Application (client) ID** de **Cosmos Notetaker Bot** (`OUTLOOK_BOT_CLIENT_ID`).
   - Activa **Supports calling** y **Supports video** (equivalente a `supportsCalling: true` y `supportsVideo: true`).
5. Guarda y publica/distribuye la app (para pruebas puedes usar “Install in Teams” o “Add to team” según lo que ofrezca el portal).

En ambos casos, el **Bot ID** debe ser exactamente el Client ID de la app de Azure **Cosmos Notetaker Bot**.

---

## Parte 3: Webhook URL para notificaciones

Hay dos usos distintos de “webhook” en tu escenario: **llamadas** (Graph Communications) y **suscripciones** (transcripciones/grabaciones).

### 3.1 URL de callback para llamadas (Graph Communications API)

Cuando tu backend **crea** la llamada para unirse a la reunión (POST `/communications/calls`), envía en el cuerpo el **callbackUri**. Microsoft usará esa URL para enviar eventos de la llamada (estado, participantes, etc.).

- **Dónde se configura**: en el código del backend, no en Azure. En tu proyecto ya se usa `settings.backend_public_url` para construir la URL.
- Ejemplo en `teams_graph_service.py` (o equivalente):

  ```text
  callbackUri = f"{settings.backend_public_url}/api/webhooks/teams/call"
  ```

- **Qué debes tener**:
  1. Variable de entorno `BACKEND_PUBLIC_URL` con la URL pública HTTPS del backend (por ejemplo `https://tu-dominio.com` o `https://xxx.ngrok-free.app`).
  2. Un endpoint en tu backend que responda a POST en `/api/webhooks/teams/call` (o la ruta que uses) y que:
     - Valide la petición (token/firma si aplica).
     - Procese las notificaciones de estado de llamada (por ejemplo, para saber cuándo el bot está en la reunión o cuándo debe colgar).

No se configura en el registro de la app en Azure; la URL la define tu código y debe ser accesible desde internet (HTTPS).

### 3.2 URL de notificaciones para transcripciones y grabaciones (suscripciones Graph)

Para recibir avisos cuando una transcripción o grabación esté disponible, creas una **suscripción** de Graph (por ejemplo en `TeamsSubscriptionService`) y en esa suscripción indicas una **notificationUrl**.

- **Dónde se configura**: también en el código, al llamar al servicio de suscripciones; por ejemplo:

  ```text
  notification_url = f"{settings.backend_public_url}/api/webhooks/teams/notifications"
  ```

- **Qué debes tener**:
  1. La misma `BACKEND_PUBLIC_URL` (HTTPS).
  2. Un endpoint (por ejemplo `POST /api/webhooks/teams/notifications`) que:
     - Responda al **validation request** de Microsoft (GET con `validationToken` en query) devolviendo ese mismo `validationToken` en texto plano.
     - Procese las notificaciones POST con los cambios (transcripción/grabación disponible) y, si la suscripción lo requiere, renueve la suscripción antes de que expire.

De nuevo, esta URL no se configura en Azure; la define tu backend y debe ser HTTPS y pública.

### 3.3 (Opcional) Azure Bot Service – Messaging endpoint

Si en el futuro usas **Azure Bot Service** (Bot Framework) para que Teams envíe mensajes o eventos al bot (por ejemplo, cuando un usuario añade el bot a un equipo o le escribe), ahí sí configurarías un **Messaging endpoint** en el recurso “Bot” de Azure:

- Recurso **Bot** en Azure → **Configuración** → **Messaging endpoint**: por ejemplo `https://tu-dominio.com/api/bot/messages` (o la ruta que expongas para el Bot Framework).

Para el flujo actual (unirse a reuniones vía Graph API y recibir transcripciones/grabaciones vía suscripciones), **no es obligatorio** tener Azure Bot Service; basta con la app registrada en Azure AD y el manifest de Teams con `supportsCalling`/`supportsVideo` y las URLs de callback/notificaciones en tu código.

---

## Resumen de comprobaciones

| Qué | Dónde | Valor / Acción |
|-----|--------|----------------|
| Permisos de aplicación (Calls.*, OnlineMeetings.ReadWrite.All) | Azure AD → Cosmos Notetaker Bot → Permisos de API | Añadidos y **consentimiento de administrador** concedido |
| Permisos delegados (Transcript/Recording) | Igual | `OnlineMeetingTranscript.Read.All` añadido; admin concedido |
| supportsCalling / supportsVideo | Manifest de Teams (o Developer Portal) | `true` en el bot con `botId` = OUTLOOK_BOT_CLIENT_ID |
| Callback llamadas | Código + .env | `BACKEND_PUBLIC_URL` + endpoint `/api/webhooks/teams/call` |
| Notificaciones transcripciones/grabaciones | Código + .env | Misma `BACKEND_PUBLIC_URL` + endpoint `/api/webhooks/teams/notifications` |

Con esto tienes la app **Cosmos Notetaker Bot** configurada en Azure con los permisos correctos (aplicación y delegados, con admin donde toca), el bot registrado en Teams con llamadas y vídeo habilitados, y las URLs de webhook definidas y listas para implementar en tu backend.
