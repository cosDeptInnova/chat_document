# Cómo Funciona la Integración de Calendarios

Esta guía explica en detalle cómo funciona la integración de Google Calendar y Outlook Calendar en Cosmos NoteTaker.

---

## 📋 Índice

1. [Flujo OAuth y Almacenamiento de Tokens](#flujo-oauth)
2. [Cuándo se Sincroniza](#cuando-se-sincroniza)
3. [Cómo Funcionan los Tokens](#tokens)
4. [Qué Eventos se Sincronizan](#eventos-sincronizados)
5. [Sincronización Automática vs Manual](#sincronizacion)
6. [Renovación de Tokens](#renovacion-tokens)

---

## 🔐 Flujo OAuth y Almacenamiento de Tokens

### Paso 1: Autorización Inicial

1. **Usuario hace clic en "Conectar"** en Google Calendar o Outlook
2. **Redirección a Google/Outlook** para autorizar la aplicación
3. **Usuario autoriza** con su cuenta personal
4. **Google/Outlook devuelve tokens**:
   - **Access Token**: Token de acceso (corta duración, ~1 hora)
   - **Refresh Token**: Token para renovar (larga duración, puede durar meses o indefinidamente si no se revoca)

### Paso 2: Almacenamiento

Los tokens se guardan **encriptados** en la base de datos, en el campo `settings` del usuario:

```json
{
  "google_calendar": {
    "access_token": "ya29.a0AfH6...",
    "refresh_token": "1//0gX...",
    "token_expires_at": "2025-12-20T23:03:07.846989",
    "connected_at": "2025-12-20T22:03:07.846989",
    "calendar_id": "primary"
  }
}
```

**Importante**: Los tokens son **específicos de cada usuario**. Cada usuario tiene sus propios tokens almacenados.

---

## ⏰ Cuándo se Sincroniza

### Sincronización Automática (Inmediata)

La sincronización se ejecuta automáticamente en estos momentos:

1. **Inmediatamente después de conectar**:
   - Cuando el usuario autoriza y se completan los tokens
   - Se sincronizan los eventos de los próximos 30 días

### Sincronización Manual

El usuario puede sincronizar manualmente:

1. **Desde la interfaz**: Botón "Sincronizar ahora" en la página de Integraciones
2. **Desde la API**: Endpoint `/api/integrations/calendar/sync`

### ❌ NO hay Sincronización Automática en Background

**IMPORTANTE**: Actualmente **NO hay sincronización automática programada** que se ejecute:
- ❌ Cuando creas un nuevo evento en Google Calendar
- ❌ En intervalos regulares (cada X minutos)
- ❌ Cuando inicias sesión en el frontend

**¿Por qué?**
- Requeriría una tarea programada (Celery, APScheduler, cron)
- Aumentaría el costo de recursos
- Google Calendar puede notificar cambios vía webhooks (implementación futura)

---

## 🔑 Cómo Funcionan los Tokens

### Access Token

- **Duración**: ~1 hora (3600 segundos)
- **Uso**: Para hacer peticiones a la API de Google Calendar
- **Expiración**: Automática después de 1 hora

### Refresh Token

- **Duración**: Larga duración (meses o indefinida si no se revoca)
- **Uso**: Para obtener un nuevo Access Token cuando el actual expira
- **Expiración**: Solo expira si:
  - El usuario revoca el acceso manualmente
  - La aplicación no se usa por mucho tiempo (6 meses sin uso)
  - Hay cambios de seguridad en la cuenta

### Renovación Automática

El sistema **renueva automáticamente** el Access Token cuando:
1. **Está por expirar** (menos de 5 minutos restantes)
2. **Está expirado** y se intenta usar

**Proceso de renovación**:
```python
# Si el token expira en menos de 5 minutos
if time_until_expiry < 300:
    # Usar refresh_token para obtener nuevo access_token
    new_token = refresh_token(refresh_token)
    # Guardar nuevo token en base de datos
    user.settings["google_calendar"]["access_token"] = new_token
```

**No necesitas hacer nada**: El sistema lo hace automáticamente.

---

## 📅 Qué Eventos se Sincronizan

### Eventos que SE sincronizan

La sincronización busca eventos de los **próximos 30 días** y solo crea reuniones para eventos que:

1. **Tienen una URL de reunión** detectada en:
   - **Google Calendar**:
     - `hangoutLink` (Google Meet)
     - `conferenceData.entryPoints` (enlaces de video)
     - Descripción con URLs de Teams, Zoom, Google Meet
   - **Outlook Calendar**:
     - `onlineMeeting.joinUrl` (Teams)
     - Contenido del body con URLs de reuniones
     - Location con URLs

2. **No existen ya en la base de datos**:
   - Verifica si ya hay una reunión con esa URL y ese organizador
   - Evita duplicados

### Eventos que NO se sincronizan

- ❌ Eventos sin URL de reunión
- ❌ Eventos que ya existen en la base de datos
- ❌ Eventos más antiguos de 30 días
- ❌ Eventos sin fecha/hora de inicio

---

## 🔄 Sincronización Automática vs Manual

### Sincronización Automática (Actual)

**Cuándo se ejecuta**:
- ✅ Inmediatamente después de conectar OAuth (una sola vez)

**Qué hace**:
- Obtiene eventos de los próximos 30 días
- Crea reuniones para eventos con URLs de video

**Limitaciones**:
- ❌ Solo se ejecuta una vez al conectar
- ❌ No detecta nuevos eventos creados después
- ❌ No detecta cambios en eventos existentes

### Sincronización Manual

**Cuándo se ejecuta**:
- Cuando el usuario hace clic en "Sincronizar ahora"

**Qué hace**:
- Igual que la automática
- Busca eventos de los próximos 30 días
- Crea reuniones nuevas (evita duplicados)

### Sincronización Automática Futura (Implementación Recomendada)

Para tener sincronización automática, podrías implementar:

#### Opción A: Tarea Programada (Celery/APScheduler)
```python
# Ejecutar cada 15 minutos
@celery.task
def sync_all_calendars():
    users = get_users_with_calendar_integrations()
    for user in users:
        sync_user_calendars(user)
```

#### Opción B: Webhooks de Google Calendar
```python
# Google notifica cuando hay cambios
@router.post("/webhooks/google-calendar")
async def google_calendar_webhook():
    # Google envía notificación de cambio
    # Sincronizar solo ese calendario
```

#### Opción C: Sincronización al Iniciar Sesión
```python
# En el endpoint de login
async def login():
    # ... autenticación ...
    if user.has_calendar_integration:
        sync_user_calendars(user)  # En background
```

---

## 🔄 Renovación de Tokens

### ¿Cuándo expira el Refresh Token?

El Refresh Token **NO expira automáticamente** a menos que:

1. **Usuario revoca acceso**:
   - Va a Google Account > Security > Third-party apps
   - Revoca el acceso a tu aplicación

2. **Inactividad prolongada**:
   - Si la app no se usa por 6 meses
   - Google puede revocar tokens inactivos

3. **Cambios de seguridad**:
   - Cambio de contraseña
   - Detección de actividad sospechosa

### ¿Qué pasa si el Refresh Token expira?

Si el Refresh Token expira:

1. **La próxima vez que intentes sincronizar**:
   - El sistema intenta renovar el Access Token
   - Falla porque el Refresh Token es inválido
   - Retorna error

2. **Solución**:
   - El usuario debe **volver a conectar** Google Calendar
   - Se obtienen nuevos tokens

### ¿Necesitas renovar manualmente?

**NO**. El sistema renueva automáticamente:
- ✅ Access Token se renueva automáticamente cuando expira
- ✅ Refresh Token se usa automáticamente para renovar
- ✅ Todo funciona en background

**El usuario no necesita hacer nada**.

---

## 📊 Resumen del Flujo Completo

### Escenario 1: Usuario Conecta Google Calendar

```
1. Usuario hace clic en "Conectar Google Calendar"
   ↓
2. Redirección a Google → Usuario autoriza
   ↓
3. Google devuelve tokens → Backend los guarda
   ↓
4. Sincronización automática (próximos 30 días)
   ↓
5. Se crean reuniones para eventos con URLs de video
   ↓
6. Usuario ve las reuniones en NoteTaker
```

### Escenario 2: Usuario Crea Nuevo Evento en Google Calendar

```
1. Usuario crea evento en Google Calendar (con URL de Teams)
   ↓
2. ❌ NO se sincroniza automáticamente (no hay webhook/tarea programada)
   ↓
3. Usuario hace clic en "Sincronizar ahora" en NoteTaker
   ↓
4. Sistema busca eventos de próximos 30 días
   ↓
5. Encuentra el nuevo evento → Crea reunión en NoteTaker
```

### Escenario 3: Token Expira

```
1. Usuario intenta sincronizar (o se ejecuta automáticamente)
   ↓
2. Sistema detecta que Access Token expiró (o está por expirar)
   ↓
3. Usa Refresh Token para obtener nuevo Access Token
   ↓
4. Guarda nuevo Access Token en base de datos
   ↓
5. Continúa con la sincronización normalmente
```

---

## ❓ Preguntas Frecuentes

### ¿Se sincroniza automáticamente cuando creo un evento?

**NO**. Actualmente solo se sincroniza:
- Una vez al conectar
- Cuando haces clic en "Sincronizar ahora"

Para sincronización automática, necesitarías implementar una de las opciones mencionadas arriba.

### ¿Cuánto tiempo duran los tokens?

- **Access Token**: ~1 hora (se renueva automáticamente)
- **Refresh Token**: Indefinido (o hasta que el usuario revoque)

### ¿Necesito volver a autorizar?

Solo si:
- Revocas el acceso manualmente en Google
- El Refresh Token expira (muy raro)
- Hay un problema con los tokens guardados

### ¿Funciona sin estar logueado en el frontend?

**Sí, los tokens están en el backend**. El backend puede sincronizar independientemente de si el usuario está logueado o no, pero actualmente solo se ejecuta cuando:
- El usuario conecta (una vez)
- El usuario hace clic en "Sincronizar ahora"

---

## 🚀 Mejoras Futuras Recomendadas

1. **Sincronización programada** (cada 15-30 minutos)
2. **Webhooks de Google Calendar** (notificaciones en tiempo real)
3. **Sincronización al iniciar sesión**
4. **Sincronización incremental** (solo cambios desde última sincronización)
5. **Notificaciones al usuario** cuando se crea una nueva reunión

---

## 📝 Notas Técnicas

- Los tokens se almacenan en `user.settings` (JSON) en la base de datos
- Se usa `flag_modified()` para que SQLAlchemy detecte cambios en JSON
- Los tokens se renuevan automáticamente 5 minutos antes de expirar
- La sincronización busca eventos de los próximos 30 días
- Se evitan duplicados comparando `meeting_url` y `organizer_email`

