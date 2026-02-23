# Guía: Crear Aplicaciones en Azure AD para Outlook Calendar y Grabaciones de Teams

Esta guía te explica cómo crear **DOS aplicaciones separadas** en Azure AD:
1. **Aplicación para Usuarios**: Sincronizar calendarios de usuarios (solo lectura)
2. **Aplicación para el Bot**: Capturar audio de reuniones de Teams (acceso a grabaciones)

## ⚠️ IMPORTANTE: Dos Aplicaciones Separadas

**Necesitas crear DOS aplicaciones separadas** en Azure AD con credenciales diferentes:

- **Aplicación de Usuarios** (`OUTLOOK_CLIENT_ID` / `OUTLOOK_CLIENT_SECRET`):
  - Permisos: `Calendars.Read` + `User.Read`
  - Los usuarios normales autorizan esta app para leer sus calendarios
  - **NO incluye** `OnlineMeetingRecording.Read.All` (los usuarios no lo necesitan)

- **Aplicación del Bot** (`OUTLOOK_BOT_CLIENT_ID` / `OUTLOOK_BOT_CLIENT_SECRET`):
  - Permisos: `OnlineMeetingRecording.Read.All` + `User.Read`
  - Solo el bot autoriza esta app para acceder a grabaciones de Teams
  - **NO incluye** `Calendars.Read` (el bot no necesita leer calendarios)

**¿Por qué separar?**
- Los usuarios normales **NO deben ver** el permiso `OnlineMeetingRecording.Read.All` en la pantalla de consentimiento
- Solo el bot necesita acceso a grabaciones
- Más seguro: cada aplicación tiene solo los permisos mínimos necesarios
- Mejor experiencia de usuario: los usuarios solo ven permisos relevantes para ellos

---

## 🌐 Configuración con ngrok (Desarrollo)

Si usas **ngrok** para exponer tu aplicación local a internet (necesario para que Microsoft pueda redirigir después de la autorización OAuth):

### Requisitos previos
- Tener ngrok instalado y corriendo
- Conocer tu URL de ngrok (ejemplo: `https://untensible-unlicentiously-shaunta.ngrok-free.dev`)

### Pasos importantes
1. **Configura el URI de redirección en Azure AD** con tu URL de ngrok:
   - Para usuarios: `https://tu-dominio-ngrok.ngrok-free.dev/api/integrations/oauth/callback/outlook`
   - Para bot: `https://tu-dominio-ngrok.ngrok-free.dev/api/integrations/oauth/callback/outlook` (puede ser el mismo)
2. **Configura `OUTLOOK_REDIRECT_URI` y `OUTLOOK_BOT_REDIRECT_URI` en `.env`** con la misma URL de ngrok
3. **Asegúrate de que ngrok esté corriendo** antes de intentar autorizar la aplicación

### ⚠️ Advertencias sobre ngrok
- **La URL de ngrok cambia** cada vez que reinicias ngrok (a menos que uses un plan de pago)
- Si la URL cambia, debes actualizar:
  1. El URI de redirección en ambas aplicaciones de Azure AD
  2. `OUTLOOK_REDIRECT_URI` y `OUTLOOK_BOT_REDIRECT_URI` en tu `.env`
- Para desarrollo, considera usar un plan de ngrok con dominio fijo o configurar un dominio personalizado

---

## 📋 Permisos: Delegados vs. de Aplicación

### ¿Por qué usamos Permisos Delegados?

**Permisos Delegados** (Delegated Permissions):
- ✅ La aplicación actúa **"en nombre del usuario"** que autorizó
- ✅ Cada usuario debe autorizar la aplicación individualmente (OAuth)
- ✅ La aplicación solo puede acceder a recursos del usuario que autorizó
- ✅ Más seguro: cada usuario controla qué puede hacer la app con sus datos
- ✅ **Usamos esto** porque necesitamos que cada usuario autorice para su calendario

**Permisos de Aplicación** (Application Permissions):
- ❌ La aplicación actúa **por sí misma** (sin usuario)
- ❌ Requiere consentimiento del administrador para toda la organización
- ❌ Puede acceder a recursos de TODOS los usuarios sin autorización individual
- ❌ Menos seguro: la app tiene acceso global
- ❌ **NO usamos esto** porque necesitamos autorización por usuario

### Resumen de Permisos por Aplicación

#### Aplicación de Usuarios

| Permiso              | Tipo         | Requiere Admin | Para qué                       |
|---------             |------        |----------------|----------                      |
| `Calendars.Read`     | **Delegado** | ❌ No          | Leer calendarios de usuarios  |
| `User.Read`          | **Delegado** | ❌ No          | Leer perfil básico del usuario |
| `offline_access`     | **OAuth 2.0**| ❌ No          | Obtener refresh_token para mantener acceso sin reautenticación |

**Nota**: 
- Usamos `Calendars.Read` (solo lectura) en lugar de `Calendars.ReadWrite` porque **NO necesitamos escribir** en calendarios, solo leer eventos y participantes.
- `offline_access` es un scope estándar de OAuth 2.0 que permite obtener `refresh_token` para renovar tokens automáticamente sin que el usuario tenga que volver a autenticarse. **No se configura en Azure AD**, se solicita automáticamente cuando se incluye en los scopes.

#### Aplicación del Bot

| Permiso                           | Tipo         | Requiere Admin | Para qué                       |
|---------                          |------        |----------------|----------                      |
| `User.Read`                       | **Delegado** | ❌ No          | Leer perfil básico del bot     |
| `OnlineMeetingRecording.Read.All` | **Delegado** | ✅ **SÍ**      | Obtener grabaciones de Teams   |
| `offline_access`                  | **OAuth 2.0**| ❌ No          | Obtener refresh_token para mantener acceso sin reautenticación |

**Nota**: `offline_access` es un scope estándar de OAuth 2.0 que permite obtener `refresh_token` para renovar tokens automáticamente sin que el bot tenga que volver a autenticarse. **No se configura en Azure AD**, se solicita automáticamente cuando se incluye en los scopes.

**Todos son permisos DELEGADOS** porque:
1. Cada usuario debe autorizar para su calendario
2. El bot debe autorizar para obtener grabaciones
3. No queremos acceso global sin autorización

---

## Paso 1: Registrar Aplicación de Usuarios en Azure AD

### 1.1 Acceder a Azure Portal

1. Ve a [Azure Portal](https://portal.azure.com/)
2. Inicia sesión con una cuenta que tenga permisos de administrador en tu organización
3. Si no tienes acceso, pide a tu administrador de TI que te ayude

### 1.2 Ir a Azure Active Directory

1. En el menú lateral izquierdo, busca **"Azure Active Directory"** (o **"Microsoft Entra ID"**)
2. Haz clic para abrirlo

### 1.3 Registrar Nueva Aplicación (Usuarios)

1. En el menú lateral izquierdo, ve a **"Registros de aplicaciones"**
2. Haz clic en **"+ Nuevo registro"** (o **"+ New registration"**)

3. Completa el formulario:
   - **Nombre**: `Cosmos Notetaker Usuarios` (o el nombre que prefieras)
   - **Tipos de cuenta admitidos**: 
      - Si quieres que funcione con cualquier cuenta Microsoft: **"Cuentas en cualquier directorio organizativo y cuentas Microsoft personales"**
   - **URI de redirección**: 
     - Tipo: **"Web"**
     - **Para desarrollo con ngrok**:
       - URI: `https://untensible-unlicentiously-shaunta.ngrok-free.dev/api/integrations/oauth/callback/outlook`
     - **Para desarrollo local sin ngrok**:
       - URI: `http://localhost:7000/api/integrations/oauth/callback/outlook`
     - **Para producción**:
       - URI: `https://tu-dominio.com/api/integrations/oauth/callback/outlook`

4. Haz clic en **"Registrar"**

5. **¡IMPORTANTE!** Se mostrará la página de información general:
   - **ID de aplicación (cliente)**: Copia este valor → será `OUTLOOK_CLIENT_ID`
   - **ID de directorio (inquilino)**: Copia este valor → será `OUTLOOK_TENANT_ID`
   - Guárdalos en un lugar seguro

### 1.4 Configurar Permisos de API (Usuarios)

1. En la página de tu aplicación, ve a **"Permisos de API"** en el menú lateral
2. Haz clic en **"+ Añadir un permiso"** (o **"+ Add a permission"**)
3. Selecciona **"Microsoft Graph"**
4. **IMPORTANTE**: Selecciona **"Permisos delegados"** (Delegated permissions) - **NO** "Permisos de aplicación"

5. Busca y selecciona los siguientes permisos (uno por uno):
   
   **a) `Calendars.Read`**
   - Busca: `Calendars.Read`
   - Descripción: "Leer calendarios del usuario"
   - Requiere admin: ❌ No
   - Marca la casilla y haz clic en **"Añadir permisos"**
   
   **b) `User.Read`**
   - Busca: `User.Read`
   - Descripción: "Iniciar sesión y leer el perfil del usuario"
   - Requiere admin: ❌ No
   - Marca la casilla y haz clic en **"Añadir permisos"**

6. Verifica que todos los permisos aparecen en la lista con el tipo **"Delegado"**

**Nota**: 
- Esta aplicación NO necesita `OnlineMeetingRecording.Read.All` porque solo los usuarios normales la usarán.
- **IMPORTANTE**: El scope `offline_access` NO se configura aquí en Azure AD. Es un scope estándar de OAuth 2.0 que se solicita automáticamente cuando se incluye en los scopes del código. Los usuarios verán el permiso "Mantener el acceso a los datos a los que se ha concedido acceso" en la pantalla de consentimiento.

### 1.5 Crear Client Secret (Usuarios)

1. En la página de tu aplicación, ve a **"Certificados y secretos"** (o **"Certificates & secrets"**)
2. En la sección **"Secretos de cliente"**, haz clic en **"+ Nuevo secreto de cliente"** (o **"+ New client secret"**)
3. Completa el formulario:
   - **Descripción**: `Notetaker Usuarios Secret` (o cualquier descripción)
   - **Expira**: Selecciona cuándo expira (recomendado: 24 meses o "Nunca" para desarrollo)
4. Haz clic en **"Agregar"** (o **"Add"**)
5. **¡CRÍTICO!** Se mostrará el **Valor** del secreto:
   - **Copia este valor INMEDIATAMENTE** (solo se muestra una vez)
   - Guárdalo en un lugar seguro → será `OUTLOOK_CLIENT_SECRET`
   - Si lo pierdes, tendrás que crear un nuevo secreto

---

## Paso 2: Registrar Aplicación del Bot en Azure AD

### 2.1 Registrar Nueva Aplicación (Bot)

1. En **"Registros de aplicaciones"**, haz clic en **"+ Nuevo registro"** nuevamente

2. Completa el formulario:
   - **Nombre**: `Cosmos Notetaker Bot` (o el nombre que prefieras)
   - **Tipos de cuenta admitidos**: 
      - Si quieres que funcione con cualquier cuenta Microsoft: **"Cuentas en cualquier directorio organizativo y cuentas Microsoft personales"**
   - **URI de redirección**: 
     - Tipo: **"Web"**
     - **Para desarrollo con ngrok**:
       - URI: `https://tu-dominio-ngrok.ngrok-free.dev/api/integrations/oauth/callback/outlook`
     - **Para desarrollo local sin ngrok**:
       - URI: `http://localhost:7000/api/integrations/oauth/callback/outlook`
     - **Para producción**:
       - URI: `https://tu-dominio.com/api/integrations/oauth/callback/outlook`
     - **Nota**: Puede ser el mismo URI que la aplicación de usuarios

3. Haz clic en **"Registrar"**

4. **¡IMPORTANTE!** Se mostrará la página de información general:
   - **ID de aplicación (cliente)**: Copia este valor → será `OUTLOOK_BOT_CLIENT_ID`
   - **ID de directorio (inquilino)**: Copia este valor (debe ser el mismo que el anterior) → será `OUTLOOK_TENANT_ID`
   - Guárdalos en un lugar seguro

### 2.2 Configurar Permisos de API (Bot)

1. En la página de tu aplicación, ve a **"Permisos de API"** en el menú lateral
2. Haz clic en **"+ Añadir un permiso"** (o **"+ Add a permission"**)
3. Selecciona **"Microsoft Graph"**
4. **IMPORTANTE**: Selecciona **"Permisos delegados"** (Delegated permissions) - **NO** "Permisos de aplicación"

5. Busca y selecciona los siguientes permisos (uno por uno):
   
   **a) `User.Read`**
   - Busca: `User.Read`
   - Descripción: "Iniciar sesión y leer el perfil del usuario"
   - Requiere admin: ❌ No
   - Marca la casilla y haz clic en **"Añadir permisos"**
   
   **b) `OnlineMeetingRecording.Read.All`** ⚠️ **CRÍTICO**
   - Busca: `OnlineMeetingRecording.Read.All`
   - Descripción: "Leer grabaciones de reuniones en línea en nombre del usuario"
   - Requiere admin: ✅ **SÍ** (requiere consentimiento del administrador)
   - Marca la casilla y haz clic en **"Añadir permisos"**

6. Verifica que todos los permisos aparecen en la lista con el tipo **"Delegado"**

**Nota**: 
- **IMPORTANTE**: El scope `offline_access` NO se configura aquí en Azure AD. Es un scope estándar de OAuth 2.0 que se solicita automáticamente cuando se incluye en los scopes del código. El bot verá el permiso "Mantener el acceso a los datos a los que se ha concedido acceso" en la pantalla de consentimiento.

### 2.3 Otorgar Consentimiento del Administrador (Bot)

**⚠️ CRÍTICO**: `OnlineMeetingRecording.Read.All` **REQUIERE consentimiento del administrador**. Sin esto, el bot no podrá autorizar la aplicación.

**📌 Recordatorio**: El consentimiento del administrador **NO otorga permisos de administrador** a la aplicación. Solo aprueba que la aplicación puede solicitar este permiso sensible al bot.

1. En la página de "Permisos de API", verás una tabla con todos los permisos añadidos
2. Verifica que `OnlineMeetingRecording.Read.All` muestra **"Requiere consentimiento del administrador"** en la columna de estado
3. Haz clic en el botón **"Conceder consentimiento de administrador para [tu organización]"** (o **"Grant admin consent for [your organization]"**)
4. Confirma que quieres otorgar consentimiento
5. Deberías ver que el estado cambia a **"Concedido para [tu organización]"** con un ✅ verde

**Nota**: 
- Si no tienes permisos de administrador, pide a tu administrador de TI que haga este paso
- Sin el consentimiento del administrador, el bot verá un error al intentar autorizar la aplicación
- El consentimiento del administrador aplica a TODA la organización (permite que el bot pueda autorizar, pero el bot debe hacerlo individualmente)
- **Importante**: Este paso NO da permisos de administrador a la aplicación, solo aprueba que puede solicitar este permiso sensible

### 2.4 Crear Client Secret (Bot)

1. En la página de tu aplicación, ve a **"Certificados y secretos"** (o **"Certificates & secrets"**)
2. En la sección **"Secretos de cliente"**, haz clic en **"+ Nuevo secreto de cliente"** (o **"+ New client secret"**)
3. Completa el formulario:
   - **Descripción**: `Notetaker Bot Secret` (o cualquier descripción)
   - **Expira**: Selecciona cuándo expira (recomendado: 24 meses o "Nunca" para desarrollo)
4. Haz clic en **"Agregar"** (o **"Add"**)
5. **¡CRÍTICO!** Se mostrará el **Valor** del secreto:
   - **Copia este valor INMEDIATAMENTE** (solo se muestra una vez)
   - Guárdalo en un lugar seguro → será `OUTLOOK_BOT_CLIENT_SECRET`
   - Si lo pierdes, tendrás que crear un nuevo secreto

---

## Paso 3: Configurar Variables de Entorno

Abre tu archivo `.env` en la carpeta `backend/` y añade:

### Opción A: Usando ngrok (Recomendado para desarrollo)

Si usas ngrok para exponer tu aplicación local a internet:

```env
# Outlook Calendar OAuth - Para usuarios normales (solo lectura de calendarios)
OUTLOOK_CLIENT_ID=tu-client-id-usuarios-aqui
OUTLOOK_CLIENT_SECRET=tu-client-secret-usuarios-aqui
OUTLOOK_TENANT_ID=tu-tenant-id-aqui
OUTLOOK_REDIRECT_URI=https://tu-dominio-ngrok.ngrok-free.dev/api/integrations/oauth/callback/outlook

# Outlook Calendar OAuth - Para el bot (acceso a grabaciones)
OUTLOOK_BOT_CLIENT_ID=tu-client-id-bot-aqui
OUTLOOK_BOT_CLIENT_SECRET=tu-client-secret-bot-aqui
OUTLOOK_BOT_REDIRECT_URI=https://tu-dominio-ngrok.ngrok-free.dev/api/integrations/oauth/callback/outlook

# Cuenta del bot
TEAMS_BOT_EMAIL=tu-bot@empresa.com
TEAMS_BOT_PASSWORD=tu_password_del_bot
```

**Ejemplo con ngrok**:
```env
# Aplicación de usuarios
OUTLOOK_CLIENT_ID=12345678-1234-1234-1234-123456789abc
OUTLOOK_CLIENT_SECRET=abc~DEF123ghi456JKL789mno012PQR345stu678
OUTLOOK_TENANT_ID=87654321-4321-4321-4321-cba987654321
OUTLOOK_REDIRECT_URI=https://untensible-unlicentiously-shaunta.ngrok-free.dev/api/integrations/oauth/callback/outlook

# Aplicación del bot
OUTLOOK_BOT_CLIENT_ID=98765432-4321-4321-4321-987654321abc
OUTLOOK_BOT_CLIENT_SECRET=xyz~ABC789def012GHI345jkl678MNO901pqr234
OUTLOOK_BOT_REDIRECT_URI=https://untensible-unlicentiously-shaunta.ngrok-free.dev/api/integrations/oauth/callback/outlook

# Cuenta del bot
TEAMS_BOT_EMAIL=bot@miempresa.com
TEAMS_BOT_PASSWORD=MiPasswordSegura123
```

### Opción B: Desarrollo local sin ngrok

Si desarrollas solo localmente (sin acceso desde internet):

```env
# Outlook Calendar OAuth - Para usuarios normales
OUTLOOK_CLIENT_ID=tu-client-id-usuarios-aqui
OUTLOOK_CLIENT_SECRET=tu-client-secret-usuarios-aqui
OUTLOOK_TENANT_ID=tu-tenant-id-aqui
OUTLOOK_REDIRECT_URI=http://localhost:7000/api/integrations/oauth/callback/outlook

# Outlook Calendar OAuth - Para el bot
OUTLOOK_BOT_CLIENT_ID=tu-client-id-bot-aqui
OUTLOOK_BOT_CLIENT_SECRET=tu-client-secret-bot-aqui
OUTLOOK_BOT_REDIRECT_URI=http://localhost:7000/api/integrations/oauth/callback/outlook

# Cuenta del bot
TEAMS_BOT_EMAIL=tu-bot@empresa.com
TEAMS_BOT_PASSWORD=tu_password_del_bot
```

**⚠️ IMPORTANTE sobre ngrok**:
- **La URL de ngrok debe coincidir EXACTAMENTE** con la configurada en Azure AD
- Si reinicias ngrok y obtienes una nueva URL, debes:
  1. Actualizar el URI de redirección en ambas aplicaciones de Azure AD
  2. Actualizar `OUTLOOK_REDIRECT_URI` y `OUTLOOK_BOT_REDIRECT_URI` en tu `.env`
- Asegúrate de que ngrok esté corriendo antes de intentar autorizar la aplicación
- Para obtener una URL fija, considera usar un plan de pago de ngrok o configurar un dominio personalizado

**Donde encontrar cada valor**:
- `OUTLOOK_CLIENT_ID`: ID de aplicación (cliente) de la **Aplicación de Usuarios** (Paso 1.3)
- `OUTLOOK_CLIENT_SECRET`: Valor del secreto de la **Aplicación de Usuarios** (Paso 1.5)
- `OUTLOOK_BOT_CLIENT_ID`: ID de aplicación (cliente) de la **Aplicación del Bot** (Paso 2.1)
- `OUTLOOK_BOT_CLIENT_SECRET`: Valor del secreto de la **Aplicación del Bot** (Paso 2.4)
- `OUTLOOK_TENANT_ID`: ID de directorio (inquilino) - es el mismo para ambas aplicaciones
- `OUTLOOK_REDIRECT_URI`: URI de redirección para usuarios (debe coincidir con Azure AD)
- `OUTLOOK_BOT_REDIRECT_URI`: URI de redirección para el bot (puede ser el mismo que el de usuarios)

---

## Paso 4: Crear Usuario del Bot en la Base de Datos

El bot necesita tener un usuario en la base de datos para guardar los tokens OAuth.

### Verificar si existe

```sql
SELECT id, email FROM users WHERE email = 'tu-bot@empresa.com';
```

### Si NO existe, crearlo

**Opción A: Desde la aplicación**
1. Inicia sesión en la aplicación con una cuenta admin
2. Crea un nuevo usuario con el email del bot

**Opción B: Directamente en SQL**
```sql
INSERT INTO users (
    id,
    microsoft_user_id,
    tenant_id,
    email,
    display_name,
    is_active,
    settings,
    created_at,
    updated_at
) VALUES (
    gen_random_uuid()::text,
    'bot_' || gen_random_uuid()::text,
    'bot_tenant',
    'tu-bot@empresa.com',  -- El mismo que TEAMS_BOT_EMAIL
    'Bot Account',
    true,
    '{}'::jsonb,
    NOW(),
    NOW()
);
```

---

## Paso 5: Conectar Outlook Calendar del Bot

Ahora necesitas autorizar la aplicación del bot para que acceda a las grabaciones de Teams.

### Opción A: Desde la UI de la Aplicación (Recomendado)

1. **Inicia sesión con la cuenta del bot**:
   - Si el bot tiene acceso a la aplicación, inicia sesión con `TEAMS_BOT_EMAIL`
   - Si no tiene acceso, crea una contraseña para el bot primero

2. **Ve a Configuración > Integraciones**:
   - Busca la sección de Outlook Calendar
   - Haz clic en **"Conectar"**

3. **Autoriza la aplicación**:
   - Serás redirigido a Microsoft para autorizar
   - Inicia sesión con la cuenta del bot (`TEAMS_BOT_EMAIL`)
   - **IMPORTANTE**: Solo verás los permisos de la aplicación del bot:
     - `User.Read`
     - `OnlineMeetingRecording.Read.All`
     - "Mantener el acceso a los datos a los que se ha concedido acceso" (equivalente a `offline_access`)
   - **NO verás** `Calendars.Read` (porque el bot no lo necesita)
   - Revisa los permisos solicitados
   - Haz clic en **"Aceptar"** o **"Consent"**
   - Serás redirigido de vuelta a la aplicación

4. **Verifica**:
   - Deberías ver "Conectado" en la tarjeta de Outlook Calendar
   - Los tokens se guardan en `user.settings["outlook_calendar"]`

### Opción B: Usando el Endpoint Directamente

1. **Asegúrate de que ngrok esté corriendo** (si usas ngrok):
   ```bash
   # Verifica que ngrok está activo y anotando la URL
   # Ejemplo: https://untensible-unlicentiously-shaunta.ngrok-free.dev
   ```

2. **Abre esta URL en el navegador** (reemplaza con el email del bot):
   
   **Con ngrok**:
   ```
   https://tu-dominio-ngrok.ngrok-free.dev/api/integrations/oauth/start/outlook?user_email=tu-bot@empresa.com
   ```
   
   **Sin ngrok (solo local)**:
   ```
   http://localhost:7000/api/integrations/oauth/start/outlook?user_email=tu-bot@empresa.com
   ```

3. **Autoriza en Microsoft**:
   - Inicia sesión con la cuenta del bot
   - **IMPORTANTE**: Solo verás los permisos de la aplicación del bot (no `Calendars.Read`)
   - Revisa y acepta los permisos
   - Serás redirigido de vuelta (a través de ngrok si lo usas)

4. **Verifica en la base de datos**:
   ```sql
   SELECT 
       email,
       settings->'outlook_calendar'->>'access_token' as has_token,
       settings->'outlook_calendar'->>'connected_at' as connected_at
   FROM users 
   WHERE email = 'tu-bot@empresa.com';
   ```

---

## Paso 6: Conectar Outlook Calendar de Usuarios Normales

Los usuarios normales pueden conectar sus calendarios usando la aplicación de usuarios.

1. **Inicia sesión con una cuenta de usuario normal** (no el bot)

2. **Ve a Configuración > Integraciones**:
   - Busca la sección de Outlook Calendar
   - Haz clic en **"Conectar"**

3. **Autoriza la aplicación**:
   - Serás redirigido a Microsoft para autorizar
   - Inicia sesión con tu cuenta de usuario
   - **IMPORTANTE**: Solo verás los permisos de la aplicación de usuarios:
     - `Calendars.Read` (solo lectura, no escritura)
     - `User.Read`
     - "Mantener el acceso a los datos a los que se ha concedido acceso" (equivalente a `offline_access`)
   - **NO verás** `OnlineMeetingRecording.Read.All` (porque los usuarios no lo necesitan)
   - Revisa los permisos solicitados
   - Haz clic en **"Aceptar"** o **"Consent"**
   - Serás redirigido de vuelta a la aplicación

4. **Verifica**:
   - Deberías ver "Conectado" en la tarjeta de Outlook Calendar
   - Los tokens se guardan en `user.settings["outlook_calendar"]`

---

## Paso 7: Verificar que Funciona

### Verificar desde la API

```bash
# Obtener estado de integraciones (usuario normal)
GET http://localhost:7000/api/integrations/status?user_email=usuario@empresa.com

# Obtener estado de integraciones (bot)
GET http://localhost:7000/api/integrations/status?user_email=tu-bot@empresa.com
```

Deberías ver:
```json
{
  "outlook_calendar": {
    "connected": true,
    "connected_at": "2024-01-21T10:30:00"
  }
}
```

### Probar que Puede Acceder a Grabaciones

Una vez conectado el bot, cuando el bot se una a una reunión:

1. El bot se une usando Playwright (funciona sin Outlook)
2. Después de la reunión, intenta obtener la grabación vía Graph API
3. Si la reunión tiene grabación habilitada y el bot tiene permisos, descargará el audio

---

## Paso 8: Configurar para Producción

Cuando despliegues a producción:

1. **Añade la URL de producción en ambas aplicaciones de Azure AD**:
   - Ve a cada aplicación en Azure Portal
   - **"Autenticación"** > **"URI de redirección"**
   - Haz clic en **"+ Añadir URI"** o **"+ Add URI"**
   - Añade: `https://tu-dominio.com/api/integrations/oauth/callback/outlook`
   - Haz clic en **"Guardar"**

2. **Actualiza el `.env` en producción**:
   ```env
   OUTLOOK_REDIRECT_URI=https://tu-dominio.com/api/integrations/oauth/callback/outlook
   OUTLOOK_BOT_REDIRECT_URI=https://tu-dominio.com/api/integrations/oauth/callback/outlook
   ```

**Nota**: Puedes tener múltiples URIs de redirección configuradas en cada aplicación de Azure AD:
- URI de ngrok (para desarrollo): `https://tu-dominio-ngrok.ngrok-free.dev/api/integrations/oauth/callback/outlook`
- URI de localhost (para desarrollo local): `http://localhost:7000/api/integrations/oauth/callback/outlook`
- URI de producción: `https://tu-dominio.com/api/integrations/oauth/callback/outlook`

Solo necesitas actualizar `OUTLOOK_REDIRECT_URI` y `OUTLOOK_BOT_REDIRECT_URI` en tu `.env` según el entorno que estés usando.

---

## Solución de Problemas

### Error: "AADSTS50011: The redirect URI specified in the request does not match"

**Problema**: La URL de redirección no coincide.

**Solución**:
1. Verifica que `OUTLOOK_REDIRECT_URI` o `OUTLOOK_BOT_REDIRECT_URI` en `.env` coincida EXACTAMENTE con una de las URIs en la aplicación correspondiente de Azure AD
2. Verifica que uses:
   - `http://` para localhost (sin ngrok)
   - `https://` para ngrok o producción
3. Verifica que no haya espacios o caracteres extra
4. **Si usas ngrok**: 
   - Verifica que la URL de ngrok en `.env` coincida con la URL actual de ngrok
   - Si reiniciaste ngrok y obtuviste una nueva URL, actualiza tanto Azure AD como el `.env`
   - Asegúrate de que ngrok esté corriendo antes de intentar autorizar

### Error: "AADSTS700016: Application was not found in the directory/tenant"

**Problema**: El `OUTLOOK_CLIENT_ID`, `OUTLOOK_BOT_CLIENT_ID` o `OUTLOOK_TENANT_ID` son incorrectos.

**Solución**:
1. Verifica que copiaste correctamente el ID de aplicación y Tenant ID desde Azure Portal
2. Asegúrate de que estás usando el Tenant ID correcto (no el ID de aplicación)
3. Verifica que estás usando la aplicación correcta (usuarios vs bot) según el caso

### Error: "AADSTS7000215: Invalid client secret is provided"

**Problema**: El `OUTLOOK_CLIENT_SECRET` o `OUTLOOK_BOT_CLIENT_SECRET` es incorrecto o expiró.

**Solución**:
1. Verifica que copiaste correctamente el secreto (sin espacios)
2. Si expiró, crea un nuevo secreto en Azure Portal y actualiza el `.env`
3. Verifica que estás usando el secreto de la aplicación correcta (usuarios vs bot)

### Error: "Insufficient privileges to complete the operation" al obtener grabaciones

**Problema**: No tienes el permiso `OnlineMeetingRecording.Read.All` o no se otorgó consentimiento del administrador.

**Solución**:
1. Verifica que añadiste el permiso `OnlineMeetingRecording.Read.All` como **permiso DELEGADO** (no de aplicación) en la **Aplicación del Bot**
2. **CRÍTICO**: Asegúrate de que se otorgó **consentimiento del administrador** (botón "Conceder consentimiento de administrador") en la **Aplicación del Bot**
3. Verifica que el estado del permiso muestra "Concedido para [tu organización]" con ✅ verde
4. Verifica que la cuenta del bot tiene permisos para acceder a grabaciones en Teams
5. Verifica que el bot autorizó la **Aplicación del Bot** correctamente (tokens guardados en BD)
6. Verifica que estás usando `OUTLOOK_BOT_CLIENT_ID` y `OUTLOOK_BOT_CLIENT_SECRET` (no los de usuarios)

### Los tokens no se guardan después de autorizar

**Problema**: Hay un error al guardar los tokens en la base de datos.

**Solución**:
1. Revisa los logs del backend para ver errores específicos
2. Verifica que el usuario (o bot) existe en la base de datos
3. Verifica que la base de datos tiene permisos de escritura

### El bot no puede obtener grabaciones

**Problema**: Aunque los tokens están guardados, no puede descargar grabaciones.

**Solución**:
1. Verifica que la reunión tiene grabación habilitada
2. Verifica que la reunión está calendarizada (no es una reunión ad-hoc)
3. Verifica que el `onlineMeetingId` se puede extraer de la URL
4. Revisa los logs para ver errores específicos de Graph API
5. Verifica que el bot está usando la **Aplicación del Bot** (con `OUTLOOK_BOT_CLIENT_ID`)

### Los usuarios ven el permiso `OnlineMeetingRecording.Read.All`

**Problema**: Estás usando la aplicación del bot para usuarios normales.

**Solución**:
1. Verifica que los usuarios normales están usando `OUTLOOK_CLIENT_ID` (aplicación de usuarios)
2. Verifica que solo el bot está usando `OUTLOOK_BOT_CLIENT_ID` (aplicación del bot)
3. El sistema detecta automáticamente si es el bot comparando el email con `TEAMS_BOT_EMAIL`

---

## Notas Importantes

### 1. Resumen de Permisos

#### Aplicación de Usuarios

| Permiso | Tipo | Admin | Uso |
|---------|------|-------|-----|
| `Calendars.Read` | Delegado | ❌ | Leer calendarios de usuarios (solo lectura) |
| `User.Read` | Delegado | ❌ | Leer perfil básico del usuario |
| `offline_access` | OAuth 2.0 | ❌ | Obtener refresh_token para mantener acceso sin reautenticación |

#### Aplicación del Bot

| Permiso | Tipo | Admin | Uso |
|---------|------|-------|-----|
| `User.Read` | Delegado | ❌ | Leer perfil básico del bot |
| `OnlineMeetingRecording.Read.All` | Delegado | ✅ **SÍ** | Obtener grabaciones de Teams |
| `offline_access` | OAuth 2.0 | ❌ | Obtener refresh_token para mantener acceso sin reautenticación |

**Todos son permisos DELEGADOS** (no de aplicación). Esto significa:
- Cada usuario debe autorizar la aplicación individualmente
- La aplicación actúa "en nombre del usuario" que autorizó
- No hay autenticación de aplicación sin usuario

**Nota sobre `offline_access`**:
- `offline_access` es un scope estándar de OAuth 2.0 que **NO se configura en Azure AD**
- Se solicita automáticamente cuando se incluye en los scopes del código
- Los usuarios verán el permiso "Mantener el acceso a los datos a los que se ha concedido acceso" en la pantalla de consentimiento
- Es necesario para obtener `refresh_token` y renovar tokens automáticamente sin reautenticación
- Sin este scope, los tokens expirarían después de 1 hora y los usuarios tendrían que reconectarse constantemente

### 2. Consentimiento del Administrador

- **`OnlineMeetingRecording.Read.All`** **REQUIERE** consentimiento del administrador (solo en la Aplicación del Bot)
- **⚠️ ACLARACIÓN**: El consentimiento del administrador **NO otorga permisos de administrador** a la aplicación. Solo aprueba que la aplicación puede solicitar este permiso sensible al bot.
- Sin el consentimiento del administrador, el bot verá un error al intentar autorizar la aplicación
- El consentimiento del administrador aplica a TODA la organización (permite que el bot pueda autorizar, pero el bot debe hacerlo individualmente)
- Los otros permisos (`Calendars.Read` y `User.Read`) NO requieren consentimiento del administrador, pero el usuario debe autorizarlos individualmente
- **Seguridad**: Es una medida de protección de Microsoft para datos sensibles. El administrador aprueba que la app puede pedir el permiso, pero el bot controla si lo otorga o no

### 3. Dos Aplicaciones Separadas

- **SÍ necesitas crear dos aplicaciones separadas** en Azure AD
- **Aplicación de Usuarios** (`OUTLOOK_CLIENT_ID` / `OUTLOOK_CLIENT_SECRET`):
  - Usada por usuarios normales para sincronizar calendarios
  - Permisos: `Calendars.Read` + `User.Read` + `offline_access` (automático)
  - Los usuarios ven estos permisos en la pantalla de consentimiento:
    - "Leer tus calendarios" (`Calendars.Read`)
    - "Leer tu perfil" (`User.Read`)
    - "Mantener el acceso a los datos a los que se ha concedido acceso" (`offline_access`)
- **Aplicación del Bot** (`OUTLOOK_BOT_CLIENT_ID` / `OUTLOOK_BOT_CLIENT_SECRET`):
  - Usada solo por el bot para acceder a grabaciones
  - Permisos: `OnlineMeetingRecording.Read.All` + `User.Read` + `offline_access` (automático)
  - El bot ve estos permisos en la pantalla de consentimiento:
    - "Leer tu perfil" (`User.Read`)
    - "Leer grabaciones de reuniones en línea" (`OnlineMeetingRecording.Read.All`)
    - "Mantener el acceso a los datos a los que se ha concedido acceso" (`offline_access`)
- El sistema detecta automáticamente si es el bot comparando el email con `TEAMS_BOT_EMAIL`
- Los tokens OAuth se guardan por usuario en `user.settings["outlook_calendar"]`

### 4. Cuenta del Bot

- La cuenta del bot (`TEAMS_BOT_EMAIL`) debe ser una cuenta válida de Microsoft/Outlook
- Debe tener acceso a Teams y poder unirse a reuniones
- Debe tener permisos para acceder a grabaciones (si tu organización las requiere)
- El bot debe autorizar la **Aplicación del Bot** igual que cualquier otro usuario autoriza la suya

### 5. Grabaciones de Teams

- Solo funcionan para reuniones calendarizadas (no ad-hoc)
- La grabación debe habilitarse durante la reunión
- Puede tardar varios minutos después de la reunión en estar disponible
- Requiere que el bot tenga permisos en Teams para acceder a grabaciones

### 6. Renovación de Tokens

- Los tokens de acceso expiran después de 1 hora
- El sistema renueva automáticamente usando el `refresh_token`
- El scope `offline_access` es necesario para obtener `refresh_token` y mantener el acceso sin reautenticación
- Si el `refresh_token` expira (normalmente después de 90 días), necesitarás reconectar
- Cada usuario (incluido el bot) tiene sus propios tokens
- Los usuarios verán el permiso "Mantener el acceso a los datos a los que se ha concedido acceso" en la pantalla de consentimiento (equivalente a `offline_access`)

### 7. Calendars.Read vs Calendars.ReadWrite

- **Usamos `Calendars.Read`** (solo lectura) en lugar de `Calendars.ReadWrite` porque:
  - **NO necesitamos escribir** en calendarios
  - Solo necesitamos leer eventos y obtener información de participantes
  - Es más seguro: menos permisos = menos riesgo
  - Los usuarios se sienten más cómodos otorgando solo permisos de lectura

---

## Resumen Rápido

1. ✅ Registrar **DOS aplicaciones** en Azure AD:
   - **Aplicación de Usuarios**: `Calendars.Read` + `User.Read` (+ `offline_access` automático)
   - **Aplicación del Bot**: `OnlineMeetingRecording.Read.All` + `User.Read` (+ `offline_access` automático)
2. ✅ Configurar URI de redirección en ambas aplicaciones:
   - Con ngrok: `https://tu-dominio-ngrok.ngrok-free.dev/api/integrations/oauth/callback/outlook`
   - Sin ngrok: `http://localhost:7000/api/integrations/oauth/callback/outlook`
3. ✅ **Otorgar consentimiento del administrador** para `OnlineMeetingRecording.Read.All` (solo en Aplicación del Bot)
4. ✅ Crear Client Secret en ambas aplicaciones
5. ✅ Configurar `.env` con credenciales:
   - `OUTLOOK_CLIENT_ID`, `OUTLOOK_CLIENT_SECRET` (usuarios)
   - `OUTLOOK_BOT_CLIENT_ID`, `OUTLOOK_BOT_CLIENT_SECRET` (bot)
   - `OUTLOOK_TENANT_ID` (mismo para ambas)
   - `OUTLOOK_REDIRECT_URI`, `OUTLOOK_BOT_REDIRECT_URI`
6. ✅ Crear usuario del bot en BD
7. ✅ Asegurarse de que ngrok esté corriendo (si usas ngrok)
8. ✅ Conectar Outlook Calendar del bot:
   - Con ngrok: `https://tu-dominio-ngrok.ngrok-free.dev/api/integrations/oauth/start/outlook?user_email=tu-bot@empresa.com`
   - Sin ngrok: `http://localhost:7000/api/integrations/oauth/start/outlook?user_email=tu-bot@empresa.com`
9. ✅ Autorizar en Microsoft (con la cuenta del bot) - solo verá permisos del bot
10. ✅ Verificar que los tokens se guardaron en `user.settings["outlook_calendar"]`
11. ✅ Probar obtención de grabaciones

**Recordatorio**: Son **DOS aplicaciones separadas**:
- Una para usuarios (sincronizar calendarios)
- Otra para el bot (obtener grabaciones)

---

## Próximos Pasos

Una vez configurado:

1. **Probar el bot completo**:
   - Crea una reunión de prueba en Teams
   - El bot se unirá automáticamente
   - Después de la reunión, intentará obtener la grabación vía Graph API

2. **Monitorear logs**:
   - Revisa los logs del backend para ver si hay errores
   - Verifica que se obtienen las grabaciones correctamente

3. **Configurar para producción**:
   - Añade URLs de producción en ambas aplicaciones de Azure AD
   - Actualiza variables de entorno

¡Listo! El bot ahora puede acceder a grabaciones de Teams vía Microsoft Graph API, y los usuarios pueden sincronizar sus calendarios sin ver permisos innecesarios.
