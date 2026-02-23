# Guía: Registrar Aplicación OAuth en Google Cloud Console

Esta guía te ayudará a registrar tu aplicación en Google Cloud Console para obtener las credenciales OAuth necesarias para la integración con Google Calendar.

## Requisitos Previos

- Una cuenta de Google (Gmail, Google Workspace, etc.)
- Acceso a Google Cloud Console

---

## Paso 1: Acceder a Google Cloud Console

1. Ve a [Google Cloud Console](https://console.cloud.google.com/)
2. Inicia sesión con tu cuenta de Google
3. Si es tu primera vez, acepta los términos y condiciones

---

## Paso 2: Crear un Nuevo Proyecto (o usar uno existente)

1. En la parte superior de la página, haz clic en el selector de proyectos (junto al logo de Google Cloud)
2. Haz clic en **"NUEVO PROYECTO"** (o selecciona uno existente si ya tienes uno)
3. Si creas uno nuevo:
   - **Nombre del proyecto**: `Cosmos NoteTaker` (o el nombre que prefieras)
   - **Organización**: Déjalo en blanco si no tienes una
   - Haz clic en **"CREAR"**
4. Espera a que se cree el proyecto (puede tardar unos segundos)
5. Selecciona el proyecto recién creado desde el selector de proyectos

---

## Paso 3: Habilitar Google Calendar API

1. En el menú lateral izquierdo, ve a **"APIs y servicios"** > **"Biblioteca"**
2. En el buscador, escribe: **"Google Calendar API"**
3. Haz clic en **"Google Calendar API"** en los resultados
4. Haz clic en el botón **"HABILITAR"**
5. Espera a que se habilite (puede tardar unos segundos)

**⚠️ IMPORTANTE**: Si ya creaste las credenciales OAuth y estás viendo el error:
```
Google Calendar API has not been used in project XXXX before or it is disabled
```

**Solución rápida**:
1. Ve directamente a: https://console.cloud.google.com/apis/library/calendar-json.googleapis.com
2. Selecciona tu proyecto
3. Haz clic en **"HABILITAR"**
4. Espera 1-2 minutos para que se propague
5. Intenta sincronizar de nuevo desde la aplicación

---

## Paso 4: Configurar la Pantalla de Consentimiento OAuth

1. En el menú lateral, ve a **"APIs y servicios"** > **"Pantalla de consentimiento de OAuth"**
2. Selecciona el tipo de usuario:
   - **Externo**: Si quieres que cualquier usuario con cuenta de Google pueda usar tu app
   - **Interno**: Solo para usuarios de tu organización (requiere Google Workspace)
   
   **Recomendación**: Selecciona **"Externo"** para desarrollo y producción general

3. Haz clic en **"CREAR"**

4. Completa el formulario:

   **Información de la aplicación:**
   - **Nombre de la aplicación**: `Cosmos NoteTaker` (o el nombre que prefieras)
   - **Email de soporte del usuario**: Tu email
   - **Logo de la aplicación**: (Opcional) Sube un logo si tienes uno
   - **Dominio de inicio de sesión de la aplicación**: (Opcional) Tu dominio si tienes uno
   - **Email del desarrollador**: Tu email

   **Dominios autorizados:**
   - (Opcional) Si tienes un dominio, añádelo aquí

   **Alcance:**
   - Por ahora, déjalo como está. Los scopes se configuran en las credenciales.

   **Usuarios de prueba:**
   - Si tu app está en modo "Prueba", añade aquí los emails de los usuarios que quieres que puedan probarla
   - Puedes añadir hasta 100 usuarios de prueba
   - **Importante**: En modo prueba, solo estos usuarios podrán autorizar la app
   - **⚠️ CRÍTICO**: Añade tu propio email aquí para poder probar la aplicación

5. Haz clic en **"GUARDAR Y CONTINUAR"**

6. En la siguiente pantalla (Resumen), revisa la información y haz clic en **"VOLVER AL PANEL"**

---

### ⚠️ IMPORTANTE: Añadir Usuarios de Prueba (si ya creaste la app)

Si ya creaste la aplicación y estás viendo el error "Error 403: access_denied", necesitas añadir tu email a la lista de usuarios de prueba:

1. Ve a **"APIs y servicios"** > **"Pantalla de consentimiento de OAuth"**
2. Busca la sección **"Usuarios de prueba"**
3. Haz clic en **"+ AÑADIR USUARIOS"**
4. Añade el email que quieres que pueda autorizar la app (por ejemplo: `tu-email@gmail.com`)
5. Haz clic en **"GUARDAR"**
6. **Importante**: Los cambios pueden tardar unos minutos en aplicarse. Espera 1-2 minutos y vuelve a intentar

**Nota**: Si estás desarrollando y quieres que cualquier usuario pueda usar tu app (sin restricciones), tendrás que enviar la app para verificación de Google, lo cual es un proceso más largo que requiere proporcionar información detallada sobre tu aplicación.

---

## Paso 5: Crear Credenciales OAuth 2.0

1. En el menú lateral, ve a **"APIs y servicios"** > **"Credenciales"**

2. En la parte superior, haz clic en **"+ CREAR CREDENCIALES"**

3. Selecciona **"ID de cliente de OAuth 2.0"**

4. Si te pregunta qué tipo de aplicación, selecciona **"Aplicación web"**

5. Completa el formulario:

   **Nombre:**
   - `Cosmos NoteTaker - Web Client` (o el nombre que prefieras)

   **URI de redirección autorizados:**
   - Haz clic en **"+ AÑADIR URI"**
   - Añade las siguientes URIs (una por una):
     
     **Para desarrollo local:**
     ```
     http://localhost:7000/api/integrations/oauth/callback/google
     ```
     
     **Para producción (cuando despliegues):**
     ```
     https://tu-dominio.com/api/integrations/oauth/callback/google
     ```
     
     **Si usas ngrok o túnel similar:**
     ```
     https://tu-url-ngrok.ngrok.io/api/integrations/oauth/callback/google
     ```
   
   **Importante**: 
   - Añade todas las URLs que vayas a usar
   - Las URLs deben coincidir EXACTAMENTE (incluyendo http/https, puerto, y ruta completa)
   - No uses `localhost` en producción

6. Haz clic en **"CREAR"**

7. **¡IMPORTANTE!** Se mostrará una ventana con tus credenciales:
   - **ID de cliente**: Copia este valor (lo necesitarás)
   - **Secreto de cliente**: Copia este valor (lo necesitarás)
   
   **⚠️ ADVERTENCIA**: Esta es la ÚNICA vez que verás el "Secreto de cliente" completo. 
   - Si lo pierdes, tendrás que crear nuevas credenciales
   - Guárdalo en un lugar seguro

8. Haz clic en **"LISTO"**

---

## Paso 6: Configurar Scopes (Permisos)

1. En la página de credenciales, haz clic en el nombre de tu ID de cliente OAuth 2.0 que acabas de crear

2. En la sección **"Scopes de OAuth 2.0"**, haz clic en **"AÑADIR SCOPE"**

3. Busca y selecciona los siguientes scopes:
   - `https://www.googleapis.com/auth/calendar.readonly` - Leer eventos del calendario
   - `https://www.googleapis.com/auth/calendar.events` - Crear y modificar eventos

4. Haz clic en **"ACTUALIZAR"**

---

## Paso 7: Configurar Variables de Entorno

Ahora que tienes las credenciales, añádelas a tu archivo `.env`:

1. Abre el archivo `.env` en la carpeta `backend/`

2. Añade las siguientes líneas (reemplaza con tus valores reales):

```env
# Google Calendar OAuth
GOOGLE_CLIENT_ID=tu-client-id-aqui.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=tu-client-secret-aqui
GOOGLE_REDIRECT_URI=http://localhost:7000/api/integrations/oauth/callback/google
```

**Ejemplo:**
```env
GOOGLE_CLIENT_ID=123456789-abcdefghijklmnop.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=GOCSPX-abcdefghijklmnopqrstuvwxyz
GOOGLE_REDIRECT_URI=http://localhost:7000/api/integrations/oauth/callback/google
```

3. Guarda el archivo

---

## Paso 8: Verificar que Funciona

1. Inicia tu servidor backend:
   ```bash
   cd backend
   python -m uvicorn app.main:app --reload
   ```

2. Inicia tu frontend:
   ```bash
   cd frontend
   npm run dev
   ```

3. Ve a la aplicación en el navegador: `http://localhost:3000` (o el puerto que uses)

4. Inicia sesión y ve a **Configuración** > **Integraciones**

5. Haz clic en **"Conectar"** en la tarjeta de Google Calendar

6. Deberías ser redirigido a Google para autorizar la aplicación

7. Después de autorizar, serás redirigido de vuelta a tu aplicación

---

## Solución de Problemas

### Error: "redirect_uri_mismatch"
- **Causa**: La URL de redirección no coincide exactamente con las configuradas en Google Cloud Console
- **Solución**: Verifica que la URL en `.env` coincida EXACTAMENTE con una de las URIs autorizadas en Google Cloud Console

### Error: "access_denied" o "Error 403: access_denied"
- **Causa**: La aplicación está en modo "Prueba" y tu email no está en la lista de usuarios de prueba
- **Solución**: 
  1. Ve a Google Cloud Console > "APIs y servicios" > "Pantalla de consentimiento de OAuth"
  2. Busca la sección "Usuarios de prueba"
  3. Haz clic en "+ AÑADIR USUARIOS"
  4. Añade el email del usuario que intenta autorizar (por ejemplo: `fdescartin@gmail.com`)
  5. Guarda y espera 1-2 minutos
  6. Intenta de nuevo
- **Prevención**: Añade todos los emails que necesites desde el principio en la configuración inicial

### Error: "invalid_client"
- **Causa**: El Client ID o Client Secret son incorrectos
- **Solución**: Verifica que copiaste correctamente las credenciales en el archivo `.env`

### La aplicación no aparece en Google Calendar API
- **Causa**: No has habilitado la API de Google Calendar
- **Solución**: Ve a "APIs y servicios" > "Biblioteca" y habilita "Google Calendar API"

### Error: "Google Calendar API has not been used in project XXXX before or it is disabled"
- **Causa**: La API de Google Calendar no está habilitada en tu proyecto de Google Cloud
- **Síntomas**: La conexión OAuth funciona, pero la sincronización automática falla con error 403
- **Solución**: 
  1. Ve a: https://console.cloud.google.com/apis/library/calendar-json.googleapis.com
  2. Selecciona tu proyecto (el mismo donde creaste las credenciales OAuth)
  3. Haz clic en **"HABILITAR"**
  4. Espera 1-2 minutos para que se propague
  5. Vuelve a intentar sincronizar desde la aplicación (botón "Sincronizar ahora")

---

## Notas Importantes

1. **Modo Prueba vs Producción**:
   - En modo "Prueba", solo los usuarios añadidos a "Usuarios de prueba" pueden autorizar la app
   - Para producción, necesitarás enviar la app para verificación de Google (proceso más largo)
   - **Durante desarrollo**: Usa modo "Prueba" y añade los emails necesarios
   - **En producción**: Debes enviar la app para verificación si quieres que cualquier usuario pueda usarla

2. **Límites de Cuota**:
   - Google Calendar API tiene límites de cuota gratuita
   - Por defecto: 1,000,000 requests por día
   - Para la mayoría de casos de uso, esto es más que suficiente

3. **Seguridad**:
   - NUNCA subas el archivo `.env` a Git
   - El `.env` ya está en `.gitignore` por defecto
   - Guarda las credenciales de forma segura

4. **URLs de Redirección**:
   - Para desarrollo: usa `http://localhost:7000/...`
   - Para producción: usa `https://tu-dominio.com/...`
   - Puedes tener múltiples URLs configuradas

---

## Próximos Pasos

Una vez que tengas Google Calendar funcionando, puedes repetir un proceso similar para Outlook Calendar usando Azure AD. La guía para Outlook estará en otro documento.

---

---

## Paso 9: Configurar para Producción (Aceptar Cualquier Email)

Para que tu aplicación pueda ser usada por cualquier usuario en producción (sin necesidad de añadirlos a "Usuarios de prueba"), debes enviar tu aplicación para verificación de Google.

### ¿Cuándo hacerlo?

- **Desarrollo/Pruebas**: Mantén la app en modo "Prueba" y añade usuarios de prueba
- **Producción con usuarios limitados**: Puedes mantener modo "Prueba" y añadir hasta 100 usuarios
- **Producción pública**: Envía la app para verificación

### Requisitos para Verificación

Antes de enviar tu app para verificación, necesitas:

1. **Completar la Pantalla de Consentimiento OAuth**:
   - Nombre de la aplicación
   - Logo (recomendado: 128x128px mínimo)
   - Email de soporte
   - Dominio autorizado (si tienes uno)
   - Política de privacidad (URL pública)
   - Términos de servicio (URL pública) - opcional pero recomendado
   - Descripción detallada de por qué necesitas los permisos

2. **Documentar los Scopes que usas**:
   - `https://www.googleapis.com/auth/calendar.readonly`: Explicar por qué necesitas leer el calendario
   - `https://www.googleapis.com/auth/calendar.events`: Explicar por qué necesitas crear/modificar eventos

3. **Tener URLs públicas**:
   - URL de tu aplicación en producción
   - Política de privacidad accesible públicamente
   - Términos de servicio (recomendado)

### Pasos para Enviar para Verificación

1. **Completa toda la información de la Pantalla de Consentimiento**:
   - Ve a "APIs y servicios" > "Pantalla de consentimiento de OAuth"
   - Asegúrate de que todos los campos obligatorios estén completos

2. **Añade URLs requeridas**:
   - **Política de privacidad**: URL pública donde expliques qué datos recopilas y cómo los usas
   - **Términos de servicio**: (Opcional pero recomendado)

3. **Documenta los Scopes**:
   - En la sección "Scopes", haz clic en cada scope y completa:
     - **Descripción**: Explica claramente por qué tu app necesita este permiso
     - **Justificación**: Proporciona una justificación detallada del uso

4. **Prepara información adicional** (puede ser solicitada):
   - Video o capturas de pantalla mostrando cómo usa tu app los permisos
   - Descripción detallada de la funcionalidad de tu app
   - Información sobre cómo almacenas y proteges los datos del usuario

5. **Cambia a Producción**:
   - Una vez completada toda la información, verás la opción "PUBLICAR APP"
   - Haz clic en "PUBLICAR APP"
   - Google revisará tu solicitud (puede tardar varios días a varias semanas)

### Tiempo de Verificación

- **Proceso típico**: 4-6 semanas
- **Puede ser más rápido**: Si tu app es simple y los permisos están bien documentados
- **Puede ser más lento**: Si Google necesita información adicional o si hay problemas

### Durante el Proceso de Verificación

- Puedes seguir usando la app en modo "Prueba" mientras esperas
- Google puede contactarte con preguntas o solicitar más información
- Puedes revisar el estado en la consola de Google Cloud

### Alternativa: Mantener Modo Prueba (hasta 100 usuarios)

Si no necesitas que sea pública inmediatamente:
- Puedes mantener la app en modo "Prueba"
- Añadir hasta 100 usuarios de prueba
- Esto te permite usar la app sin esperar la verificación

### Recomendación

Para Cosmos NoteTaker:
1. **Durante desarrollo**: Usa modo "Prueba" con usuarios de prueba
2. **Para producción inicial**: Puedes mantener modo "Prueba" y añadir usuarios según se registren
3. **Para escala masiva**: Envía para verificación cuando necesites más de 100 usuarios o acceso público

---

## Recursos Adicionales

- [Documentación oficial de Google OAuth 2.0](https://developers.google.com/identity/protocols/oauth2)
- [Google Calendar API Documentation](https://developers.google.com/calendar/api)
- [Guía de verificación de Google](https://support.google.com/cloud/answer/9110914)
- [Políticas de Google para OAuth](https://developers.google.com/identity/protocols/oauth2/policies)
- [Google Cloud Console](https://console.cloud.google.com/)

