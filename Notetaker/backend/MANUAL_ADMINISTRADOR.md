# Manual de Administrador - Cosmos Notetaker

## ¿Qué es Cosmos Notetaker?

Cosmos Notetaker es una plataforma empresarial para la grabación, transcripción y análisis automático de reuniones de Microsoft Teams. El sistema utiliza bots de Recall.ai que se unen automáticamente a las reuniones programadas, graban el contenido y generan transcripciones completas con análisis de IA.

### Características Principales

- **Grabación Automática**: Bots que se unen automáticamente a reuniones de Teams
- **Transcripción en Tiempo Real**: Transcripciones completas con identificación de hablantes
- **Grabación de Audio y Video**: Almacenamiento de archivos MP3 y MP4
- **Análisis de IA**: Resúmenes, puntos clave, acciones y análisis de sentimiento
- **Gestión Multi-Usuario**: Sistema de permisos granulares por licencia
- **Integración con Calendarios**: Sincronización con Google Calendar y Outlook

---

## Acceso de Administrador

### Inicio de Sesión

Los administradores se identifican por:
- Email que termina en el dominio configurado en `ADMIN_EMAIL_DOMAIN`
- O email incluido en la lista `ADMIN_EMAILS`

### Funcionalidades de Administrador

Como administrador tienes acceso a:
- **Gestión de Usuarios**: Crear, editar y eliminar usuarios
- **Gestión de Reuniones**: Ver todas las reuniones del sistema
- **Asignación de Licencias**: Configurar niveles de licencia (Basic, Advanced, PRO)
- **Monitoreo del Sistema**: Ver estado de bots, tareas de Celery, etc.

---

## Gestión de Usuarios

### Ver Lista de Usuarios

1. Haz clic en **"Usuarios"** en el menú de administración
2. Verás una tabla con todos los usuarios del sistema mostrando:
   - Email
   - Nombre para mostrar
   - Estado (Activo/Inactivo)
   - Licencia actual
   - Último acceso
   - Si es administrador

### Crear Nuevo Usuario

1. En la página de **"Usuarios"**, haz clic en **"Crear Usuario"**
2. Completa el formulario:
   - **Email**: Dirección de correo del usuario
   - **Contraseña**: Contraseña temporal (el usuario deberá cambiarla en el primer acceso)
   - **Nombre para mostrar** (opcional)
3. Haz clic en **"Crear"**

El usuario recibirá una contraseña temporal y deberá cambiarla en su primer acceso.

### Editar Usuario

1. En la lista de usuarios, haz clic en el usuario que deseas editar
2. Puedes modificar:
   - **Nombre para mostrar**
   - **Email** (con precaución)
   - **Licencia**: Basic, Advanced o PRO
   - **Estado Premium**: Activar/desactivar funcionalidades premium
3. Haz clic en **"Guardar"**

**IMPORTANTE**: Al cambiar la licencia de un usuario, todos sus permisos de acceso a reuniones se actualizarán automáticamente.

### Resetear Contraseña de Usuario

1. En la lista de usuarios, selecciona el usuario
2. Haz clic en **"Resetear Contraseña"**
3. Ingresa la nueva contraseña temporal
4. El usuario deberá cambiar esta contraseña en su próximo acceso

### Eliminar Usuario

**PRECAUCIÓN**: Eliminar un usuario puede afectar el acceso a reuniones.

1. Selecciona el usuario
2. Haz clic en **"Eliminar"**
3. Confirma la acción

---

## Gestión de Licencias

### Niveles de Licencia

El sistema tiene tres niveles de licencia:

#### Licencia Basic
- ✅ Acceso a transcripciones
- ❌ Sin acceso a audio
- ❌ Sin acceso a video

#### Licencia Advanced
- ✅ Acceso a transcripciones
- ✅ Acceso a audio
- ❌ Sin acceso a video

#### Licencia PRO
- ✅ Acceso a transcripciones
- ✅ Acceso a audio
- ✅ Acceso a video

### Asignar Licencia a Usuario

1. Ve a **"Usuarios"**
2. Selecciona el usuario
3. En el campo **"Licencia"**, selecciona:
   - `basic`
   - `advanced`
   - `pro`
4. Haz clic en **"Guardar"**

**Nota**: Los cambios de licencia se aplican inmediatamente y actualizan todos los permisos de acceso a reuniones del usuario.

### Configuración de Licencias

Las licencias se pueden asignar de dos formas:

1. **Manual**: A través de la interfaz de administración
2. **Automática**: Basada en el campo `is_premium` del usuario:
   - Si `is_premium = True` → Licencia Advanced
   - Si `is_premium = False` → Licencia Basic
   - Si tiene `license_level = "pro"` en settings → Licencia PRO

---

## Gestión de Reuniones

### Ver Todas las Reuniones

1. Haz clic en **"Reuniones"** en el menú de administración
2. Verás todas las reuniones del sistema, independientemente del usuario
3. Puedes filtrar por:
   - Estado (Pendiente, En curso, Completada, Fallida)
   - Fecha
   - Usuario organizador

### Ver Detalles de Reunión

1. Haz clic en una reunión de la lista
2. Verás información completa:
   - Detalles de la reunión (título, fecha, organizador)
   - Estado del bot de Recall.ai
   - Usuarios con acceso y sus permisos (T=Transcripción, A=Audio, V=Video)
   - Transcripción (si está disponible)
   - Archivos de audio/video (si están disponibles)

### Asignar Acceso a Reunión

Para dar acceso a un usuario a una reunión existente:

1. Abre la reunión
2. En la sección **"Usuarios asignados"**, haz clic en **"Asignar Usuario"**
3. Ingresa el email del usuario
4. Selecciona los permisos:
   - ✅ Ver transcripción
   - ✅ Ver audio (solo si el usuario tiene licencia Advanced o PRO)
   - ✅ Ver video (solo si el usuario tiene licencia PRO)
5. Haz clic en **"Asignar"**

**Nota**: Los permisos se calculan automáticamente según la licencia del usuario, pero puedes ajustarlos manualmente si es necesario.

### Eliminar Reunión

Como administrador puedes eliminar reuniones de dos formas:

#### Soft Delete (Recomendado)
- Oculta la reunión pero mantiene todos los datos
- Útil para auditoría y recuperación
- No elimina archivos ni transcripciones

#### Hard Delete
- Elimina permanentemente:
  - La reunión
  - La transcripción y segmentos
  - Los archivos de audio y video del servidor
  - Todos los accesos de usuarios
- **IRREVERSIBLE**: No se puede recuperar

**Procedimiento**:
1. Abre la reunión
2. Haz clic en **"Eliminar"**
3. Selecciona el tipo de eliminación
4. Confirma la acción

---

## Configuración del Sistema

### Variables de Entorno Importantes

Revisa el archivo `.env` del backend para configurar:

#### Administradores
```env
ADMIN_EMAIL_DOMAIN=tu-dominio.com
ADMIN_EMAILS=admin1@tu-dominio.com,admin2@tu-dominio.com
```

#### Recall.ai
```env
RECALL_API_KEY=tu_api_key_de_recall
RECALL_API_URL=https://us-west-2.recall.ai/api/v1
BOT_SERVER_URL=https://tu-dominio.com
```

#### Base de Datos
```env
DATABASE_URL=postgresql://usuario:contraseña@localhost:5432/notetaker
```

### Configuración de Bots

El sistema usa Recall.ai para gestionar los bots. Configuraciones importantes:

- **Bot Display Name**: Nombre que aparece en las reuniones (configurable por usuario)
- **Retención**: Política de retención de grabaciones (configurable)
- **Avatar del Bot**: Imagen personalizada (opcional, configurar en `BOT_AVATAR_IMAGE_URL`)

---

## Monitoreo y Mantenimiento

### Estado de Celery

Celery gestiona las tareas programadas para unir bots a las reuniones:

1. Verifica que el worker de Celery esté corriendo
2. Usa el endpoint de sincronización si es necesario:
   ```
   POST /api/meetings/sync-celery?user_email=admin@tu-dominio.com
   ```

### Sincronizar Reuniones con Celery

Si después de reiniciar Celery necesitas re-programar las reuniones:

```cmd
cd C:\notetaker\backend
sync_meetings_celery.bat
```

O usando la API:
```bash
POST http://localhost:7000/api/meetings/sync-celery?user_email=admin@tu-dominio.com
```

### Purgar Cola de Celery

Si necesitas limpiar la cola de tareas:

```cmd
cd C:\notetaker\backend
purge_celery.bat
```

### Ver Logs

Los logs del sistema se muestran en:
- **Backend**: Consola donde se ejecuta uvicorn
- **Celery**: Consola donde se ejecuta el worker
- **Base de datos**: Revisar tablas para errores registrados

Para producción, considera redirigir logs a archivos.

---

## Gestión de Permisos

### Sistema de Permisos

El sistema calcula permisos automáticamente según la licencia del usuario:

- **Al crear acceso**: Se asignan permisos según la licencia actual
- **Al consultar acceso**: Se recalculan y sincronizan si han cambiado
- **Al cambiar licencia**: Se actualizan todos los accesos existentes

### Permisos por Licencia

| Licencia | Transcripción | Audio | Video |
|----------|---------------|-------|-------|
| Basic    | ✅            | ❌    | ❌    |
| Advanced | ✅            | ✅    | ❌    |
| PRO      | ✅            | ✅    | ✅    |

### Verificar Permisos de Usuario

1. Ve a **"Usuarios"**
2. Selecciona el usuario
3. Verás su licencia actual
4. En **"Reuniones"**, puedes ver los permisos (T/A/V) para cada reunión

---

## Resolución de Problemas

### Usuario no puede acceder a audio/video

1. Verifica la licencia del usuario en **"Usuarios"**
2. Confirma que la licencia sea Advanced (audio) o PRO (audio+video)
3. Si la licencia es correcta, verifica los permisos en la reunión específica
4. Los permisos se sincronizan automáticamente, pero puedes forzar la actualización editando el usuario

### Bot no se une a la reunión

1. Verifica que Celery esté corriendo
2. Revisa el `celery_task_id` de la reunión
3. Verifica los logs de Celery para errores
4. Usa el endpoint de sincronización para re-programar

### Transcripción no se genera

1. Verifica que el bot se haya unido correctamente (estado "completed")
2. Revisa el `recall_bot_id` y `recall_status`
3. Verifica la conexión con Recall.ai
4. Revisa los logs del backend para errores de procesamiento

### Error de permisos en base de datos

Si hay problemas de permisos:

1. Conecta a PostgreSQL como superusuario
2. Verifica que el usuario `notetaker_user` tenga los permisos correctos
3. Ejecuta:
   ```sql
   GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO notetaker_user;
   GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO notetaker_user;
   ```

---

## Seguridad

### Mejores Prácticas

1. **Contraseñas**: Asegúrate de que los usuarios usen contraseñas seguras
2. **Tokens**: Los tokens de OAuth se almacenan encriptados (o mejor, en Redis)
3. **API Keys**: Mantén las API keys de Recall.ai seguras
4. **Base de Datos**: Usa conexiones SSL para producción
5. **Backups**: Realiza backups regulares de la base de datos

### Auditoría

El sistema registra:
- Creación y modificación de usuarios
- Accesos a reuniones
- Cambios de permisos
- Errores y excepciones

Revisa los logs regularmente para detectar problemas.

---

## Backup y Restauración

### Backup de Base de Datos

```cmd
pg_dump -U notetaker_user -d notetaker -F c -f backup_notetaker_YYYYMMDD.dump
```

### Restaurar Base de Datos

```cmd
pg_restore -U notetaker_user -d notetaker -c backup_notetaker_YYYYMMDD.dump
```

### Backup de Archivos

Los archivos de audio y video se almacenan en:
- `backend/storage/audio/`
- `backend/storage/video/`

Realiza backups regulares de estas carpetas.

---

## Actualizaciones

### Actualizar Código

1. Detener servicios (Backend y Celery)
2. Hacer backup de base de datos
3. Actualizar código desde repositorio
4. Actualizar dependencias:
   ```cmd
   cd backend
   notetaker\Scripts\activate
   pip install -r requirements.txt
   ```
5. Ejecutar migraciones si es necesario
6. Reiniciar servicios

### Migraciones de Base de Datos

Si hay cambios en el esquema:

1. Revisar archivos de migración en `backend/alembic/versions/`
2. Aplicar migraciones:
   ```cmd
   cd backend
   alembic upgrade head
   ```

---

## Endpoints de Administración

### API Endpoints Útiles

- `GET /api/auth/users/list?user_email=admin@dominio.com` - Listar usuarios
- `PUT /api/auth/users/{user_id}?user_email=admin@dominio.com` - Actualizar usuario
- `POST /api/auth/admin/create-user?admin_email=admin@dominio.com` - Crear usuario
- `POST /api/auth/admin/reset-password?admin_email=admin@dominio.com` - Resetear contraseña
- `GET /api/meetings/list?user_email=admin@dominio.com` - Listar todas las reuniones
- `POST /api/meetings/{meeting_id}/access` - Asignar acceso a reunión
- `POST /api/meetings/sync-celery?user_email=admin@dominio.com` - Sincronizar Celery

---

## Contacto y Soporte

Para problemas técnicos o consultas:

1. Revisa los logs del sistema
2. Consulta la documentación técnica
3. Contacta al equipo de desarrollo
4. Proporciona:
   - Descripción detallada del problema
   - Logs relevantes
   - Pasos para reproducir
   - Capturas de pantalla si aplica

---

## Glosario Técnico

- **Recall.ai**: Servicio externo que gestiona los bots de grabación
- **Celery**: Sistema de colas para tareas asíncronas
- **MeetingAccess**: Tabla que relaciona usuarios con reuniones y sus permisos
- **Soft Delete**: Eliminación lógica que mantiene los datos
- **Hard Delete**: Eliminación física que borra permanentemente
- **Alembic**: Herramienta de migraciones de base de datos
- **FastAPI**: Framework web del backend
- **PostgreSQL**: Base de datos relacional
- **Redis**: Sistema de caché y colas

---

*Última actualización: Diciembre 2025*

