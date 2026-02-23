# Manual de Instalación en Producción - Notetaker

## Índice

1. [Requisitos del Sistema](#requisitos-del-sistema)
2. [Instalación de Componentes](#instalación-de-componentes)
   - [Python 3.11](#python-311)
   - [Node.js y npm](#nodejs-y-npm)
   - [PostgreSQL](#postgresql)
   - [Redis](#redis)
   - [Docker Desktop](#docker-desktop)
   - [Celery](#celery)
3. [Configuración de Base de Datos](#configuración-de-base-de-datos)
4. [Instalación del Proyecto](#instalación-del-proyecto)
5. [Configuración de Variables de Entorno](#configuración-de-variables-de-entorno)
6. [Inicio de Servicios](#inicio-de-servicios)
7. [Mantenimiento](#mantenimiento)

---

## Requisitos del Sistema

- **Sistema Operativo:** Windows Server 2016 o superior
- **RAM:** Mínimo 8GB (recomendado 16GB)
- **Disco:** Mínimo 50GB libres
- **Red:** Acceso a Internet para descargar dependencias

---

## Instalación de Componentes

### Python 3.11

#### Descarga e Instalación

1. **Descargar Python 3.11:**
   - URL: https://www.python.org/downloads/release/python-3110/
   - Descargar: `Windows installer (64-bit)` o `Windows installer (32-bit)` según tu sistema

2. **Instalación:**
   - Ejecutar el instalador descargado
   - **IMPORTANTE:** Marcar la casilla "Add Python to PATH"
   - Seleccionar "Install Now" o "Customize installation"
   - Si eliges "Customize", asegúrate de marcar:
     - ✅ pip
     - ✅ tcl/tk and IDLE
     - ✅ Python test suite
     - ✅ py launcher
     - ✅ for all users (opcional)

3. **Verificar instalación:**
   ```cmd
   python --version
   ```
   Debe mostrar: `Python 3.11.x`

4. **Verificar pip:**
   ```cmd
   pip --version
   ```

---

### Node.js y npm

#### Descarga e Instalación

1. **Descargar Node.js:**
   - URL: https://nodejs.org/
   - Descargar la versión LTS (Long Term Support) - Recomendado: v20.x o superior
   - Archivo: `node-v20.x.x-x64.msi` (Windows Installer)

2. **Instalación:**
   - Ejecutar el instalador
   - Seguir el asistente de instalación (aceptar valores por defecto)
   - Asegurarse de que npm se instale automáticamente

3. **Verificar instalación:**
   ```cmd
   node --version
   npm --version
   ```

---

### PostgreSQL

#### Descarga e Instalación

1. **Descargar PostgreSQL:**
   - URL: https://www.postgresql.org/download/windows/
   - Descargar el instalador de EnterpriseDB
   - Versión recomendada: PostgreSQL 14 o superior

2. **Instalación:**
   - Ejecutar el instalador
   - Durante la instalación:
     - **Puerto:** 5432 (por defecto)
     - **Superusuario (postgres):** Establecer una contraseña segura y guardarla
     - **Locale:** Spanish, Spain o English, United States
   - Completar la instalación

3. **Verificar instalación:**
   ```cmd
   psql --version
   ```

4. **Crear base de datos:**
   - Abrir pgAdmin (instalado con PostgreSQL) o usar línea de comandos:
   ```cmd
   psql -U postgres
   ```
   - Ejecutar:
   ```sql
   CREATE DATABASE notetaker;
   ```
   - Salir: `\q`

---

### Redis

#### Opción 1: Redis para Windows (Recomendado)

1. **Descargar Redis para Windows:**
   - URL: https://github.com/microsoftarchive/redis/releases
   - Descargar: `Redis-x64-3.0.504.msi` o versión más reciente

2. **Instalación:**
   - Ejecutar el instalador
   - Aceptar valores por defecto
   - Redis se instalará como servicio de Windows

3. **Verificar instalación:**
   ```cmd
   redis-cli ping
   ```
   Debe responder: `PONG`

#### Opción 2: Redis con Docker (Alternativa)

Si prefieres usar Docker:
```cmd
docker run -d -p 6379:6379 --name redis redis:latest
```

---

### Docker Desktop

#### Descarga e Instalación

1. **Descargar Docker Desktop:**
   - URL: https://www.docker.com/products/docker-desktop/
   - Descargar: `Docker Desktop Installer.exe`

2. **Requisitos previos:**
   - Windows 10 64-bit: Pro, Enterprise, o Education (Build 19041 o superior)
   - Windows 11 64-bit: Home o Pro versión 21H2 o superior
   - Habilitar WSL 2 (Windows Subsystem for Linux 2)

3. **Instalación:**
   - Ejecutar el instalador
   - Seguir el asistente
   - Reiniciar el equipo si se solicita
   - Al iniciar Docker Desktop, aceptar los términos de servicio

4. **Verificar instalación:**
   ```cmd
   docker --version
   docker-compose --version
   ```

---

### Celery

Celery se instalará automáticamente cuando instales las dependencias de Python del proyecto (ver sección de instalación del proyecto).

---

## Configuración de Base de Datos

### 1. Crear Base de Datos y Usuario

1. **Conectar a PostgreSQL como superusuario:**
   ```cmd
   psql -U postgres
   ```

2. **Crear usuario y base de datos:**
   ```sql
   -- Crear usuario para la aplicación (reemplazar 'password_segura' con una contraseña fuerte)
   CREATE USER notetaker_user WITH PASSWORD 'password_segura';
   
   -- Crear base de datos
   CREATE DATABASE notetaker OWNER notetaker_user;
   
   -- Otorgar privilegios
   GRANT ALL PRIVILEGES ON DATABASE notetaker TO notetaker_user;
   
   -- Conectar a la base de datos
   \c notetaker
   
   -- Otorgar privilegios en el esquema público
   GRANT ALL ON SCHEMA public TO notetaker_user;
   GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO notetaker_user;
   GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO notetaker_user;
   
   -- Configurar permisos por defecto para tablas futuras
   ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO notetaker_user;
   ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO notetaker_user;
   ```

3. **Salir de psql:**
   ```sql
   \q
   ```

### 2. Crear Tablas

1. **Conectar a la base de datos con el usuario creado:**
   ```cmd
   psql -U notetaker_user -d notetaker
   ```

2. **Ejecutar el siguiente script SQL para crear todas las tablas:**
   ```sql
   -- Habilitar extensión UUID (si no está habilitada)
   CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
   
   -- Tabla: users
   CREATE TABLE users (
       id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::text,
       microsoft_user_id VARCHAR UNIQUE NOT NULL,
       tenant_id VARCHAR NOT NULL,
       email VARCHAR NOT NULL,
       display_name VARCHAR,
       bot_display_name VARCHAR DEFAULT 'Notetaker',
       settings JSONB DEFAULT '{}',
       access_token_encrypted VARCHAR,
       refresh_token_encrypted VARCHAR,
       token_expires_at TIMESTAMP,
       is_active BOOLEAN DEFAULT TRUE,
       is_premium BOOLEAN DEFAULT FALSE,
       hashed_password VARCHAR,
       password_reset_token VARCHAR,
       password_reset_expires TIMESTAMP,
       must_change_password BOOLEAN DEFAULT FALSE,
       created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
       updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
       last_login_at TIMESTAMP
   );
   
   CREATE INDEX idx_users_microsoft_user_id ON users(microsoft_user_id);
   CREATE INDEX idx_users_tenant_id ON users(tenant_id);
   CREATE INDEX idx_users_email ON users(email);
   CREATE INDEX idx_users_password_reset_token ON users(password_reset_token);
   
   -- Tabla: meetings
   CREATE TYPE meeting_status AS ENUM ('pending', 'joining', 'in_progress', 'completed', 'failed', 'cancelled');
   
   CREATE TABLE meetings (
       id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::text,
       user_id VARCHAR REFERENCES users(id),
       title VARCHAR,
       meeting_url TEXT NOT NULL,
       thread_id VARCHAR,
       organizer_email VARCHAR,
       organizer_name VARCHAR,
       scheduled_start_time TIMESTAMP NOT NULL,
       scheduled_end_time TIMESTAMP,
       actual_start_time TIMESTAMP,
       actual_end_time TIMESTAMP,
       status meeting_status DEFAULT 'pending' NOT NULL,
       extra_metadata JSONB DEFAULT '{}',
       audio_file_path VARCHAR,
       video_file_path VARCHAR,
       storage_type VARCHAR DEFAULT 'local',
       recall_bot_id VARCHAR,
       recall_status VARCHAR,
       recall_storage_plan VARCHAR DEFAULT 'none',
       recall_metadata JSONB DEFAULT '{}',
       celery_task_id VARCHAR,
       error_message TEXT,
       deleted_at TIMESTAMP,
       created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
       updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
   );
   
   CREATE INDEX idx_meetings_user_id ON meetings(user_id);
   CREATE INDEX idx_meetings_thread_id ON meetings(thread_id);
   CREATE INDEX idx_meetings_scheduled_start_time ON meetings(scheduled_start_time);
   CREATE INDEX idx_meetings_status ON meetings(status);
   CREATE INDEX idx_meetings_recall_bot_id ON meetings(recall_bot_id);
   CREATE INDEX idx_meetings_celery_task_id ON meetings(celery_task_id);
   CREATE INDEX idx_meetings_deleted_at ON meetings(deleted_at);
   
   -- Tabla: meeting_access (relación many-to-many con permisos)
   CREATE TABLE meeting_access (
       id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::text,
       meeting_id VARCHAR NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
       user_id VARCHAR NOT NULL REFERENCES users(id) ON DELETE CASCADE,
       can_view_transcript BOOLEAN DEFAULT TRUE NOT NULL,
       can_view_audio BOOLEAN DEFAULT FALSE NOT NULL,
       can_view_video BOOLEAN DEFAULT FALSE NOT NULL,
       created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
       UNIQUE(meeting_id, user_id)
   );
   
   CREATE INDEX idx_meeting_access_meeting_id ON meeting_access(meeting_id);
   CREATE INDEX idx_meeting_access_user_id ON meeting_access(user_id);
   
   -- Tabla: transcriptions
   CREATE TABLE transcriptions (
       id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::text,
       meeting_id VARCHAR NOT NULL UNIQUE REFERENCES meetings(id) ON DELETE CASCADE,
       language VARCHAR DEFAULT 'es',
       confidence_score REAL,
       raw_transcript_json JSONB,
       total_segments INTEGER DEFAULT 0,
       total_duration_seconds REAL DEFAULT 0.0,
       is_final BOOLEAN DEFAULT FALSE,
       is_processed BOOLEAN DEFAULT FALSE,
       started_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
       completed_at TIMESTAMP,
       created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
       updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
   );
   
   CREATE INDEX idx_transcriptions_meeting_id ON transcriptions(meeting_id);
   
   -- Tabla: transcription_segments
   CREATE TABLE transcription_segments (
       id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::text,
       transcription_id VARCHAR NOT NULL REFERENCES transcriptions(id) ON DELETE CASCADE,
       speaker_id VARCHAR NOT NULL,
       speaker_name VARCHAR,
       text TEXT NOT NULL,
       language VARCHAR DEFAULT 'es',
       start_time REAL NOT NULL,
       end_time REAL NOT NULL,
       duration REAL,
       confidence REAL,
       words JSONB,
       created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
   );
   
   CREATE INDEX idx_transcription_segments_transcription_id ON transcription_segments(transcription_id);
   CREATE INDEX idx_transcription_segments_speaker_id ON transcription_segments(speaker_id);
   CREATE INDEX idx_transcription_segments_speaker_name ON transcription_segments(speaker_name);
   CREATE INDEX idx_transcription_segments_start_time ON transcription_segments(start_time);
   
   -- Tabla: summaries
   CREATE TABLE summaries (
       id VARCHAR PRIMARY KEY DEFAULT gen_random_uuid()::text,
       meeting_id VARCHAR NOT NULL UNIQUE REFERENCES meetings(id) ON DELETE CASCADE,
       transcription_id VARCHAR REFERENCES transcriptions(id),
       summary_text TEXT,
       summary_json JSONB,
       key_points JSONB,
       action_items JSONB,
       participants_summary JSONB,
       sentiment_analysis JSONB,
       llm_model VARCHAR,
       llm_service VARCHAR,
       processing_time_seconds REAL,
       is_final BOOLEAN DEFAULT FALSE,
       error_message TEXT,
       created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
       updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
       completed_at TIMESTAMP
   );
   
   CREATE INDEX idx_summaries_meeting_id ON summaries(meeting_id);
   CREATE INDEX idx_summaries_transcription_id ON summaries(transcription_id);
   
   -- Función para actualizar updated_at automáticamente
   CREATE OR REPLACE FUNCTION update_updated_at_column()
   RETURNS TRIGGER AS $$
   BEGIN
       NEW.updated_at = CURRENT_TIMESTAMP;
       RETURN NEW;
   END;
   $$ language 'plpgsql';
   
   -- Triggers para actualizar updated_at
   CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
       FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
   
   CREATE TRIGGER update_meetings_updated_at BEFORE UPDATE ON meetings
       FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
   
   CREATE TRIGGER update_transcriptions_updated_at BEFORE UPDATE ON transcriptions
       FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
   
   CREATE TRIGGER update_summaries_updated_at BEFORE UPDATE ON summaries
       FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
   ```

3. **Verificar tablas creadas:**
   ```sql
   \dt
   ```
   
   Deben aparecer las siguientes tablas:
   - users
   - meetings
   - meeting_access
   - transcriptions
   - transcription_segments
   - summaries

4. **Verificar permisos:**
   ```sql
   \dp
   ```

5. **Salir de psql:**
   ```sql
   \q
   ```

### 3. Actualizar DATABASE_URL en .env

Después de crear el usuario y la base de datos, actualizar el archivo `.env` del backend:

```env
DATABASE_URL=postgresql://notetaker_user:password_segura@localhost:5432/notetaker
```

**IMPORTANTE:** Reemplazar `password_segura` con la contraseña real que configuraste en el paso 1.

---

## Instalación del Proyecto

### 1. Extraer el paquete ZIP

1. Extraer el archivo `notetaker-production.zip` en la ubicación deseada (ej: `C:\notetaker\`)

2. La estructura debe quedar así:
   ```
   C:\notetaker\
   ├── backend\
   │   ├── app\
   │   ├── scripts\
   │   ├── *.bat
   │   ├── *.py
   │   └── requirements.txt
   ├── frontend\
   │   ├── src\
   │   ├── package.json
   │   └── ...
   └── MANUAL_INSTALACION_PRODUCCION.md
   ```

### 2. Instalar Backend (Python)

1. **Abrir PowerShell o CMD como Administrador**

2. **Navegar al directorio backend:**
   ```cmd
   cd C:\notetaker\backend
   ```

3. **Crear entorno virtual:**
   ```cmd
   python -m venv notetaker
   ```

4. **Activar entorno virtual:**
   ```cmd
   notetaker\Scripts\activate
   ```

5. **Actualizar pip:**
   ```cmd
   python -m pip install --upgrade pip
   ```

6. **Instalar dependencias:**
   ```cmd
   pip install -r requirements.txt
   ```

7. **Instalar Playwright (requerido para automatización de navegador):**
   ```cmd
   playwright install
   ```

### 3. Instalar Frontend (Node.js)

1. **Abrir PowerShell o CMD**

2. **Navegar al directorio frontend:**
   ```cmd
   cd C:\notetaker\frontend
   ```

3. **Instalar dependencias:**
   ```cmd
   npm install
   ```

4. **Compilar para producción:**
   ```cmd
   npm run build
   ```

   Esto generará los archivos en la carpeta `dist\`

---

## Configuración de Variables de Entorno

### Backend (.env)

1. **Copiar archivo de ejemplo:**
   ```cmd
   cd C:\notetaker\backend
   copy env.example .env
   ```

2. **Editar `.env` con tus valores:**
   ```env
   # Base de datos PostgreSQL
   DATABASE_URL=postgresql://usuario:contraseña@localhost:5432/notetaker
   
   # Redis
   REDIS_URL=redis://localhost:6379/0
   
   # Secret Key (generar uno nuevo y seguro)
   SECRET_KEY=tu_clave_secreta_muy_segura_aqui
   
   # JWT Secret (generar uno nuevo y seguro)
   JWT_SECRET=tu_jwt_secret_muy_seguro_aqui
   
   # Admin Configuration
   ADMIN_EMAIL_DOMAIN=tu-dominio.com
   ADMIN_EMAILS=admin1@tu-dominio.com,admin2@tu-dominio.com
   
   # Recall.ai API
   RECALL_API_KEY=tu_recall_api_key
   RECALL_API_URL=https://us-west-2.recall.ai/api/v1
   
   # Bot Server URL (URL pública de tu backend)
   BOT_SERVER_URL=https://tu-dominio.com
   
   # Email (opcional)
   SMTP_HOST=smtp.gmail.com
   SMTP_PORT=587
   SMTP_USER=tu_email@gmail.com
   SMTP_PASSWORD=tu_contraseña_app
   SMTP_FROM=tu_email@gmail.com
   
   # Google OAuth (opcional)
   GOOGLE_CLIENT_ID=tu_google_client_id
   GOOGLE_CLIENT_SECRET=tu_google_client_secret
   
   # Outlook OAuth (opcional)
   OUTLOOK_CLIENT_ID=tu_outlook_client_id
   OUTLOOK_CLIENT_SECRET=tu_outlook_client_secret
   ```

### Frontend (.env)

1. **Crear archivo `.env` en el directorio frontend:**
   ```cmd
   cd C:\notetaker\frontend
   ```

2. **Contenido del archivo:**
   ```env
   VITE_API_BASE_URL=http://localhost:7000
   ```

   **Para producción, cambiar a:**
   ```env
   VITE_API_BASE_URL=https://tu-dominio-backend.com
   ```

3. **Recompilar después de cambiar .env:**
   ```cmd
   npm run build
   ```

---

## Inicio de Servicios

### 1. Iniciar Redis

Si Redis está instalado como servicio de Windows, debería iniciarse automáticamente.

**Verificar estado:**
```cmd
redis-cli ping
```

**Si no está corriendo, iniciar manualmente:**
```cmd
net start Redis
```

### 2. Iniciar Backend (FastAPI)

**Opción A: Usando script batch (Recomendado para desarrollo):**
```cmd
cd C:\notetaker\backend
start_server.bat
```

**Opción B: Como servicio de Windows (Recomendado para producción):**
Ver sección de "Configuración como Servicio" más abajo.

**Opción C: Manualmente:**
```cmd
cd C:\notetaker\backend
notetaker\Scripts\activate
uvicorn app.main:app --host 0.0.0.0 --port 7000
```

### 3. Iniciar Celery Worker

**Opción A: Usando script batch:**
```cmd
cd C:\notetaker\backend
start_celery.bat
```

**Opción B: Como servicio de Windows:**
```cmd
cd C:\notetaker\backend
install_celery_service.bat
```

**Opción C: Manualmente:**
```cmd
cd C:\notetaker\backend
notetaker\Scripts\activate
celery -A app.celery_app worker --loglevel=info --pool=solo
```

### 4. Iniciar Frontend (React)

**Para desarrollo:**
```cmd
cd C:\notetaker\frontend
npm run dev
```

**Para producción (servir archivos estáticos):**
Usar un servidor web como IIS, Nginx, o servir desde el backend FastAPI.

---

## Configuración como Servicio de Windows

### Backend como Servicio

1. **Instalar NSSM (Non-Sucking Service Manager):**
   - Descargar: https://nssm.cc/download
   - Extraer y copiar `nssm.exe` a `C:\Windows\System32\`

2. **Crear servicio para Backend:**
   ```cmd
   nssm install NotetakerBackend
   ```
   
   En la ventana que aparece:
   - **Path:** `C:\notetaker\backend\notetaker\Scripts\python.exe`
   - **Startup directory:** `C:\notetaker\backend`
   - **Arguments:** `-m uvicorn app.main:app --host 0.0.0.0 --port 7000`
   - **Service name:** `NotetakerBackend`
   - Click en "Install service"

3. **Iniciar servicio:**
   ```cmd
   net start NotetakerBackend
   ```

### Celery como Servicio

Ya existe un script para esto:
```cmd
cd C:\notetaker\backend
install_celery_service.bat
```

---

## Mantenimiento

### Sincronizar Reuniones con Celery

Si después de purgar Celery necesitas re-programar las reuniones:

```cmd
cd C:\notetaker\backend
sync_meetings_celery.bat
```

O usando el endpoint API:
```bash
POST http://localhost:7000/api/meetings/sync-celery?user_email=admin@tu-dominio.com
```

### Purgar Cola de Celery

```cmd
cd C:\notetaker\backend
purge_celery.bat
```

### Reiniciar Servicios

```cmd
net stop NotetakerBackend
net start NotetakerBackend

net stop NotetakerCelery
net start NotetakerCelery
```

### Ver Logs

Los logs del backend se muestran en la consola donde se ejecuta. Para producción, considera redirigir la salida a archivos:

```cmd
uvicorn app.main:app --host 0.0.0.0 --port 7000 >> logs\backend.log 2>&1
```

---

## Verificación Post-Instalación

1. **Verificar Backend:**
   - Abrir navegador: http://localhost:7000/docs
   - Debe mostrar la documentación de la API

2. **Verificar Frontend:**
   - Abrir navegador: http://localhost:5173 (desarrollo) o la URL de producción
   - Debe mostrar la interfaz de login

3. **Verificar Celery:**
   - Verificar que el worker esté corriendo y escuchando tareas

4. **Verificar Redis:**
   ```cmd
   redis-cli ping
   ```

5. **Verificar Base de Datos:**
   ```cmd
   psql -U postgres -d notetaker -c "\dt"
   ```

---

## Solución de Problemas

### Error: "Module not found"
- Verificar que el entorno virtual esté activado
- Reinstalar dependencias: `pip install -r requirements.txt`

### Error: "Connection refused" en Redis
- Verificar que Redis esté corriendo: `redis-cli ping`
- Iniciar Redis: `net start Redis`

### Error: "Connection refused" en PostgreSQL
- Verificar que PostgreSQL esté corriendo
- Verificar credenciales en `.env`
- Verificar que la base de datos `notetaker` exista

### Error: "Celery worker no responde"
- Verificar que Redis esté corriendo
- Reiniciar el worker de Celery
- Verificar logs del worker

---

## Contacto y Soporte

Para problemas o dudas, consultar la documentación del proyecto o contactar al equipo de desarrollo.

