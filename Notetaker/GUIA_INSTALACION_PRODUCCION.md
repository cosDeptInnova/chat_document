# Guía de Instalación en Producción - Notetaker2.0

## Índice

1. [Requisitos Previos](#requisitos-previos)
2. [Preparación del Servidor](#preparación-del-servidor)
3. [Descompresión e Instalación](#descompresión-e-instalación)
4. [Configuración de Base de Datos](#configuración-de-base-de-datos)
5. [Configuración de Variables de Entorno](#configuración-de-variables-de-entorno)
6. [Instalación de Dependencias](#instalación-de-dependencias)
7. [Inicio de Servicios](#inicio-de-servicios)
8. [Verificación Post-Instalación](#verificación-post-instalación)
9. [Configuración como Servicios de Windows](#configuración-como-servicios-de-windows)
10. [Solución de Problemas](#solución-de-problemas)

---

## Requisitos Previos

Antes de comenzar, asegúrate de tener instalado en el servidor de producción:

- **Python 3.11** o superior
- **Node.js 20.x** o superior (LTS recomendado)
- **PostgreSQL** (versión 14 o superior)
- **Redis** (versión 3.0 o superior)
- **Git** (opcional, solo si necesitas clonar repositorios)

### Verificar Instalaciones

Abre PowerShell o CMD y ejecuta:

```powershell
python --version
node --version
npm --version
psql --version
redis-cli --version
```

Si alguna de estas herramientas no está instalada, instálala antes de continuar.

---

## Preparación del Servidor

### 1. Crear Directorio de Instalación

Crea un directorio donde se instalará la aplicación:

```powershell
# Ejemplo: Crear en C:\notetaker
New-Item -ItemType Directory -Path "C:\notetaker" -Force
cd C:\notetaker
```

### 2. Verificar Conexión a Base de Datos

Verifica que puedes conectarte a la base de datos PostgreSQL:

```powershell
# Reemplaza con tus credenciales reales
psql -h localhost -U cosmos_user -d cosmos_notetaker -c "SELECT version();"
```

Si la conexión falla, verifica:
- Que PostgreSQL esté corriendo
- Que el usuario y contraseña sean correctos
- Que la base de datos `cosmos_notetaker` exista

### 3. Verificar Redis

Verifica que Redis esté corriendo:

```powershell
redis-cli ping
```

Debe responder: `PONG`

Si Redis no está corriendo, inícialo:

```powershell
# Si está instalado como servicio
net start Redis

# O si está en un puerto diferente (ej: 6380)
redis-cli -p 6380 ping
```

---

## Descompresión e Instalación

### 1. Descomprimir el ZIP

1. Copia el archivo `notetaker2.0-produccion-*.zip` al servidor de producción
2. Descomprime el archivo en el directorio creado anteriormente:

```powershell
# Ejemplo: si el ZIP está en C:\temp\
Expand-Archive -Path "C:\temp\notetaker2.0-produccion-*.zip" -DestinationPath "C:\notetaker" -Force
```

3. Verifica la estructura:

```powershell
cd C:\notetaker
Get-ChildItem
```

Debes ver:
- `backend/` - Código del backend Python
- `frontend/` - Código del frontend React
- `docs/` - Documentación
- Archivos `.md` de documentación

---

## Configuración de Base de Datos

### Verificar Tablas Existentes

Como mencionaste que la base de datos ya existe con las mismas tablas, verifica que las tablas estén presentes:

```powershell
psql -h localhost -U cosmos_user -d cosmos_notetaker -c "\dt"
```

Debes ver tablas como:
- `users`
- `meetings`
- `transcriptions`
- `transcription_segments`
- `summaries`
- `meeting_access`

### Si Faltan Tablas

Si alguna tabla falta, puedes ejecutar el script de creación (solo creará las que no existan):

```powershell
cd C:\notetaker\backend
python create_tables.py
```

**NOTA:** Este script usa SQLAlchemy para crear tablas. Asegúrate de tener el `.env` configurado correctamente antes de ejecutarlo.

---

## Configuración de Variables de Entorno

### 1. Configurar Backend (.env)

1. **Copiar archivo de ejemplo:**

```powershell
cd C:\notetaker\backend
Copy-Item env.example .env
```

2. **Editar el archivo `.env`** con tus valores de producción:

```powershell
notepad .env
```

**Configuración mínima requerida:**

```env
# Base de datos (usar las mismas credenciales que mencionaste)
DATABASE_URL=postgresql://cosmos_user:Cos4321@localhost:5432/cosmos_notetaker

# Redis (ajustar puerto si es diferente)
REDIS_URL=redis://localhost:6380/0

# Secret Keys (generar nuevas para producción)
SECRET_KEY="tu_secret_key_muy_segura_minimo_32_caracteres"
JWT_SECRET_KEY="tu_jwt_secret_key_muy_segura_minimo_32_caracteres"

# Debug debe estar en False en producción
DEBUG=False

# URL del frontend en producción
FRONTEND_URL=https://tu-dominio-frontend.com

# VEXA API (ajustar según tu servidor VEXA)
VEXA_API_BASE_URL=http://172.29.14.10:8056
VEXA_API_KEY=tu_vexa_api_key_aqui

# Administradores
ADMIN_EMAILS=fdescartin@cosgs.com,rcarro@cosgs.com

# Microsoft Graph API (si aplica)
GRAPH_TENANT_ID=e0b4b780-6188-44ad-a603-f87d1bf174f4
GRAPH_CLIENT_ID=a268e492-2d53-4c56-a5b7-f6fd631739c8
GRAPH_CLIENT_SECRET=~IV8Q~W52WFOtNbZpYt~5vwZ5z6kWuhPqzDE5dkA
GRAPH_USER_EMAIL=rpa@cosgs.com
ADMIN_DEFAULT_PASSWORD=C@sm@s-2025

# Google OAuth (si aplica)
GOOGLE_CLIENT_ID=3427560620-rv3qho8lgj5bq05ojr61o0p1pud5fb3q.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=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
GOOGLE_REDIRECT_URI=https://tu-dominio-backend.com/api/integrations/oauth/callback/google

# Outlook OAuth (si aplica)
OUTLOOK_CLIENT_ID=8599880b-3ba5-42f6-b04e-63b1501eac45
OUTLOOK_CLIENT_SECRET=rMT8Q~PntdKKB7IJdqb9S~HtbZ2xPu8jK9bd6a57
OUTLOOK_TENANT_ID=e0b4b780-6188-44ad-a603-f87d1bf174f4
OUTLOOK_REDIRECT_URI=https://tu-dominio-backend.com/api/integrations/oauth/callback/outlook

# URL pública del backend (para webhooks)
BACKEND_PUBLIC_URL=https://tu-dominio-backend.com

# Storage
STORAGE_TYPE=local
AUDIO_STORAGE_PATH=./storage/audio

# SSL Verification (ajustar según necesidad)
SSL_VERIFY=False
```

**IMPORTANTE:**
- Reemplaza `tu-dominio-backend.com` y `tu-dominio-frontend.com` con tus URLs reales
- Genera nuevas `SECRET_KEY` y `JWT_SECRET_KEY` seguras para producción
- Ajusta las URLs de VEXA según tu configuración
- Verifica que todas las API keys sean correctas

### 2. Configurar Frontend (.env)

1. **Crear archivo `.env` en frontend:**

```powershell
cd C:\notetaker\frontend
New-Item -ItemType File -Path ".env" -Force
```

2. **Editar el archivo `.env`:**

```powershell
notepad .env
```

**Contenido:**

```env
VITE_API_BASE_URL=https://tu-dominio-backend.com
```

**NOTA:** Reemplaza con la URL real de tu backend en producción.

---

## Instalación de Dependencias

### 1. Instalar Dependencias del Backend

1. **Navegar al directorio backend:**

```powershell
cd C:\notetaker\backend
```

2. **Crear entorno virtual:**

```powershell
python -m venv notetaker
```

3. **Activar entorno virtual:**

```powershell
.\notetaker\Scripts\Activate.ps1
```

Si PowerShell muestra un error de política de ejecución, ejecuta primero:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

4. **Actualizar pip:**

```powershell
python -m pip install --upgrade pip
```

5. **Instalar dependencias:**

```powershell
pip install -r requirements.txt
```

Esto puede tardar varios minutos. Espera a que termine completamente.

6. **Instalar Playwright (si es necesario):**

```powershell
playwright install
```

### 2. Instalar Dependencias del Frontend

1. **Navegar al directorio frontend:**

```powershell
cd C:\notetaker\frontend
```

2. **Instalar dependencias:**

```powershell
npm install
```

Esto puede tardar varios minutos.

3. **Compilar para producción:**

```powershell
npm run build
```

Esto generará los archivos estáticos en la carpeta `dist/`.

---

## Inicio de Servicios

### 1. Verificar Servicios Previos

Antes de iniciar la aplicación, asegúrate de que estos servicios estén corriendo:

```powershell
# Verificar PostgreSQL
Get-Service -Name postgresql*

# Verificar Redis
redis-cli ping
```

### 2. Iniciar Backend (FastAPI)

**Opción A: Manualmente (para pruebas)**

Abre una ventana de PowerShell:

```powershell
cd C:\notetaker\backend
.\notetaker\Scripts\Activate.ps1
uvicorn app.main:app --host 0.0.0.0 --port 7000
```

**Opción B: Usando script batch**

```powershell
cd C:\notetaker\backend
.\start_server.bat
```

**NOTA:** El script `start_server.bat` usa `--reload` que es para desarrollo. Para producción, modifica el script o ejecuta manualmente sin `--reload`:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 7000
```

### 3. Iniciar Celery Worker

Abre otra ventana de PowerShell:

```powershell
cd C:\notetaker\backend
.\notetaker\Scripts\Activate.ps1
celery -A app.celery_app worker --loglevel=info --pool=solo
```

O usando el script batch:

```powershell
cd C:\notetaker\backend
.\start_celery.bat
```

### 4. Servir Frontend

**Opción A: Usando el backend FastAPI (recomendado)**

El backend puede servir los archivos estáticos del frontend. Configura esto en tu servidor web (Nginx, IIS) o usa el backend directamente.

**Opción B: Servidor web independiente**

Si prefieres servir el frontend por separado, puedes usar un servidor web como Nginx o IIS apuntando a la carpeta `frontend/dist/`.

---

## Verificación Post-Instalación

### 1. Verificar Backend

1. Abre un navegador y ve a: `http://localhost:7000/docs`
2. Debe mostrarse la documentación interactiva de la API (Swagger UI)
3. Prueba un endpoint simple como `GET /api/health` (si existe)

### 2. Verificar Frontend

1. Si el frontend está siendo servido, abre la URL correspondiente
2. Debe mostrarse la página de login
3. Verifica que no haya errores en la consola del navegador (F12)

### 3. Verificar Celery

1. Verifica que el worker de Celery esté corriendo y escuchando tareas
2. Revisa los logs del worker para asegurarte de que no hay errores

### 4. Verificar Conexión a Base de Datos

```powershell
cd C:\notetaker\backend
.\notetaker\Scripts\Activate.ps1
python -c "from app.database import engine; from sqlalchemy import text; conn = engine.connect(); result = conn.execute(text('SELECT COUNT(*) FROM users')); print(f'Usuarios en BD: {result.scalar()}'); conn.close()"
```

### 5. Verificar Redis

```powershell
redis-cli ping
redis-cli -p 6380 ping  # Si usas puerto diferente
```

---

## Configuración como Servicios de Windows

Para producción, es recomendable ejecutar el backend y Celery como servicios de Windows para que se inicien automáticamente.

### Instalar NSSM (Non-Sucking Service Manager)

1. Descarga NSSM desde: https://nssm.cc/download
2. Extrae `nssm.exe` (versión de 64 bits)
3. Copia `nssm.exe` a `C:\Windows\System32\` o a una carpeta en el PATH

### Crear Servicio para Backend

1. **Abrir PowerShell como Administrador**

2. **Instalar servicio:**

```powershell
nssm install NotetakerBackend
```

3. **En la ventana que aparece, configurar:**

- **Path:** `C:\notetaker\backend\notetaker\Scripts\python.exe`
- **Startup directory:** `C:\notetaker\backend`
- **Arguments:** `-m uvicorn app.main:app --host 0.0.0.0 --port 7000`

4. **En la pestaña "Environment", agregar variables si es necesario:**

```
PYTHONPATH=C:\notetaker\backend
```

5. **Click en "Install service"**

6. **Iniciar servicio:**

```powershell
net start NotetakerBackend
```

### Crear Servicio para Celery

1. **Instalar servicio:**

```powershell
nssm install NotetakerCelery
```

2. **Configurar:**

- **Path:** `C:\notetaker\backend\notetaker\Scripts\python.exe`
- **Startup directory:** `C:\notetaker\backend`
- **Arguments:** `-m celery -A app.celery_app worker --loglevel=info --pool=solo`

3. **Instalar e iniciar:**

```powershell
net start NotetakerCelery
```

### Gestionar Servicios

```powershell
# Ver estado
Get-Service NotetakerBackend
Get-Service NotetakerCelery

# Detener
net stop NotetakerBackend
net stop NotetakerCelery

# Iniciar
net start NotetakerBackend
net start NotetakerCelery

# Reiniciar
Restart-Service NotetakerBackend
Restart-Service NotetakerCelery
```

---

## Solución de Problemas

### Error: "Module not found"

**Causa:** El entorno virtual no está activado o las dependencias no están instaladas.

**Solución:**

```powershell
cd C:\notetaker\backend
.\notetaker\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Error: "Connection refused" en Redis

**Causa:** Redis no está corriendo o está en un puerto diferente.

**Solución:**

```powershell
# Verificar si Redis está corriendo
redis-cli ping

# Si no responde, iniciar Redis
net start Redis

# O verificar puerto diferente
redis-cli -p 6380 ping
```

### Error: "Connection refused" en PostgreSQL

**Causa:** PostgreSQL no está corriendo o las credenciales son incorrectas.

**Solución:**

```powershell
# Verificar que PostgreSQL esté corriendo
Get-Service -Name postgresql*

# Probar conexión manual
psql -h localhost -U cosmos_user -d cosmos_notetaker

# Verificar DATABASE_URL en .env
```

### Error: "Celery worker no responde"

**Causa:** Redis no está disponible o hay un problema con la configuración.

**Solución:**

1. Verificar que Redis esté corriendo
2. Verificar `REDIS_URL` en `.env`
3. Reiniciar el worker de Celery
4. Revisar logs del worker

### Error: "Port 7000 already in use"

**Causa:** Otro proceso está usando el puerto 7000.

**Solución:**

```powershell
# Ver qué proceso usa el puerto
netstat -ano | findstr :7000

# Detener el proceso o cambiar el puerto en la configuración
```

### Error al compilar frontend

**Causa:** Problemas con dependencias de Node.js.

**Solución:**

```powershell
cd C:\notetaker\frontend
# Limpiar e reinstalar
Remove-Item -Recurse -Force node_modules
Remove-Item -Force package-lock.json
npm install
npm run build
```

### Error: "Permission denied" al crear servicios

**Causa:** No tienes permisos de administrador.

**Solución:** Ejecuta PowerShell como Administrador.

---

## Comandos Útiles

### Ver logs del backend

Si el backend está corriendo como servicio, los logs pueden estar en:
- La consola donde se ejecuta (si es manual)
- Archivos de log si están configurados
- Event Viewer de Windows (si está configurado)

### Sincronizar reuniones con Celery

```powershell
cd C:\notetaker\backend
.\notetaker\Scripts\Activate.ps1
python sync_meetings_celery.py
```

### Purgar cola de Celery

```powershell
cd C:\notetaker\backend
.\purge_celery.bat
```

---

## Próximos Pasos

Después de la instalación exitosa:

1. **Configurar SSL/HTTPS** si aún no está configurado
2. **Configurar firewall** para permitir tráfico en los puertos necesarios
3. **Configurar backup** de la base de datos
4. **Configurar monitoreo** de los servicios
5. **Revisar documentación adicional** en la carpeta `docs/`

---

## Contacto y Soporte

Para problemas o dudas adicionales, consulta:
- Documentación en la carpeta `docs/`
- Archivos `README.md` en backend y frontend
- Manuales de administración y usuario en `backend/`

---

**Última actualización:** Febrero 2026
