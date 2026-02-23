# Instalación de Celery como Servicio de Windows

Este documento explica cómo instalar Celery como un servicio de Windows que se inicia automáticamente.

## Requisitos

1. **NSSM (Non-Sucking Service Manager)**
   - Descarga desde: https://nssm.cc/download
   - Extrae `nssm.exe` en una carpeta (ej: `C:\nssm`)
   - Añade esa carpeta al PATH de Windows O copia `nssm.exe` en la carpeta `backend`

## Instalación

### Opción 1: Script Automático (Recomendado)

1. Ejecuta `install_celery_service.bat` como **Administrador**
2. El script te guiará paso a paso
3. El servicio se instalará con el nombre `CosmosNotetakerCelery`
4. Se configurará para iniciarse automáticamente con Windows

### Opción 2: Instalación Manual

Si prefieres hacerlo manualmente:

```batch
REM 1. Abre PowerShell o CMD como Administrador
REM 2. Navega al directorio backend
cd E:\notetaker\backend

REM 3. Instala el servicio
nssm install CosmosNotetakerCelery "E:\notetaker\backend\notetaker\Scripts\python.exe" "E:\notetaker\backend\start_celery.py"

REM 4. Configura el directorio de trabajo
nssm set CosmosNotetakerCelery AppDirectory "E:\notetaker\backend"

REM 5. Configura inicio automático
nssm set CosmosNotetakerCelery Start SERVICE_AUTO_START

REM 6. Inicia el servicio
net start CosmosNotetakerCelery
```

## Gestión del Servicio

### Usando el Script de Gestión

Ejecuta `manage_celery_service.bat` para un menú interactivo que permite:
- Iniciar el servicio
- Detener el servicio
- Reiniciar el servicio
- Ver el estado
- Ver los logs

### Usando Comandos de Windows

```batch
REM Iniciar servicio
net start CosmosNotetakerCelery

REM Detener servicio
net stop CosmosNotetakerCelery

REM Ver estado
sc query CosmosNotetakerCelery

REM Ver logs (si están configurados)
type logs\celery_stdout.log
type logs\celery_stderr.log
```

### Usando la Interfaz de Windows

1. Abre **Servicios** (`services.msc`)
2. Busca **CosmosNotetakerCelery**
3. Haz clic derecho para iniciar/detener/configurar

## Desinstalación

Ejecuta `uninstall_celery_service.bat` como **Administrador** para desinstalar el servicio.

O manualmente:

```batch
net stop CosmosNotetakerCelery
nssm remove CosmosNotetakerCelery confirm
```

## Logs

Los logs se guardan en:
- `logs\celery_stdout.log` - Salida estándar
- `logs\celery_stderr.log` - Errores

## Configuración Avanzada

### Cambiar el puerto de Redis

Si Redis está en otro puerto, edita `.env`:

```env
REDIS_URL=redis://localhost:6380/0
```

### Verificar que el Servicio Funciona

1. Abre **Administrador de tareas** → Pestaña **Servicios**
2. Busca `CosmosNotetakerCelery`
3. Debe estar en estado **En ejecución**

### Troubleshooting

**El servicio no inicia:**
- Verifica que Redis esté corriendo
- Revisa los logs en `logs\celery_stderr.log`
- Verifica que el entorno virtual exista en `notetaker\Scripts\python.exe`

**El servicio se detiene automáticamente:**
- Revisa los logs para ver el error
- Verifica la configuración de Redis en `.env`
- Asegúrate de que todas las dependencias estén instaladas

**Los schedules no funcionan:**
- Verifica que el servicio esté corriendo: `sc query CosmosNotetakerCelery`
- Revisa los logs para ver si hay errores
- Asegúrate de que Redis esté accesible

## Ventajas de Usar Servicio

✅ **Inicio automático**: Se inicia con Windows  
✅ **Persistencia**: Sigue corriendo aunque cierres sesión  
✅ **Recuperación**: Se reinicia automáticamente si falla  
✅ **Logs**: Guarda logs automáticamente  
✅ **Gestión fácil**: Se gestiona desde Servicios de Windows  

## Notas

- El servicio necesita permisos de administrador para instalarse
- Una vez instalado, funciona con permisos del sistema
- Los logs se guardan automáticamente en la carpeta `logs`
- Si cambias el código, reinicia el servicio para aplicar cambios

