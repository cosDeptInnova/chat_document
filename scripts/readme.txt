COSMOS – Scripts de despliegue/arranque (PowerShell)
===================================================

1. Objetivo
-----------
Esta carpeta contiene scripts PowerShell para:
- Arrancar/parar todos los microservicios (start-all / stop-all)
- Arrancar/parar servicios individuales (start-service / stop-service)
- Consultar el estado de ejecución (status)
- Exportar variables de entorno de una sesión PowerShell a ficheros config\<servicio>.env
- Gestionar PID files y logs de salida/error
- Liberar puertos ocupados de forma robusta (incluye parada de servicios Windows y kill tree)

2. Requisitos previos
---------------------
Sistema: Windows (PowerShell) con permisos suficientes.
Requisitos recomendados:
- Anaconda/Miniconda instalado
- Entorno conda con dependencias del proyecto (ver services.psd1 -> CondaEnv)
- Docker Desktop (si se despliegan redis/postgresql/qdrant/clamav por docker-compose)
- Puertos libres según configuración (services.psd1)

NOTA: Algunos cierres de procesos/servicios requieren ejecutar PowerShell "Como Administrador".

3. Estructura creada automáticamente
------------------------------------
Al ejecutar los scripts, se crearán (si no existen) en el repo:
- run\        -> PID files + launchers generados
- logs\       -> logs *.out.log y *.err.log por servicio
- config\     -> ficheros .env consolidados
- scripts\    -> (esta carpeta)
- EVIDENCIAS_HITO2\ (carpeta de evidencias)

4. Configuración (services.psd1)
--------------------------------
Los scripts leen scripts\services.psd1 (PowerShell data file), donde se define:
- CondaEnv (nombre del entorno conda)
- GlobalEnvFile (opcional: .env global)
- FailFast (opcional)
- StartOrder (orden recomendado de arranque)
- Services: listado con Name, Path, Port, Args, etc.

IMPORTANTE: Mantener actualizado services.psd1 cuando se añadan o modifiquen microservicios.

5. Arranque/Parada/Estado
-------------------------
Abrir PowerShell en la carpeta /scripts y ejecutar:

Arrancar todo:
  .\start-all.ps1

Parar todo:
  .\stop-all.ps1

Ver estado:
  .\status.ps1

Arrancar un servicio:
  .\start-service.ps1 -Name <NombreServicio>

Parar un servicio:
  .\stop-service.ps1 -Name <NombreServicio>

Los logs se almacenan en:
  ..\logs\<servicio>.out.log
  ..\logs\<servicio>.err.log

6. Gestión robusta de puertos
-----------------------------
Antes de arrancar un servicio con "Port" definido en services.psd1, el arranque:
- Detecta procesos escuchando en el puerto (Get-NetTCPConnection y fallback netstat)
- Opcionalmente para servicios Windows asociados a ese PID
- Mata el árbol de procesos (Stop-Process y fallback taskkill /T /F)
- Reintenta hasta liberar el puerto

Si el puerto no se libera, se muestra diagnóstico (PID, nombre proceso, cmdline y servicios).

7. Exportación de variables de entorno a config\<servicio>.env
--------------------------------------------------------------
Script: export-session-env-to-config.ps1

Uso típico (desde la carpeta del microservicio, p.ej. /auth, /chatdoc, etc.):
  ..\scripts\export-session-env-to-config.ps1

Opciones:
-RepoRoot <ruta>         -> si no puede autodetectar el repo (busca docker-compose.yml hacia arriba)
-Merge:$true|$false      -> fusiona con config\<svc>.env existente (por defecto true)
-ImportLocalDotEnv       -> importa .env local si existe (por defecto true)
-Show                    -> muestra resultado por consola (enmascara secretos)
-ShowSecrets             -> muestra secretos sin enmascarar (NO recomendado)
-Backup:$true|$false     -> crea backup del fichero antes de sobrescribir (por defecto true)

NOTA: Este script extrae claves de entorno del historial (Get-History) buscando asignaciones tipo:
  $env:VAR = ...
Por tanto, se debe ejecutar en la misma consola donde se exportaron esas variables.

8. Consideraciones de seguridad
-------------------------------
- Los ficheros config\<svc>.env pueden contener secretos (tokens/keys/passwords). Protegerlos.
- Evitar usar -ShowSecrets en entornos compartidos.
- Ejecutar como admin solo cuando sea necesario (liberar puertos/servicios).
- Revisar logs ante fallos de arranque o timeouts.

9. Troubleshooting rápido
-------------------------
- "Cannot find 'conda'": ejecutar 'conda init powershell' y reiniciar la consola, o asegurar conda en PATH.
- "port XXXX still busy": ejecutar PowerShell como administrador y relanzar stop-all / start-all.
- "timeout port did not open": revisar logs/<svc>.err.log y comprobar args/paths/env.