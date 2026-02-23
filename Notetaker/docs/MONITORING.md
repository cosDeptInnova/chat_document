# Monitoreo de Celery (Health Check)

## Contexto
El sistema cuenta con una tarea periódica que verifica la salud del sistema cada 10 minutos. Esta tarea comprueba:
1. Tamaño de las colas en Redis.
2. Estado de los workers de Celery.
3. Uso del pool de conexiones de la base de datos.

## Solución de Problemas (Troubleshooting)

### Alerta: Worker de Celery caído
Si recibes un correo de alerta `worker_down`, el sistema ha detectado que el worker no responde incluso tras varios reintentos con timeout extendido. 

#### 1. Verificar estado real
Ejecuta el script de prueba para ver el diagnóstico actual:
```powershell
cd backend
.\notetaker\Scripts\activate
python test_monitor.py
```

#### 2. Acciones de recuperación
- **Si el estado es `False`**: El worker está realmente caído o bloqueado.
  - **Reinicio manual**: Cierra la terminal del worker y ejecuta `.\start_celery.bat` en la carpeta `backend`.
  - **Servicio Windows**: Si está instalado como servicio, reinícialo desde `services.msc` (CosmosNotetakerCelery).
- **Si el estado es `uncertain`**: Es un falso positivo. El parche aplicado ahora ignora estos casos automáticamente para no molestarte con correos. Si ocurre, es simplemente que el worker está trabajando intensamente en otra tarea.

## Referencias
- [monitoring_service.py](file:///e:/notetaker2.0/backend/app/services/monitoring_service.py): Lógica de monitoreo.
- [test_monitor.py](file:///e:/notetaker2.0/backend/test_monitor.py): Script de verificación manual.
