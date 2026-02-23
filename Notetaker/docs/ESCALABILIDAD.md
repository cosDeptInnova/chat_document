# Escalabilidad y Configuración de Producción

Este documento describe las configuraciones recomendadas para escalar Notetaker a ~200 usuarios concurrentes y optimizar el rendimiento del sistema.

## Arquitectura General

El sistema está compuesto por:
- **API FastAPI (Uvicorn)**: Procesa peticiones HTTP y delega trabajo pesado a Celery
- **Workers Celery**: Ejecutan tareas asíncronas (sync de calendarios, procesamiento de reuniones)
- **Redis**: Broker y backend de resultados para Celery
- **Base de Datos PostgreSQL**: Almacenamiento persistente

## Configuración de Uvicorn

### Workers Múltiples

Para entornos con carga, se recomienda usar múltiples workers de Uvicorn:

```bash
# Desarrollo (1 worker)
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Producción (4 workers recomendado para ~200 usuarios)
uvicorn app.main:app --workers 4 --host 0.0.0.0 --port 8000
```

**Nota**: Con el sync de calendario movido a Celery (Fase 2), el API es mucho más ligero y puede manejar más peticiones concurrentes.

### Variables de Entorno

Asegúrate de configurar:
- `DATABASE_URL`: URL de conexión a PostgreSQL
- `REDIS_URL`: URL de conexión a Redis (usado por Celery)
- `BACKEND_PUBLIC_URL`: URL pública del backend (para webhooks de calendarios)

## Timeouts

### Timeouts en Nginx/Proxy

Si usas Nginx como proxy reverso, configura timeouts adecuados:

```nginx
# nginx.conf
proxy_read_timeout 60s;      # Para peticiones largas como GET /api/meetings/list
proxy_connect_timeout 10s;
proxy_send_timeout 60s;
```

### Timeouts en Cliente (Frontend)

Configura timeouts en las peticiones HTTP del frontend:

```typescript
// Ejemplo con axios
const apiClient = axios.create({
  timeout: 30000,  // 30 segundos para peticiones normales
});

// Para peticiones específicas que pueden tardar más
apiClient.get('/api/meetings/list', { timeout: 60000 });  // 60 segundos
```

### Timeouts Recomendados

| Endpoint | Timeout Recomendado | Motivo |
|----------|-------------------|--------|
| `GET /api/meetings/list` | 30-60s | Puede procesar muchas reuniones |
| `GET /api/auth/me` | 10s | Debería ser rápido |
| `POST /api/integrations/webhook/*` | 5s | Webhooks deben responder rápido |
| `POST /api/calendar/sync` | 60s | Sync manual puede tardar |

## Pool de Base de Datos

### Configuración de SQLAlchemy

Asegúrate de que el pool de conexiones sea adecuado para workers uvicorn + workers Celery:

```python
# app/database.py
engine = create_engine(
    DATABASE_URL,
    pool_size=20,           # Conexiones en el pool
    max_overflow=10,        # Conexiones adicionales permitidas
    pool_timeout=30,        # Tiempo de espera para obtener conexión
    pool_recycle=3600,      # Renovar conexiones cada hora
)
```

**Cálculo aproximado**:
- 4 workers uvicorn × 5 conexiones = 20 conexiones
- 2 workers Celery × 5 conexiones = 10 conexiones
- Total recomendado: 30-40 conexiones (pool_size + max_overflow)

## Workers Celery

### Configuración de Celery

El sync de calendario por webhook Outlook se ejecuta en Celery (ver Fase 2 del plan de optimización). Configura workers adecuados:

```bash
# Worker general (para tareas de reuniones y sync de calendario)
celery -A app.celery_app worker --loglevel=info --concurrency=4

# Opcional: Worker dedicado para sync de calendario (si hay mucha carga)
celery -A app.celery_app worker --loglevel=info --concurrency=2 -Q calendar_sync
```

### Colas de Celery

Actualmente se usan las siguientes colas:
- `celery`: Cola por defecto (tareas de reuniones, sync de calendario)
- `summary_queue`: Cola para procesamiento de resúmenes

Si necesitas aislar el sync de calendario, puedes crear una cola dedicada:

```python
# app/celery_app.py
task_queues={
    "celery": {"exchange": "celery", "routing_key": "celery"},
    "calendar_sync": {"exchange": "calendar", "routing_key": "calendar"},
    "summary_queue": {"exchange": "summary", "routing_key": "summary"},
}
```

### Límites de Tiempo

Las tareas de Celery tienen límites configurados:
- `task_time_limit`: 300 segundos (5 minutos) - máximo absoluto
- `task_soft_time_limit`: 240 segundos (4 minutos) - límite suave

El sync de calendario normalmente tarda 25-40 segundos, por lo que estos límites son adecuados.

## Optimizaciones Implementadas

### Fase 1: Cooldown tras Sync Completado

Tras completar un sync para un usuario, no se programa otro sync hasta que pasen **90 segundos** (cooldown). Esto evita el bucle de sincronización cada ~13-26 segundos.

**Configuración**: `COOLDOWN_SECONDS = 90` en `app/api/routes/integrations.py`

### Fase 2: Sync en Celery

El sync de calendario por webhook Outlook se ejecuta en Celery, no en el proceso API. Esto evita bloquear uvicorn durante 25-40 segundos.

**Archivo**: `app/tasks/calendar_tasks.py`

### Fase 3: Evitar Trabajo Redundante

Se verifica explícitamente si la fecha realmente cambió antes de actualizar reuniones y reprogramar tareas Celery. Esto reduce logs repetitivos y trabajo innecesario.

### Fase 4: Webhooks Optimizados

Webhooks con body vacío o sin `subscriptionId` se responden inmediatamente (202) sin hacer queries a la base de datos.

## Monitoreo

### Logs

- **API**: `backend/logs/DD_MM_YYYY.log`
- **Celery**: `backend/logs/celery_DD_MM_YYYY.log` (ver Fase 6)

### Métricas Recomendadas

Monitorea:
- Tiempo de respuesta de endpoints críticos (`GET /api/meetings/list`, `GET /api/auth/me`)
- Número de tareas Celery en cola
- Uso de conexiones de base de datos
- Memoria y CPU de workers uvicorn y Celery

### Alertas por Email

El sistema envía automáticamente alertas por email a los administradores configurados en `ADMIN_EMAILS` cuando se detectan problemas críticos:

- **Timeouts frecuentes en endpoints**: Cuando un endpoint tiene múltiples timeouts
- **Cola de Celery con más de 100 tareas pendientes**: Indica que el sistema está sobrecargado
- **Pool de BD agotado**: Cuando se están usando casi todas las conexiones disponibles
- **Workers Celery caídos**: Cuando no hay workers activos procesando tareas

**Configuración**:
- Los emails se envían a las direcciones configuradas en `ADMIN_EMAILS` en el archivo `.env`
- Formato: `ADMIN_EMAILS=admin1@ejemplo.com,admin2@ejemplo.com`
- Las alertas tienen un cooldown de 30 minutos para evitar spam (misma alerta no se envía más de una vez cada 30 minutos)

**Uso programático**:
```python
from app.services.monitoring_service import (
    alert_timeout,
    alert_celery_queue,
    alert_db_pool_exhausted,
    alert_celery_worker_down,
    check_and_alert_system_health
)

# Ejemplo: alertar sobre timeout
alert_timeout("/api/meetings/list", timeout_seconds=30, request_count=5)

# Verificar salud del sistema (puede llamarse desde tarea Celery periódica)
check_and_alert_system_health()
```

## Escalado Horizontal

### API (Uvicorn)

Para escalar el API:
1. Aumentar número de workers uvicorn (hasta 2× número de CPUs)
2. Usar un load balancer (Nginx, HAProxy) delante de múltiples instancias
3. Asegurar que Redis y PostgreSQL sean accesibles desde todas las instancias

### Celery

Para escalar Celery:
1. Añadir más workers Celery en la misma o diferentes máquinas
2. Usar colas dedicadas para aislar carga (ej: `calendar_sync`, `summary_queue`)
3. Monitorear el uso de Redis (broker)

### Base de Datos

Para escalar la base de datos:
1. Aumentar `pool_size` y `max_overflow` en SQLAlchemy
2. Considerar read replicas para consultas de solo lectura
3. Optimizar índices en tablas frecuentemente consultadas

## Checklist de Producción

- [ ] Uvicorn configurado con múltiples workers (4+)
- [ ] Timeouts configurados en Nginx/proxy (60s para endpoints largos)
- [ ] Pool de BD configurado adecuadamente (30-40 conexiones)
- [ ] Workers Celery configurados y monitoreados
- [ ] Redis accesible y con memoria suficiente
- [ ] Logs configurados (API y Celery)
- [ ] Monitoreo de métricas críticas
- [ ] Alertas configuradas para problemas comunes
