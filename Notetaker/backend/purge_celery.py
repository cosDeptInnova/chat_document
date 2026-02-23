"""Script para purgar todas las tareas programadas en Celery."""
import sys
from pathlib import Path

# Asegurar que estamos en el directorio correcto
backend_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(backend_dir))

try:
    from app.celery_app import celery_app
    from kombu import Queue
    
    print("Purgando todas las tareas de Celery...")
    
    # Purgar todas las tareas de la cola usando la conexión directa
    with celery_app.connection() as conn:
        # Obtener el nombre de la cola por defecto
        default_queue = celery_app.conf.task_default_queue or 'celery'
        queue = Queue(default_queue, channel=conn.channel())
        total_purged = queue.purge()
    
    print(f"Tareas purgadas: {total_purged}")
    print("Cola de Celery limpiada completamente")
    
except Exception as e:
    print(f"Error purgando tareas de Celery: {e}")
    import traceback
    traceback.print_exc()
    print("\nAsegurate de que:")
    print("   1. Redis esta corriendo")
    print("   2. Estas usando el entorno virtual correcto")
    print("\nAlternativa: Usa el comando directo de Celery:")
    print("   celery -A app.celery_app purge")
    sys.exit(1)

