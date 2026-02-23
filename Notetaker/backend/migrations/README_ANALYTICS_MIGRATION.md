# Migración: Tabla user_analytics_monthly

## 📋 Instrucciones para producción

**IMPORTANTE:** Ejecuta este script SQL en la base de datos de producción ANTES de desplegar el nuevo código.

### Pasos:

1. **Conectarse a la base de datos PostgreSQL de producción**

2. **Ejecutar el script SQL:**
   ```bash
   psql -U tu_usuario -d nombre_base_datos -f migrations/add_user_analytics_monthly.sql
   ```
   
   O directamente en psql:
   ```sql
   \i migrations/add_user_analytics_monthly.sql
   ```

3. **Verificar que la tabla se creó correctamente:**
   ```sql
   \d user_analytics_monthly
   ```
   
   Deberías ver la estructura de la tabla con todos los campos.

4. **Verificar índices:**
   ```sql
   SELECT indexname FROM pg_indexes WHERE tablename = 'user_analytics_monthly';
   ```
   
   Deberías ver 3 índices creados.

### Estructura de la tabla:

- `id`: UUID único
- `user_id`: FK a users.id
- `year`, `month`: Año y mes (1-12)
- Métricas agregadas: meetings_count, total_hours, participation, etc.
- `created_at`, `updated_at`: Timestamps

### Notas:

- La tabla tiene un constraint UNIQUE en (user_id, year, month) para evitar duplicados
- Los índices mejoran el rendimiento de consultas por usuario y por período
- La tabla se llenará automáticamente cuando el nuevo código se ejecute

### Post-migración:

Después de ejecutar el script y desplegar el código, las métricas se guardarán automáticamente cada mes. Los datos históricos del mes actual se pueden generar retroactivamente si es necesario.
