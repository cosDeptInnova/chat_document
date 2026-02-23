# Aplicar Migración de Campos de Contraseña

## Problema
El usuario de la base de datos no tiene permisos suficientes para ejecutar migraciones de Alembic automáticamente.

## Solución

### Opción 1: Ejecutar Script SQL Manualmente (Recomendado)

1. Conéctate a PostgreSQL con un usuario administrador (postgres o superusuario):
   ```bash
   psql -U postgres -d cosmos_notetaker
   ```
   O usa tu herramienta de administración de PostgreSQL (pgAdmin, DBeaver, etc.)

2. Ejecuta el script SQL:
   ```sql
   -- Copia y pega el contenido de add_password_fields.sql
   ```

3. Después de ejecutar el script SQL, marca las migraciones como ejecutadas:
   ```bash
   cd E:\notetaker\backend
   .\notetaker\Scripts\Activate.ps1
   alembic stamp add_password_fields
   alembic stamp 9225cea883fe
   ```

### Opción 2: Otorgar Permisos al Usuario de la Aplicación

Si prefieres que Alembic pueda ejecutar las migraciones automáticamente, otorga permisos al usuario:

```sql
-- Conectado como postgres o superusuario
GRANT ALL PRIVILEGES ON TABLE users TO tu_usuario_aplicacion;
GRANT ALL PRIVILEGES ON TABLE transcriptions TO tu_usuario_aplicacion;
ALTER TABLE users OWNER TO tu_usuario_aplicacion;
ALTER TABLE transcriptions OWNER TO tu_usuario_aplicacion;
```

Luego ejecuta:
```bash
alembic upgrade head
```

### Verificación

Después de aplicar los cambios, verifica que las columnas existen:

```sql
SELECT column_name, data_type, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'users' 
AND column_name IN ('hashed_password', 'password_reset_token', 'password_reset_expires', 'must_change_password');
```

Deberías ver las 4 columnas listadas.

