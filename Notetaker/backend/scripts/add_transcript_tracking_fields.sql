-- Script SQL para añadir campos de tracking de transcripcion a la tabla meetings
-- Ejecutar directamente en PostgreSQL si Alembic no esta disponible

-- Verificar si los campos ya existen antes de añadirlos
DO $$
BEGIN
    -- Añadir transcript_task_id si no existe
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'meetings' 
        AND column_name = 'transcript_task_id'
    ) THEN
        ALTER TABLE meetings ADD COLUMN transcript_task_id VARCHAR;
        RAISE NOTICE 'Campo transcript_task_id añadido';
    ELSE
        RAISE NOTICE 'Campo transcript_task_id ya existe';
    END IF;

    -- Añadir transcript_scheduled_time si no existe
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'meetings' 
        AND column_name = 'transcript_scheduled_time'
    ) THEN
        ALTER TABLE meetings ADD COLUMN transcript_scheduled_time TIMESTAMP;
        RAISE NOTICE 'Campo transcript_scheduled_time añadido';
    ELSE
        RAISE NOTICE 'Campo transcript_scheduled_time ya existe';
    END IF;
END $$;

-- Crear indices si no existen
CREATE INDEX IF NOT EXISTS ix_meetings_transcript_task_id ON meetings(transcript_task_id);
CREATE INDEX IF NOT EXISTS ix_meetings_transcript_scheduled_time ON meetings(transcript_scheduled_time);

-- Verificar que los campos se añadieron correctamente
SELECT 
    column_name, 
    data_type, 
    is_nullable
FROM information_schema.columns 
WHERE table_name = 'meetings' 
AND column_name IN ('transcript_task_id', 'transcript_scheduled_time')
ORDER BY column_name;
