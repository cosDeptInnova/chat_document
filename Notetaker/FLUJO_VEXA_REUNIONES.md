# Flujo de Vexa: Reuniones y Transcripciones

## Resumen del Flujo Actual

### 1. Cuando el bot entra a una reunión

1. **Bot se une a la reunión** → Estado local: `JOINING` → `IN_PROGRESS`
2. **Bot-manager** crea el contenedor del bot con `botManagerCallbackUrl` configurado:
   ```
   botManagerCallbackUrl: "http://bot-manager:8080/bots/internal/callback/exited"
   ```
3. El bot **NO notifica directamente** al backend de Notetaker cuando entra/sale

### 2. Cuando el bot sale de la reunión

**SÍ existe webhook/callback:**
- El bot llama a `/bots/internal/callback/exited` en bot-manager cuando sale
- Bot-manager actualiza el estado de la reunión en su BD a `completed` o `failed`
- **PERO:** El backend de Notetaker **NO recibe notificación directa** de este callback

### 3. Cuando un usuario hace clic en una reunión (GET /api/meetings/{meeting_id})

**Flujo actual en `get_meeting()`:**

1. **Verifica si hay transcripción en BD local:**
   ```python
   transcription = db.query(Transcription).filter(Transcription.meeting_id == meeting_id).first()
   has_segments_in_db = (segmentos en BD > 0)
   ```

2. **Decide si verificar VEXA** (`should_check_vexa`):
   - ✅ **SÍ verifica** si:
     - Estado `IN_PROGRESS` + han pasado ≥2 min desde inicio
     - Estado `COMPLETED` pero sin segmentos en BD
     - Estado `JOINING` sin segmentos
   - ❌ **NO verifica** si:
     - Estado `IN_PROGRESS` pero <2 min desde inicio
     - Ya hay segmentos en BD

3. **Si debe verificar VEXA:**
   ```python
   # 1. Verificar estado en VEXA
   vexa_status = vexa.get_meeting_status(
       native_meeting_id=meeting.recall_bot_id,
       platform="teams"
   )
   
   # 2. Si está completed/failed en VEXA pero localmente IN_PROGRESS:
   if vexa_meeting_status in ("completed", "failed"):
       # 3. Intentar obtener transcripción
       data = vexa.get_transcript(...)
       
       # 4. Si hay segmentos, guardarlos en BD
       if segments:
           save_vexa_transcript_to_db(db, meeting, data)
           db.commit()
   ```

4. **Respuesta al usuario:**
   - Si hay segmentos en BD → los devuelve
   - Si no hay segmentos → devuelve estado `IN_PROGRESS` o `COMPLETED` según corresponda

---

## Confirmación del Flujo Actual

### ✅ Lo que SÍ tienes implementado:

1. **Webhook cuando el bot sale:**
   - ✅ Bot llama a `/bots/internal/callback/exited` en bot-manager
   - ✅ Bot-manager actualiza estado en su BD
   - ❌ **NO notifica al backend de Notetaker** (solo actualiza BD interna de bot-manager)

2. **Verificación del estado del bot:**
   - ✅ Cuando usuario hace clic → `get_meeting()` verifica VEXA si es necesario
   - ✅ Usa `vexa.get_meeting_status()` para saber si el bot sigue activo
   - ✅ Si el bot ya no está → obtiene transcripción y la guarda

3. **Obtención de transcripciones:**
   - ✅ Si hay segmentos en BD → siempre los devuelve de BD (no vuelve a pedir a VEXA)
   - ✅ Si no hay segmentos → intenta obtener de VEXA y guardar
   - ✅ Una vez guardada en BD, siempre se muestra desde BD

4. **Estados de reunión:**
   - ✅ `IN_PROGRESS` → bot está dentro
   - ✅ `COMPLETED` → bot salió, reunión terminada
   - ✅ `FAILED` → bot falló

---

## Problemas Identificados

### 1. **No hay notificación en tiempo real cuando el bot sale**

**Problema:**
- El bot llama a `/bots/internal/callback/exited` en bot-manager
- Bot-manager actualiza su BD pero **NO notifica al backend de Notetaker**
- El backend de Notetaker solo se entera cuando un usuario hace clic en la reunión

**Impacto:**
- Si nadie hace clic, el estado puede quedar `IN_PROGRESS` indefinidamente
- No hay actualización automática del estado

### 2. **Verificación solo cuando usuario hace clic**

**Problema:**
- La verificación con VEXA solo ocurre cuando alguien llama a `GET /api/meetings/{meeting_id}`
- Si nadie consulta la reunión, nunca se actualiza el estado

**Impacto:**
- Reuniones pueden quedar "colgadas" en `IN_PROGRESS`
- No hay sincronización automática

### 3. **No hay progreso estimado de transcripción**

**Problema:**
- Si VEXA devuelve `completed` pero sin segmentos, solo se marca como `COMPLETED`
- No se muestra progreso estimado (ej. "Transcripción en proceso: 60%")

**Impacto:**
- Usuario no sabe si la transcripción está procesándose o si falló

---

## Propuestas de Mejora

### Opción 1: Webhook desde Bot-Manager al Backend de Notetaker (RECOMENDADA)

**Implementación:**
1. Bot-manager llama a un endpoint del backend de Notetaker cuando el bot sale:
   ```
   POST http://notetaker-backend/api/internal/vexa/bot-exited
   Body: {
     "meeting_id": "...",
     "recall_bot_id": "...",
     "status": "completed" | "failed",
     "exit_code": 0 | 1
   }
   ```

2. Backend de Notetaker:
   - Actualiza estado de reunión inmediatamente
   - Intenta obtener transcripción de VEXA
   - Guarda transcripción si está disponible
   - Notifica al frontend (WebSocket/Polling) si hay usuarios viendo la reunión

**Ventajas:**
- ✅ Actualización en tiempo real
- ✅ No depende de que un usuario haga clic
- ✅ Más eficiente (solo se verifica cuando realmente termina)

**Desventajas:**
- ⚠️ Requiere configurar URL del backend en bot-manager
- ⚠️ Requiere autenticación entre servicios

---

### Opción 2: Polling periódico desde Backend de Notetaker

**Implementación:**
1. Tarea Celery/Background que corre cada 1-2 minutos
2. Busca reuniones en estado `IN_PROGRESS` con `recall_bot_id`
3. Verifica estado en VEXA para cada una
4. Si están `completed` → obtiene transcripción y actualiza estado

**Ventajas:**
- ✅ No requiere cambios en bot-manager
- ✅ Funciona sin webhooks

**Desventajas:**
- ⚠️ Menos eficiente (verifica todas las reuniones aunque no hayan terminado)
- ⚠️ Puede haber delay de 1-2 minutos

---

### Opción 3: Verificación mejorada en `get_meeting()` (ACTUAL + MEJORAS)

**Mejoras al flujo actual:**
1. **Siempre verificar estado del bot si está `IN_PROGRESS`:**
   ```python
   if meeting.status == MeetingStatus.IN_PROGRESS:
       # Verificar SIEMPRE, no solo si han pasado 2 min
       should_check_vexa = True
   ```

2. **Mostrar progreso estimado:**
   ```python
   if vexa_meeting_status == "completed" and not segments:
       # Devolver estado con progreso estimado
       return {
           "status": "processing_transcription",
           "progress_estimate": "60%",  # Basado en tiempo transcurrido
           "message": "Transcripción en proceso..."
       }
   ```

3. **Cache de verificación:**
   - Guardar timestamp de última verificación
   - No verificar VEXA si se verificó hace <30 segundos

**Ventajas:**
- ✅ Mejora el flujo actual sin cambios grandes
- ✅ Más responsive cuando usuarios consultan

**Desventajas:**
- ⚠️ Sigue dependiendo de que alguien haga clic
- ⚠️ No actualiza automáticamente

---

## Recomendación Final

**Implementar Opción 1 (Webhook) + Opción 3 (Mejoras en get_meeting) como fallback:**

1. **Webhook desde bot-manager** para actualización inmediata
2. **Mejoras en `get_meeting()`** como fallback si el webhook falla
3. **Polling periódico** solo para reuniones "huérfanas" (sin `recall_bot_id` o webhook fallido)

---

## Checklist de Implementación

### Fase 1: Webhook Bot-Manager → Notetaker Backend
- [ ] Crear endpoint en Notetaker: `POST /api/internal/vexa/bot-exited`
- [ ] Configurar autenticación (API key compartida o JWT)
- [ ] Modificar bot-manager para llamar a este endpoint cuando el bot sale
- [ ] Probar flujo completo

### Fase 2: Mejoras en `get_meeting()`
- [ ] Verificar siempre si está `IN_PROGRESS` (no solo después de 2 min)
- [ ] Agregar cache de verificación (no verificar si se hizo hace <30s)
- [ ] Agregar progreso estimado de transcripción

### Fase 3: Polling de seguridad (opcional)
- [ ] Tarea Celery que verifica reuniones `IN_PROGRESS` cada 5 minutos
- [ ] Solo para reuniones que llevan >10 minutos en `IN_PROGRESS`
- [ ] Log de reuniones "huérfanas" para debugging

---

## Preguntas para Confirmar

1. **¿El bot-manager puede hacer llamadas HTTP al backend de Notetaker?**
   - ¿Tiene acceso de red?
   - ¿Qué URL usaría?

2. **¿Hay autenticación entre servicios?**
   - ¿API key compartida?
   - ¿JWT?
   - ¿IP whitelist?

3. **¿Quieres progreso estimado de transcripción?**
   - ¿O solo mostrar "En proceso" / "Completada"?

4. **¿Prefieres polling periódico o solo webhook?**
   - Polling = más carga pero más seguro
   - Solo webhook = más eficiente pero depende de que funcione
