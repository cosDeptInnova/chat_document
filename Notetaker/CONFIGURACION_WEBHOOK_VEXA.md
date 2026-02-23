# Configuración del Webhook Vexa → Notetaker

## Resumen

Se ha implementado un webhook que notifica al backend de Notetaker cuando el bot sale de una reunión. Esto permite actualizar el estado de la reunión y obtener la transcripción **inmediatamente**, sin esperar a que un usuario haga clic en la reunión.

---

## Configuración Requerida

### 1. Backend de Notetaker

**No requiere configuración adicional** - el endpoint ya está creado y funcionando.

El endpoint está disponible en:
```
POST /api/internal/vexa/bot-exited
```

**Autenticación:** Header `Authorization: Bearer <API_KEY>`

**Payload esperado:**
```json
{
  "recall_bot_id": "9366473044740",
  "exit_code": 0,
  "status": "completed",
  "platform": "teams",
  "reason": "self_initiated_leave"
}
```

---

### 2. Bot-Manager (Vexa)

**Agregar variables de entorno en `docker-compose.yml` o `.env`:**

```yaml
bot-manager:
  environment:
    # ... otras variables ...
    - NOTETAKER_BACKEND_URL=http://notetaker-backend:7000  # URL del backend de Notetaker
    # Usa ADMIN_TOKEN (ya configurado) o define NOTETAKER_WEBHOOK_API_KEY
    - NOTETAKER_WEBHOOK_API_KEY=${VEXA_API_KEY}  # Opcional: API key específica para webhook
```

**Variables de entorno:**
- `NOTETAKER_BACKEND_URL`: URL base del backend de Notetaker (ej: `http://172.29.14.10:7000` o `http://notetaker-backend:7000` si están en la misma red Docker)
- `NOTETAKER_WEBHOOK_API_KEY` (opcional): API key para autenticación. Si no se define, usa `ADMIN_TOKEN`

**Nota:** El backend de Notetaker usa `vexa_api_key` de su configuración para verificar el webhook. Asegúrate de que `NOTETAKER_WEBHOOK_API_KEY` (o `ADMIN_TOKEN`) coincida con `VEXA_API_KEY` en el `.env` de Notetaker.

---

## Flujo Completo

### Cuando el bot sale de la reunión:

1. **Bot** → Llama a `/bots/internal/callback/exited` en **bot-manager**
2. **Bot-manager** → Procesa el callback y actualiza su BD
3. **Bot-manager** → Llama a `/api/internal/vexa/bot-exited` en **Notetaker** (nuevo webhook)
4. **Notetaker** → Busca reunión por `recall_bot_id`
5. **Notetaker** → Actualiza estado a `COMPLETED` o `FAILED`
6. **Notetaker** → Intenta obtener transcripción de VEXA
7. **Notetaker** → Si hay segmentos, los guarda en BD

### Cuando un usuario hace clic en la reunión:

1. **Usuario** → `GET /api/meetings/{meeting_id}`
2. **Notetaker** → Verifica si hay transcripción en BD
3. **Si hay transcripción** → La devuelve desde BD (no vuelve a pedir a VEXA)
4. **Si no hay transcripción** → Verifica estado en VEXA y obtiene transcripción si está disponible

---

## Ventajas del Webhook

✅ **Actualización inmediata:** El estado se actualiza tan pronto como el bot sale  
✅ **Transcripción automática:** Se obtiene y guarda automáticamente  
✅ **No depende de usuarios:** No necesita que alguien haga clic para actualizar  
✅ **Fallback robusto:** Si el webhook falla, `get_meeting()` sigue funcionando como antes  

---

## Troubleshooting

### El webhook no se llama

1. **Verificar configuración:**
   ```bash
   # En el contenedor de bot-manager
   docker exec <bot-manager-container> env | grep NOTETAKER
   ```

2. **Verificar logs de bot-manager:**
   ```bash
   docker compose logs bot-manager | grep -i "notetaker\|webhook"
   ```

3. **Verificar conectividad:**
   ```bash
   # Desde bot-manager hacia Notetaker
   docker exec <bot-manager-container> curl -v http://notetaker-backend:7000/api/health
   ```

### El webhook falla con 401 (No autorizado)

- Verificar que `NOTETAKER_WEBHOOK_API_KEY` (o `ADMIN_TOKEN`) coincida con `VEXA_API_KEY` en Notetaker
- Verificar que el header `Authorization: Bearer <API_KEY>` se envía correctamente

### El webhook falla con 404

- Verificar que la URL `NOTETAKER_BACKEND_URL` sea correcta
- Verificar que el endpoint `/api/internal/vexa/bot-exited` esté registrado en `main.py`

### La reunión no se encuentra en Notetaker

- Verificar que `recall_bot_id` en Notetaker coincida con `platform_specific_id` en bot-manager
- Verificar logs: `docker compose logs backend | grep -i "recall_bot_id"`

---

## Pruebas

### Probar el webhook manualmente:

```bash
# Desde bot-manager (o cualquier máquina con acceso)
curl -X POST http://notetaker-backend:7000/api/internal/vexa/bot-exited \
  -H "Authorization: Bearer <VEXA_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "recall_bot_id": "9366473044740",
    "exit_code": 0,
    "status": "completed",
    "platform": "teams",
    "reason": "test"
  }'
```

### Verificar que funciona:

1. Crear una reunión y unir el bot
2. Esperar a que el bot salga (o expulsarlo manualmente)
3. Verificar logs de bot-manager: debería aparecer "Webhook a Notetaker exitoso"
4. Verificar logs de Notetaker: debería aparecer "Webhook bot-exited recibido"
5. Consultar la reunión: debería estar en estado `COMPLETED` con transcripción

---

## Notas Importantes

- El webhook es **asíncrono** (se ejecuta en background) - no bloquea el callback del bot
- Si el webhook falla, **no afecta** el procesamiento normal del callback en bot-manager
- El endpoint en Notetaker es **idempotente** - se puede llamar múltiples veces sin problemas
- Si la transcripción no está disponible inmediatamente, se intentará de nuevo cuando el usuario consulte la reunión
