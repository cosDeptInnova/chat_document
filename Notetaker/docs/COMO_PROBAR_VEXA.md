# Cómo probar la integración con VEXA

## 1. Configurar VEXA_API_KEY

Antes de probar, necesitas configurar `VEXA_API_KEY` en tu `.env`:

### Opción A: Crear usuario y token (recomendado para self-hosted)

1. **Conecta a la Admin API de VEXA** (puerto 8057):
   ```bash
   curl -X POST http://172.29.14.10:8057/admin/users \
     -H "Content-Type: application/json" \
     -H "X-Admin-API-Key: e6erL2YR^V$D$Xu@D7&m8xy5pmWArtC6$uqfYj3rMpeV6s2&3PM28hyN$KmyhELN4rJ7fCbj9neL85@pA5j2%DCBRcbxwAHUxbihmmT6NRjmgM$Zw3V$deQ$uhq4TKmH" \
     -d '{
       "email": "notetaker@tudominio.com",
       "name": "Notetaker Bot",
       "max_concurrent_bots": 10
     }'
   ```

2. **Obtén el user_id** de la respuesta (ej: `{"id": 1, ...}`)

3. **Genera un token para ese usuario**:
   ```bash
   curl -X POST http://172.29.14.10:8057/admin/users/1/tokens \
     -H "X-Admin-API-Key: e6erL2YR^V$D$Xu@D7&m8xy5pmWArtC6$uqfYj3rMpeV6s2&3PM28hyN$KmyhELN4rJ7fCbj9neL85@pA5j2%DCBRcbxwAHUxbihmmT6NRjmgM$Zw3V$deQ$uhq4TKmH"
   ```

4. **Copia el token** de la respuesta (ej: `{"token": "AbCdEf123...", ...}`)

5. **Añádelo a `.env`**:
   ```env
   VEXA_API_KEY=AbCdEf123...
   ```

### Opción B: Usar ADMIN_API_TOKEN directamente (si tu instalación lo permite)

Algunas instalaciones self-hosted permiten usar el mismo token de admin para la User API. Prueba:

```env
VEXA_API_KEY=e6erL2YR^V$D$Xu@D7&m8xy5pmWArtC6$uqfYj3rMpeV6s2&3PM28hyN$KmyhELN4rJ7fCbj9neL85@pA5j2%DCBRcbxwAHUxbihmmT6NRjmgM$Zw3V$deQ$uhq4TKmH
```

---

## 2. Ejecutar script de prueba

```bash
cd E:\notetaker2.0\backend
python test_vexa.py
```

El script prueba:
- ✅ Conexión con VEXA (`GET /bots/status`)
- ✅ Parser de URLs de Teams
- ✅ Crear bot en reunión (opcional, requiere datos reales)
- ✅ Obtener transcripción (opcional, requiere bot activo)

---

## 3. Probar manualmente con curl

### Verificar conexión

```bash
curl -X GET "http://172.29.14.10:8056/bots/status" \
  -H "X-API-Key: TU_VEXA_API_KEY"
```

### Crear bot en reunión Teams

**Obtén `native_meeting_id` y `passcode` desde la URL de Teams:**

- URL ejemplo: `https://teams.live.com/meet/9366473044740?p=waw4q9dPAvdIG3aknh`
- `native_meeting_id` = `9366473044740` (solo los dígitos)
- `passcode` = `waw4q9dPAvdIG3aknh` (del parámetro `?p=`)

```bash
curl -X POST "http://172.29.14.10:8056/bots" \
  -H "X-API-Key: TU_VEXA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "platform": "teams",
    "native_meeting_id": "9366473044740",
    "passcode": "waw4q9dPAvdIG3aknh",
    "bot_name": "Cosmos Notetaker Bot",
    "language": "es"
  }'
```

**Respuesta esperada:**
```json
{
  "id": 123,
  "status": "active",
  ...
}
```

### Obtener transcripción

```bash
curl -X GET "http://172.29.14.10:8056/transcripts/teams/9366473044740" \
  -H "X-API-Key: TU_VEXA_API_KEY"
```

**Respuesta esperada:**
```json
{
  "segments": [
    {
      "text": "Hola a todos",
      "speaker": "Juan",
      "absolute_start_time": "2025-01-30T10:00:00Z",
      "absolute_end_time": "2025-01-30T10:00:03Z"
    }
  ]
}
```

---

## 4. Probar desde la aplicación (notetaker2.0)

### Crear reunión y unir bot automáticamente

1. **Crea una reunión** desde el frontend o API con una URL de Teams válida
2. **El bot se unirá automáticamente** 1 minuto antes del inicio programado (tarea Celery `join_bot_to_meeting`)
3. **La transcripción se obtendrá** automáticamente después de la reunión (tarea `fetch_vexa_transcript_for_meeting`)

### Unir bot manualmente (API)

```bash
curl -X POST "http://localhost:7000/api/vexa/bot/join-meeting" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer TU_JWT_TOKEN" \
  -d '{
    "meeting_id": "uuid-de-la-reunion"
  }'
```

### Obtener transcripción manualmente (API)

```bash
curl -X GET "http://localhost:7000/api/vexa/meetings/{meeting_id}/transcript" \
  -H "Authorization: Bearer TU_JWT_TOKEN"
```

---

## 5. Verificar logs

Revisa los logs de Celery y del backend para ver el flujo:

```bash
# Logs de Celery
tail -f E:\notetaker2.0\backend\logs\celery_*.log

# Logs del backend (si usas uvicorn)
# Verás mensajes como:
# [CELERY] Iniciando join_bot_to_meeting (VEXA) para reunion ...
# VEXA bot started for teams meeting native_id=...
```

---

## 6. Troubleshooting

### Error: "VEXA_API_KEY no está configurada"
- Verifica que `.env` tenga `VEXA_API_KEY=...`
- Reinicia el backend después de cambiar `.env`

### Error: "VEXA API error 401"
- El token no es válido o no tiene permisos
- Crea un nuevo token desde Admin API

### Error: "Could not parse Teams meeting ID from URL"
- La URL de Teams no tiene el formato esperado
- Guarda `vexa_native_meeting_id` y `vexa_passcode` manualmente en `meeting.extra_metadata`

### Bot no se une a la reunión
- Verifica que el `native_meeting_id` sea correcto (solo dígitos, 10-15 caracteres)
- Verifica que el `passcode` sea correcto si la reunión lo requiere
- Revisa los logs de VEXA en el servidor (172.29.14.10)

### No hay transcripción después de la reunión
- Verifica que el bot estuviera activo durante la reunión
- Llama manualmente a `GET /api/vexa/meetings/{meeting_id}/transcript`
- Verifica que la reunión tenga `recall_bot_id` (native_meeting_id) guardado

---

## 7. Documentación adicional

- **API VEXA completa:** `docs/VEXA_API.md`
- **Documentación oficial VEXA:** archivos en `e:\Descargas\` (deployment.md, user_api_guide.md, websocket.md)
