# Instrucciones para Aplicar Certificados SSL

## Resumen de Mejoras Implementadas

### 1. Headers de Seguridad HTTP
Se ha implementado un middleware que añade automáticamente los siguientes headers de seguridad a todas las respuestas:

- **Strict-Transport-Security (HSTS)**: Fuerza a los navegadores a usar HTTPS durante 1 año
- **Content-Security-Policy (CSP)**: Política de seguridad de contenido
- **X-Content-Type-Options**: Previene MIME type sniffing
- **X-Frame-Options**: Previene clickjacking
- **X-XSS-Protection**: Protección contra XSS
- **Referrer-Policy**: Controla información de referrer
- **Permissions-Policy**: Desactiva características no necesarias

### 2. Gestión de Certificados SSL
Se ha implementado una interfaz completa en la sección de administración para:

- Ver resumen del certificado SSL actual
- Importar nuevos certificados SSL
- Gestionar certificados intermedios (manual o automático)
- Ver la cadena completa de certificados

## Configuración de Nginx para SSL (Docker en Windows Server)

### Archivo de Configuración Modificado

Se ha creado un archivo `nginx.conf.modificado` con las siguientes modificaciones:

1. **Bloque HTTP (Puerto 80)**: Redirige todo el tráfico a HTTPS
2. **Bloque HTTPS (Puerto 443)**: Configuración SSL completa con certificados

### Pasos para Aplicar la Configuración en Docker

#### Paso 1: Identificar el Contenedor de Nginx

Primero, identifica el nombre o ID del contenedor de nginx:

```powershell
# Listar contenedores Docker
docker ps

# Buscar contenedor de nginx
docker ps | Select-String "nginx"
```

Anota el nombre o ID del contenedor (ejemplo: `nginx-container` o `abc123def456`).

#### Paso 2: Preparar Directorio para Certificados SSL en el Host

En Windows Server, crea un directorio para montar los certificados SSL:

```powershell
# Crear directorio para certificados SSL (ajusta la ruta según tu estructura)
New-Item -ItemType Directory -Force -Path "C:\nginx\ssl"

# O si prefieres en el mismo directorio del proyecto
New-Item -ItemType Directory -Force -Path "E:\nginx\ssl"
```

#### Paso 3: Copiar Certificados desde el Backend

Después de importar un certificado SSL desde la interfaz de administración, los archivos se guardan en:
- `E:\notetaker2.0\backend\ssl_certs\certificate.crt` - Certificado principal
- `E:\notetaker2.0\backend\ssl_certs\private.key` - Clave privada
- `E:\notetaker2.0\backend\ssl_certs\chain.crt` - Certificados intermedios (si existen)
- `E:\notetaker2.0\backend\ssl_certs\fullchain.crt` - Certificado completo (certificado + cadena)

**Copiar certificados al directorio de nginx:**

```powershell
# Copiar certificados al directorio de nginx
Copy-Item "E:\notetaker2.0\backend\ssl_certs\fullchain.crt" -Destination "C:\nginx\ssl\notetaker.crt"
Copy-Item "E:\notetaker2.0\backend\ssl_certs\private.key" -Destination "C:\nginx\ssl\notetaker.key"
```

#### Paso 4: Aplicar Configuración de Nginx en Docker

**Opción A: Reemplazar archivo completo (Recomendado si no tienes otras configuraciones)**

```powershell
# 1. Hacer backup de la configuración actual del contenedor
docker cp nginx-container:/etc/nginx/conf/nginx.conf C:\nginx\nginx.conf.backup

# 2. Copiar nueva configuración al contenedor
docker cp nginx.conf.modificado nginx-container:/etc/nginx/conf/nginx.conf

# 3. Verificar sintaxis dentro del contenedor
docker exec nginx-container nginx -t

# 4. Si está OK, recargar nginx
docker exec nginx-container nginx -s reload
```

**Opción B: Modificar manualmente (Recomendado si tienes otras configuraciones)**

1. **Copiar configuración actual del contenedor al host:**

```powershell
docker cp nginx-container:/etc/nginx/conf/nginx.conf C:\nginx\nginx.conf
```

2. **Editar el archivo** `C:\nginx\nginx.conf` con un editor de texto y realizar estos cambios:

1. **Modificar el bloque `server` del puerto 80** para redirigir a HTTPS:

```nginx
# Reemplazar todo el bloque server del puerto 80 con:
server {
    listen 80;
    listen [::]:80;
    server_name notetaker.cosgs.com;

    # Redireccionar todo el trafico HTTP a HTTPS
    return 301 https://$server_name$request_uri;
}
```

2. **Añadir nuevo bloque `server` para HTTPS** después del bloque HTTP:

```nginx
# Configuracion HTTPS con certificados SSL
server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name notetaker.cosgs.com;

    # Rutas de certificados SSL
    ssl_certificate /etc/nginx/ssl/notetaker.crt;
    ssl_certificate_key /etc/nginx/ssl/notetaker.key;

    # Configuracion SSL recomendada
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers 'ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256:ECDHE-ECDSA-AES256-GCM-SHA384:ECDHE-RSA-AES256-GCM-SHA384:ECDHE-ECDSA-CHACHA20-POLY1305:ECDHE-RSA-CHACHA20-POLY1305:DHE-RSA-AES128-GCM-SHA256:DHE-RSA-AES256-GCM-SHA384';
    ssl_prefer_server_ciphers on;
    ssl_session_cache shared:SSL:10m;
    ssl_session_timeout 10m;
    ssl_session_tickets off;

    # Headers de seguridad
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains; preload" always;
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # [AQUI VAN TODOS TUS LOCATIONS EXISTENTES: /webhook/recall, /api/, /ws/, /]
    # Copia todos los bloques location del bloque HTTP original
}
```

3. **Copiar configuración modificada de vuelta al contenedor:**

```powershell
docker cp C:\nginx\nginx.conf nginx-container:/etc/nginx/conf/nginx.conf
```

4. **Verificar y recargar:**

```powershell
# Verificar sintaxis de nginx dentro del contenedor
docker exec nginx-container nginx -t

# Si la verificación es exitosa, recargar nginx
docker exec nginx-container nginx -s reload
```

#### Paso 5: Montar Volúmenes de Certificados (Recomendado)

Para que los certificados persistan y se actualicen fácilmente, monta el directorio como volumen en Docker:

**Si usas docker-compose**, añade o modifica en tu `docker-compose.yml`:

```yaml
services:
  nginx:
    image: nginx:latest
    volumes:
      - C:\nginx\conf:/etc/nginx/conf
      - C:\nginx\ssl:/etc/nginx/ssl  # Montar directorio de certificados
    ports:
      - "80:80"
      - "443:443"
```

**Si usas docker run**, añade el volumen al comando:

```powershell
docker run -d `
  -v C:\nginx\conf:/etc/nginx/conf `
  -v C:\nginx\ssl:/etc/nginx/ssl `
  -p 80:80 `
  -p 443:443 `
  --name nginx-container `
  nginx:latest
```

Con esto, cuando actualices los certificados en `C:\nginx\ssl\`, estarán disponibles automáticamente en el contenedor.

### Configuración Completa de Nginx

El archivo `nginx.conf.modificado` contiene la configuración completa con:

- ✅ Redirección HTTP → HTTPS en puerto 80
- ✅ Configuración SSL en puerto 443
- ✅ Todos los bloques `location` originales preservados
- ✅ Headers de seguridad añadidos
- ✅ Configuración SSL optimizada (TLS 1.2 y 1.3)

### Nota sobre Cloudflare

Si actualmente usas Cloudflare para manejar SSL, tienes dos opciones:

**Opción 1: Mantener Cloudflare (SSL Flexible)**
- Cloudflare maneja SSL entre cliente y Cloudflare
- Nginx solo necesita HTTP (puerto 80)
- No necesitas certificados en nginx
- **Desventaja**: No cumple con el requisito de Norton de redirección HTTP→HTTPS directa

**Opción 2: SSL Completo (Recomendado)**
- Cloudflare maneja SSL entre cliente y Cloudflare
- Nginx también tiene SSL entre Cloudflare y servidor
- Necesitas certificados en nginx (puedes usar certificados auto-firmados o Let's Encrypt)
- **Ventaja**: Cumple con todos los requisitos de seguridad

Para la Opción 2 en Windows Server, puedes usar certificados auto-firmados:

```powershell
# Generar certificado auto-firmado (solo para SSL completo con Cloudflare)
# Requiere OpenSSL instalado en Windows o usar el contenedor Docker

# Opción A: Usar OpenSSL en Windows (si está instalado)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 `
  -keyout C:\nginx\ssl\notetaker.key `
  -out C:\nginx\ssl\notetaker.crt `
  -subj "/CN=notetaker.cosgs.com"

# Opción B: Usar contenedor Docker con OpenSSL
docker run --rm -v C:\nginx\ssl:/ssl `
  alpine/openssl req -x509 -nodes -days 365 -newkey rsa:2048 `
  -keyout /ssl/notetaker.key `
  -out /ssl/notetaker.crt `
  -subj "/CN=notetaker.cosgs.com"
```

Para Let's Encrypt en Windows Server, puedes usar:
- Certbot en WSL (Windows Subsystem for Linux)
- Certbot en un contenedor Docker
- O importar certificados obtenidos desde otro servidor

## Verificación

Después de aplicar los cambios:

1. **Verificar redirección HTTP** (PowerShell):
   ```powershell
   # Usar Invoke-WebRequest
   $response = Invoke-WebRequest -Uri "http://notetaker.cosgs.com" -Method Head -MaximumRedirection 0 -ErrorAction SilentlyContinue
   $response.StatusCode  # Debe ser 301
   $response.Headers.Location  # Debe ser https://notetaker.cosgs.com/...

   # O usar curl si está disponible
   curl.exe -I http://notetaker.cosgs.com
   ```

2. **Verificar certificado SSL** (desde el contenedor o con OpenSSL):
   ```powershell
   # Opción A: Desde el contenedor Docker
   docker exec nginx-container openssl s_client -connect localhost:443 -servername notetaker.cosgs.com

   # Opción B: Si tienes OpenSSL en Windows
   openssl s_client -connect notetaker.cosgs.com:443 -servername notetaker.cosgs.com
   ```

3. **Verificar headers de seguridad** (PowerShell):
   ```powershell
   $response = Invoke-WebRequest -Uri "https://notetaker.cosgs.com" -Method Head
   $response.Headers  # Debe incluir Strict-Transport-Security y otros headers

   # O usar curl
   curl.exe -I https://notetaker.cosgs.com
   ```

4. **Verificar logs del contenedor**:
   ```powershell
   docker logs nginx-container
   docker logs nginx-container --tail 50  # Últimas 50 líneas
   ```

5. **Usar herramientas online**:
   - [SSL Labs SSL Test](https://www.ssllabs.com/ssltest/)
   - [Security Headers](https://securityheaders.com/)

## Solución de Problemas

### Error: "nginx: [emerg] SSL_CTX_use_certificate_file() failed"
- Verifica que la ruta del certificado es correcta en la configuración
- Verifica que el archivo existe en el contenedor:
  ```powershell
  docker exec nginx-container ls -la /etc/nginx/ssl/
  ```
- Verifica que el volumen está montado correctamente
- Verifica que el archivo existe en el host: `Test-Path C:\nginx\ssl\notetaker.crt`

### Error: "nginx: [emerg] SSL_CTX_use_PrivateKey_file() failed"
- Verifica que la clave privada coincide con el certificado
- Verifica que el archivo no está corrupto
- Verifica que ambos archivos están en el contenedor:
  ```powershell
  docker exec nginx-container ls -la /etc/nginx/ssl/
  ```

### Error: "nginx: [emerg] bind() to 0.0.0.0:443 failed (98: Address already in use)"
- Otro servicio o contenedor está usando el puerto 443
- Verifica qué está usando el puerto:
  ```powershell
  netstat -ano | findstr :443
  Get-NetTCPConnection -LocalPort 443
  ```
- Detén el servicio o contenedor que está usando el puerto
- O cambia el puerto mapeado en Docker: `-p 8443:443`

### Error: "Redirección infinita"
- Verifica que el bloque HTTPS tiene `ssl_certificate` y `ssl_certificate_key` correctamente configurados
- Verifica que no hay conflictos con Cloudflare si lo usas
- Revisa los logs del contenedor:
  ```powershell
  docker logs nginx-container
  ```

### Certificado no se aplica después de importar
- Los certificados se guardan en `E:\notetaker2.0\backend\ssl_certs\` pero nginx los necesita en `C:\nginx\ssl\`
- Debes copiar manualmente los archivos (ver Paso 3)
- Considera crear un script PowerShell automatizado para sincronizar certificados (ver abajo)

### Error: "docker cp: no such file or directory"
- Verifica que el contenedor está corriendo: `docker ps`
- Verifica el nombre exacto del contenedor: `docker ps --format "{{.Names}}"`
- Verifica que la ruta dentro del contenedor es correcta

### Error: "nginx: [emerg] open() failed (13: Permission denied)"
- En Windows con Docker, los permisos de archivos pueden ser diferentes
- Asegúrate de que los archivos en `C:\nginx\ssl\` son accesibles
- Verifica que el volumen está montado correctamente en docker-compose o docker run

## Script de Sincronización Automática (PowerShell)

Puedes crear un script PowerShell que sincronice automáticamente los certificados desde el backend:

```powershell
# sync_ssl_certs.ps1

$BackendSslDir = "E:\notetaker2.0\backend\ssl_certs"
$NginxSslDir = "C:\nginx\ssl"
$ContainerName = "nginx-container"  # Ajusta según tu contenedor

# Verificar que los archivos existen
if (-not (Test-Path "$BackendSslDir\fullchain.crt") -or -not (Test-Path "$BackendSslDir\private.key")) {
    Write-Host "Error: No se encontraron los archivos de certificado en $BackendSslDir" -ForegroundColor Red
    exit 1
}

# Crear directorio si no existe
if (-not (Test-Path $NginxSslDir)) {
    New-Item -ItemType Directory -Force -Path $NginxSslDir | Out-Null
    Write-Host "Directorio $NginxSslDir creado" -ForegroundColor Green
}

# Copiar archivos
Copy-Item "$BackendSslDir\fullchain.crt" -Destination "$NginxSslDir\notetaker.crt" -Force
Copy-Item "$BackendSslDir\private.key" -Destination "$NginxSslDir\notetaker.key" -Force

Write-Host "Certificados copiados a $NginxSslDir" -ForegroundColor Green

# Verificar que el contenedor está corriendo
$container = docker ps --filter "name=$ContainerName" --format "{{.Names}}"
if (-not $container) {
    Write-Host "Error: El contenedor $ContainerName no está corriendo" -ForegroundColor Red
    exit 1
}

# Verificar sintaxis de nginx dentro del contenedor
$testResult = docker exec $ContainerName nginx -t 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "Error: La configuración de nginx tiene errores:" -ForegroundColor Red
    Write-Host $testResult
    exit 1
}

# Recargar nginx
docker exec $ContainerName nginx -s reload
if ($LASTEXITCODE -eq 0) {
    Write-Host "Certificados SSL sincronizados y nginx recargado exitosamente" -ForegroundColor Green
} else {
    Write-Host "Error al recargar nginx" -ForegroundColor Red
    exit 1
}
```

**Ejecutar el script después de cada importación de certificado:**

```powershell
# Ejecutar el script
.\sync_ssl_certs.ps1

# O si tienes restricciones de ejecución de scripts:
powershell -ExecutionPolicy Bypass -File .\sync_ssl_certs.ps1
```

**Nota**: Si usas volúmenes montados en Docker, los archivos se sincronizan automáticamente y solo necesitas recargar nginx:

```powershell
# Script simplificado si usas volúmenes
docker exec nginx-container nginx -s reload
```

## Notas Importantes

- Los certificados se guardan en `E:\notetaker2.0\backend\ssl_certs\` por defecto
- Después de importar un certificado, **debes copiar los archivos a `C:\nginx\ssl\`** y recargar el contenedor
- Si usas volúmenes montados en Docker, los archivos se sincronizan automáticamente
- Si usas Cloudflare, considera si quieres SSL completo o flexible
- El contenedor Docker debe tener acceso a los archivos de certificado (usar volúmenes)
- En Windows Server con Docker, las rutas deben usar formato Windows (`C:\nginx\ssl`) en el host
- Dentro del contenedor, las rutas son Linux (`/etc/nginx/ssl`) - Docker traduce automáticamente con volúmenes
