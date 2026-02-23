"""
Crea un usuario y token en VEXA Admin API para usar como VEXA_API_KEY.
Ejecutar una vez; luego poner el token en .env como VEXA_API_KEY.
"""
import os
import sys
import json
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

# Cargar .env para ADMIN_API_TOKEN si existe
env_file = backend_dir / ".env"
if env_file.exists():
    with open(env_file, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and not os.environ.get(k):
                    os.environ[k] = v

import httpx

# Admin API: Vexa Lite usa un solo puerto 8056; full stack puede usar 8057
ADMIN_URL = os.environ.get("VEXA_ADMIN_URL", os.environ.get("VEXA_API_BASE_URL", "http://localhost:8056"))
ADMIN_TOKEN = os.environ.get("ADMIN_API_TOKEN", "").strip() or os.environ.get("VEXA_API_KEY", "").strip() or "vexa-dev-admin-key-2025"

def main():
    base = ADMIN_URL.rstrip("/")
    headers = {
        "Content-Type": "application/json",
        "X-Admin-API-Key": ADMIN_TOKEN,
    }
    # Probar tambien X-Admin-API-Token por si la instalacion lo usa
    alt_headers = {**headers, "X-Admin-API-Token": ADMIN_TOKEN}
    # 1) Crear usuario
    print("1. Creando usuario en Admin API (%s)..." % base)
    try:
        with httpx.Client(timeout=15.0) as client:
            r = client.post(
                f"{base}/admin/users",
                headers=headers,
                json={
                    "email": "notetaker@cosgs.com",
                    "name": "Notetaker Bot",
                    "max_concurrent_bots": 10,
                },
            )
            if r.status_code == 403:
                r = client.post(
                    f"{base}/admin/users",
                    headers=alt_headers,
                    json={
                        "email": "notetaker@cosgs.com",
                        "name": "Notetaker Bot",
                        "max_concurrent_bots": 10,
                    },
                )
    except Exception as e:
        print("   Error de conexion:", e)
        return
    if r.status_code not in (200, 201) and base.endswith("8057"):
        base = base.replace(":8057", ":8056")
        print("   Probando Admin en puerto 8056...")
        try:
            with httpx.Client(timeout=15.0) as client:
                r = client.post(
                    f"{base}/admin/users",
                    headers=headers,
                    json={"email": "notetaker@cosgs.com", "name": "Notetaker Bot", "max_concurrent_bots": 10},
                )
        except Exception as e2:
            print("   Error:", e2)
    if r.status_code not in (200, 201):
        print("   Error:", r.status_code, r.text[:300])
        print("   Comprueba que ADMIN_API_TOKEN en el servidor VEXA coincida con el que usas.")
        return
    data = r.json()
    user_id = data.get("id")
    if not user_id:
        print("   Respuesta sin id:", data)
        return
    print("   OK. User ID:", user_id)
    # 2) Crear token
    print("2. Generando token para el usuario...")
    try:
        with httpx.Client(timeout=15.0) as client:
            r2 = client.post(
                f"{base}/admin/users/{user_id}/tokens",
                headers={"X-Admin-API-Key": ADMIN_TOKEN},
            )
    except Exception as e:
        print("   Error:", e)
        return
    if r2.status_code not in (200, 201):
        print("   Error:", r2.status_code, r2.text[:300])
        return
    token_data = r2.json()
    token = token_data.get("token")
    if not token:
        print("   Respuesta sin token:", token_data)
        return
    print("   OK. Token generado.")
    print("")
    print("Anade a tu .env (backend):")
    print("VEXA_API_KEY=" + token)
    print("")
    print("(Guarda este token; no se puede volver a ver.)")

if __name__ == "__main__":
    main()
