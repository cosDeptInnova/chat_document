/** Utilidades para manejar tokens JWT en el frontend. */

/**
 * Verifica si un token JWT está realmente expirado (ya pasó su fecha de expiración).
 * @param token Token JWT
 * @returns true si el token está expirado
 */
export function isTokenExpired(token: string): boolean {
  try {
    // Decodificar el payload del token (sin verificar la firma)
    const base64Url = token.split('.')[1];
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    const jsonPayload = decodeURIComponent(
      atob(base64)
        .split('')
        .map((c) => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join('')
    );
    
    const payload = JSON.parse(jsonPayload);
    const exp = payload.exp; // Timestamp de expiración
    
    if (!exp) {
      return true; // Si no hay exp, considerar como expirado
    }
    
    const now = Math.floor(Date.now() / 1000); // Timestamp actual en segundos
    const timeUntilExp = exp - now; // Segundos hasta expiración
    
    // Solo considerar expirado si ya pasó la fecha de expiración
    return timeUntilExp <= 0;
  } catch (error) {
    console.error('Error verificando token:', error);
    return true; // Si hay error, considerar como expirado
  }
}

/**
 * Verifica si un token JWT está próximo a expirar (útil para mostrar advertencias).
 * @param token Token JWT
 * @param hoursBefore Horas antes de la expiración para considerar "próximo a expirar"
 * @returns true si el token está próximo a expirar (pero aún no expirado)
 */
export function isTokenExpiringSoon(token: string, hoursBefore: number = 1): boolean {
  try {
    const base64Url = token.split('.')[1];
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    const jsonPayload = decodeURIComponent(
      atob(base64)
        .split('')
        .map((c) => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join('')
    );
    
    const payload = JSON.parse(jsonPayload);
    const exp = payload.exp;
    
    if (!exp) {
      return false;
    }
    
    const now = Math.floor(Date.now() / 1000);
    const timeUntilExp = exp - now;
    const hoursUntilExp = timeUntilExp / 3600;
    
    // Solo considerar "próximo a expirar" si aún no está expirado pero falta poco tiempo
    return timeUntilExp > 0 && hoursUntilExp < hoursBefore;
  } catch (error) {
    console.error('Error verificando token:', error);
    return false;
  }
}

/**
 * Obtiene la fecha de expiración de un token.
 * @param token Token JWT
 * @returns Fecha de expiración o null si no se puede decodificar
 */
export function getTokenExpiration(token: string): Date | null {
  try {
    const base64Url = token.split('.')[1];
    const base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
    const jsonPayload = decodeURIComponent(
      atob(base64)
        .split('')
        .map((c) => '%' + ('00' + c.charCodeAt(0).toString(16)).slice(-2))
        .join('')
    );
    
    const payload = JSON.parse(jsonPayload);
    const exp = payload.exp;
    
    if (!exp) {
      return null;
    }
    
    return new Date(exp * 1000); // Convertir de segundos a milisegundos
  } catch (error) {
    console.error('Error obteniendo expiración del token:', error);
    return null;
  }
}

