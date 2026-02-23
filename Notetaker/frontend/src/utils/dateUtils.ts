import { format } from 'date-fns';
import { es } from 'date-fns/locale';

/**
 * Parsea scheduled_start_time (ISO o naive UTC) a Date en hora local.
 * Misma lógica que formatLocalDateTime para consistencia.
 */
function parseMeetingDate(dateString: string): Date {
  if (dateString.includes('T') && !dateString.includes('Z') && !dateString.includes('+') && !dateString.includes('-', 10)) {
    return new Date(dateString + 'Z');
  }
  return new Date(dateString);
}

/**
 * Indica si la reunión cae en el día actual (fecha local del usuario).
 * Cambia según el día actual.
 */
export function isMeetingToday(dateString: string): boolean {
  const d = parseMeetingDate(dateString);
  const today = new Date();
  return d.getFullYear() === today.getFullYear() &&
    d.getMonth() === today.getMonth() &&
    d.getDate() === today.getDate();
}

/**
 * Convierte una fecha UTC (naive o ISO string) a hora local y la formatea.
 * 
 * @param dateString - Fecha en formato ISO string o naive datetime string (asume UTC)
 * @param formatString - Formato de fecha (por defecto: "d 'de' MMMM, yyyy 'a las' HH:mm")
 * @returns Fecha formateada en hora local
 */
export function formatLocalDateTime(
  dateString: string,
  formatString: string = "d 'de' MMMM, yyyy 'a las' HH:mm"
): string {
  // Si la fecha viene sin timezone (naive), asumimos que es UTC
  // Si viene con 'Z' o timezone, JavaScript lo interpreta correctamente
  let date: Date;
  
  if (dateString.includes('T') && !dateString.includes('Z') && !dateString.includes('+') && !dateString.includes('-', 10)) {
    // Es naive datetime (sin timezone), asumimos UTC
    // Agregar 'Z' para indicar UTC
    date = new Date(dateString + 'Z');
  } else {
    // Ya tiene timezone o es ISO string completo
    date = new Date(dateString);
  }
  
  return format(date, formatString, { locale: es });
}

