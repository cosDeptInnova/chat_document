import { useEffect, useRef } from 'react';
import { usersApi } from '../services/api';

/**
 * Hook para enviar heartbeats periódicos al servidor.
 * Indica que el usuario está online y activo en la página.
 * 
 * @param userEmail Email del usuario autenticado
 * @param enabled Si está habilitado (por defecto true)
 * @param interval Intervalo en milisegundos entre heartbeats (por defecto 15000 = 15 segundos)
 */
export const useHeartbeat = (userEmail: string | null | undefined, enabled: boolean = true, interval: number = 15000) => {
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const isActiveRef = useRef(false);

  useEffect(() => {
    if (!enabled || !userEmail) {
      // Limpiar intervalo si está deshabilitado o no hay email
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      isActiveRef.current = false;
      return;
    }

    // Función para enviar heartbeat
    const sendHeartbeat = async () => {
      try {
        await usersApi.heartbeat(userEmail);
        isActiveRef.current = true;
      } catch (error) {
        console.error('Error enviando heartbeat:', error);
        // No detener el intervalo si hay un error, seguir intentando
      }
    };

    // Enviar heartbeat inmediatamente al montar
    sendHeartbeat();

    // Configurar intervalo para enviar heartbeats periódicos
    intervalRef.current = setInterval(sendHeartbeat, interval);

    // Enviar heartbeat cuando la ventana vuelve a estar activa (si estaba inactiva)
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible' && isActiveRef.current) {
        sendHeartbeat();
      }
    };

    // Enviar heartbeat cuando la página vuelve a estar en foco
    const handleFocus = () => {
      if (isActiveRef.current) {
        sendHeartbeat();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);
    window.addEventListener('focus', handleFocus);

    // Limpiar al desmontar
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      document.removeEventListener('visibilitychange', handleVisibilityChange);
      window.removeEventListener('focus', handleFocus);
      isActiveRef.current = false;
    };
  }, [userEmail, enabled, interval]);
};
