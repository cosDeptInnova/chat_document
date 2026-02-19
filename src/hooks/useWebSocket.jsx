import { useState, useRef, useCallback, useEffect } from 'react';

export function useWebSocket(options = {}) {
  const [connectionStatus, setConnectionStatus] = useState('disconnected'); // 'disconnected' | 'connecting' | 'connected' | 'reconnecting'
  const [latency, setLatency] = useState(45);

  const socketRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const reconnectAttempts = useRef(0);
  const maxReconnectAttempts = 10;
  const baseReconnectDelay = 1000;
  const messageQueueRef = useRef([]);

  // Keep-alive system para prevenir timeouts durante TTS largos
  const keepAliveTimerRef = useRef(null);

  // FIX #1: Resource Tracker para cleanup sistemático
  const resourceTrackerRef = useRef({
    timers: new Set(),
    cleanup: () => {
      // Limpiar todos los timers tracked
      resourceTrackerRef.current.timers.forEach(timer => {
        clearTimeout(timer);
        clearInterval(timer);
      });
      resourceTrackerRef.current.timers.clear();

      // Limpiar keep-alive específicamente
      if (keepAliveTimerRef.current) {
        clearInterval(keepAliveTimerRef.current);
        keepAliveTimerRef.current = null;
      }

      // Limpiar reconnect timer específicamente
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = null;
      }

      // FIX #4: Limpiar message queue para prevenir crecimiento indefinido
      messageQueueRef.current = [];
    }
  });

  // Refs para callbacks para evitar stale closures
  const onMessageRef = useRef(options.onMessage);
  const onTTSAudioRef = useRef(options.onTTSAudio);
  const onConnectRef = useRef(options.onConnect);
  const onDisconnectRef = useRef(options.onDisconnect);
  const onErrorRef = useRef(options.onError);
  const audioContextRef = useRef(options.audioContextRef);

  // Sincronizar refs cuando cambien las options
  useEffect(() => { onMessageRef.current = options.onMessage; }, [options.onMessage]);
  useEffect(() => { onTTSAudioRef.current = options.onTTSAudio; }, [options.onTTSAudio]);
  useEffect(() => { onConnectRef.current = options.onConnect; }, [options.onConnect]);
  useEffect(() => { onDisconnectRef.current = options.onDisconnect; }, [options.onDisconnect]);
  useEffect(() => { onErrorRef.current = options.onError; }, [options.onError]);
  useEffect(() => { audioContextRef.current = options.audioContextRef; }, [options.audioContextRef]);

  const getWebSocketUrl = useCallback(() => {
    // URL del backend
    // Always use the COSMOS production endpoint as specified
    return 'wss://cosmos.cosgs.com/voice';
  }, []);

  // FIX #2: Keep-alive mejorado con cleanup garantizado
  const startKeepAlive = useCallback(() => {
    // Limpiar timer existente si hay uno
    if (keepAliveTimerRef.current) {
      clearInterval(keepAliveTimerRef.current);
      resourceTrackerRef.current.timers.delete(keepAliveTimerRef.current);
    }

    const keepAliveTimer = setInterval(() => {
      // Mantener AudioContext activo (implica conexión activa)
      if (audioContextRef.current?.current) {
        if (audioContextRef.current.current.state !== 'running') {
          audioContextRef.current.current.resume();
        }
      }

      // Enviar ping adicional para mantener WebSocket vivo
      if (socketRef.current?.readyState === WebSocket.OPEN) {
        try {
          socketRef.current.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
        } catch (error) {
          console.warn('[KEEPALIVE] Error enviando ping:', error);
        }
      }
      // Keep-alive silencioso durante TTS
    }, 2000); // Cada 2 segundos

    keepAliveTimerRef.current = keepAliveTimer;
    // Track timer para cleanup automático
    resourceTrackerRef.current.timers.add(keepAliveTimer);
  }, []);

  // Función para parar keep-alive con cleanup garantizado
  const stopKeepAlive = useCallback(() => {
    if (keepAliveTimerRef.current) {
      clearInterval(keepAliveTimerRef.current);
      resourceTrackerRef.current.timers.delete(keepAliveTimerRef.current);
      keepAliveTimerRef.current = null;
    }
  }, []);

  // FIX #3: Event handlers con referencias débiles para evitar closures circulares
  const createWebSocketHandlers = useCallback(() => {
    return {
      onopen: () => {
        console.log("[WebSocket] Conexión establecida");
        setConnectionStatus('connected');
        reconnectAttempts.current = 0;
        onConnectRef.current?.();

        // Flush queued messages
        const queuedMessages = messageQueueRef.current;
        messageQueueRef.current = [];
        queuedMessages.forEach(message => {
          if (socketRef.current?.readyState === WebSocket.OPEN) {
            socketRef.current.send(JSON.stringify(message));
          }
        });
      },

      onmessage: (event) => {
        // Handle binary data (TTS audio from backend)
        if (event.data instanceof Blob) {
          console.log('[WebSocket] TTS audio recibido (Blob):', event.data.size, 'bytes');
          onTTSAudioRef.current?.(event.data);
          return;
        }

        if (event.data instanceof ArrayBuffer) {
          console.log('[WebSocket] TTS audio recibido (ArrayBuffer):', event.data.byteLength, 'bytes');
          const blob = new Blob([event.data], { type: 'audio/ogg' });
          onTTSAudioRef.current?.(blob);
          return;
        }

        // Handle text messages - try JSON first, then fallback to text tokens
        const message = event.data.trim();

        try {
          // If it looks like JSON, parse it
          if (message.startsWith('{') || message.startsWith('[')) {
            const data = JSON.parse(message);

            console.log('[WebSocket] Mensaje JSON recibido:', data.type);

            // Handle pong for latency calculation
            if (data.type === 'pong') {
              const currentTime = Date.now();
              const roundTripTime = currentTime - data.timestamp;
              setLatency(Math.round(roundTripTime / 2));
              return;
            }

            // CAMBIO: Ya no reseteamos isStreamingRef aquí
            // Dejamos que el VAD maneje el estado del streaming
            if (data.type === 'speech_end') {
              console.log('[WebSocket] speech_end recibido');
            }

            onMessageRef.current?.(data);
            return;
          }

          // Handle text tokens (raw strings from backend)
          console.log('[WebSocket] Token de texto recibido:', message);

          // Process each line as a potential token
          const lines = message.split('\n').map(line => line.trim()).filter(Boolean);
          for (const line of lines) {
            let tokenData = null;

            if (line === 'streaming_ready') {
              tokenData = { type: 'streaming_ready' };
              console.log('[WebSocket] streaming_ready confirmado por backend');
            } else if (line === 'interruption_detected') {
              tokenData = { type: 'interruption_detected' };
              console.log('[WebSocket] interruption_detected recibido');
            } else if (line === 'tts_interrupted') {
              tokenData = { type: 'tts_interrupted' };
              console.log('[WebSocket] tts_interrupted recibido');
            } else if (line.startsWith('pong:')) {
              const timestamp = parseInt(line.split(':')[1]);
              const currentTime = Date.now();
              const roundTripTime = currentTime - timestamp;
              setLatency(Math.round(roundTripTime / 2));
              return;
            } else {
              // Si no es un token de comando, tratarlo como mensaje del agente
              tokenData = {
                type: 'agent_message',
                text: line,
                tts_ready: true // Asumir que tendrá TTS
              };
            }

            if (tokenData) {
              onMessageRef.current?.(tokenData);
            }
          }
        } catch (error) {
          console.error('[WebSocket] Error procesando mensaje:', error);
          console.error('Datos raw:', event.data);
          console.error('Tipo de datos:', typeof event.data);
        }
      },

      onclose: (event) => {
        console.log(`[WebSocket] Conexión cerrada (code: ${event.code})`);
        setConnectionStatus('disconnected');
        // Parar keep-alive cuando se cierre la conexión
        stopKeepAlive();
        onDisconnectRef.current?.();

        // Auto-reconnect if not manually closed
        if (event.code !== 1000 && reconnectAttempts.current < maxReconnectAttempts) {
          scheduleReconnect();
        }
      },

      onerror: (error) => {
        console.error('[WebSocket] Error:', error);
        onErrorRef.current?.(error);
      }
    };
  }, [stopKeepAlive]);

  const connect = useCallback(() => {
    if (socketRef.current?.readyState === WebSocket.OPEN){
      console.log('[WebSocket] Ya existe una conexión activa, ignorando connect()');
      return;
    }

    // FIX #1: Cleanup antes de nueva conexión
    resourceTrackerRef.current.cleanup();
    setConnectionStatus('connecting');
    console.log('[WebSocket] Iniciando conexión...');

    try {
      const wsUrl = getWebSocketUrl();
      socketRef.current = new WebSocket(wsUrl);
      // Configure to receive TTS audio as Blob
      socketRef.current.binaryType = 'blob';

      // FIX #3: Asignar handlers con referencias débiles
      const handlers = createWebSocketHandlers();
      socketRef.current.onopen = handlers.onopen;
      socketRef.current.onmessage = handlers.onmessage;
      socketRef.current.onclose = handlers.onclose;
      socketRef.current.onerror = handlers.onerror;
    } catch (error) {
      console.error('[WebSocket] Error creando conexión:', error);
      setConnectionStatus('disconnected');
    }
  }, [getWebSocketUrl, createWebSocketHandlers]);

  // FIX #1: scheduleReconnect con timer tracking
  const scheduleReconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      resourceTrackerRef.current.timers.delete(reconnectTimeoutRef.current);
    }

    const delay = baseReconnectDelay * Math.pow(2, reconnectAttempts.current);
    console.log(`[WebSocket] Reintentando conexión en ${delay}ms (intento ${reconnectAttempts.current + 1}/${maxReconnectAttempts})`);
    setConnectionStatus('reconnecting');
    reconnectAttempts.current++;

    const timeoutId = setTimeout(() => {
      connect();
    }, delay);

    reconnectTimeoutRef.current = timeoutId;
    // Track timer para cleanup automático
    resourceTrackerRef.current.timers.add(timeoutId);
  }, [connect]);

  const disconnect = useCallback(() => {
    console.log('[WebSocket] Desconectando...');
    
    // FIX #1: Cleanup sistemático de todos los recursos
    resourceTrackerRef.current.cleanup();

    if (socketRef.current) {
      socketRef.current.close(1000); // Normal closure
      socketRef.current = null;
    }

    // Clear any queued messages (ya incluido en cleanup)
    setConnectionStatus('disconnected');
    reconnectAttempts.current = 0;
  }, []);

  const sendMessage = useCallback((message) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      try {
        socketRef.current.send(JSON.stringify(message));
        console.log('[WebSocket] Mensaje enviado:', message.type || 'unknown');
        return true;
      } catch (error) {
        console.error('[WebSocket] Error enviando mensaje:', error);
        return false;
      }
    } else if (connectionStatus === 'connecting' || connectionStatus === 'reconnecting') { // Queue message if not connected (but connecting)
        // FIX #4: Limitar tamaño de queue para evitar crecimiento indefinido
        if (messageQueueRef.current.length < 100) { // Límite de 100 mensajes
            messageQueueRef.current.push(message);
            console.log('[WebSocket] Message queued (conectando):', message);
        } else {
            console.warn('[WebSocket] Cola de mensajes llena, descartando el más antiguo');
            messageQueueRef.current.shift(); // Remover el más antiguo
            messageQueueRef.current.push(message);
        }
        return true;
    } else {
        console.warn('[WebSocket] No conectado, mensaje no enviado:', message);
        return false;
    }
  }, [connectionStatus]);

  const sendBinaryData = useCallback((data) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      try {
        socketRef.current.send(data);
        return true;
      } catch (error) {
        console.error('[WebSocket] Error enviando audio:', error);
        return false;
      }
    } else {
      console.warn('[WebSocket] No conectado, audio no enviado');
      return false;
    }
  }, []);

  const sendRawMessage = useCallback((rawMessage) => {
    if (socketRef.current?.readyState === WebSocket.OPEN) {
      try {
        socketRef.current.send(rawMessage);
        console.log('[WebSocket] Mensaje raw enviado:', rawMessage);
        return true;
      } catch (error) {
        console.error('[WebSocket] Error enviando mensaje raw:', error);
        return false;
      }
    } else {
      console.warn('[WebSocket] No conectado, mensaje raw no enviado:', rawMessage);
      return false;
    }
  }, []);

  // FIX #1: Cleanup mejorado en unmount
  useEffect(() => {
    return () => {
      console.log('[WebSocket] Limpieza en unmount');
      // Cleanup sistemático de todos los recursos
      resourceTrackerRef.current.cleanup();

      // Cerrar WebSocket si está abierto
      if (socketRef.current) {
        socketRef.current.close(1000);
        socketRef.current = null;
      }
    };
  }, []);

  return {
    connectionStatus,
    latency,
    connect,
    disconnect,
    sendMessage,
    sendBinaryData,
    sendRawMessage,
    isConnected: connectionStatus === 'connected',
    startKeepAlive, // Exportar para control externo
    stopKeepAlive   // Exportar para control externo
  };
}