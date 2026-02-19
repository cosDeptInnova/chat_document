import { useState, useRef, useEffect, useCallback } from "react";
import { FiMic, FiVolume2, FiXCircle, FiTrash2, FiDownload } from "react-icons/fi";
import { FaRobot } from "react-icons/fa";

import cosmosIconGradient from "../../images/cosmos_2_1758106496232.png";
import { Button } from "../../components/utils/Button";
import { useAudioControls } from "../../hooks/useAudioControls";
import { useWebSocket } from "../../hooks/useWebSocket";
import VoiceControls from "../../components/voice/VoiceControls";

import { cn } from "../../lib/utils";

export default function VoiceAgentPage({ isDarkMode, onEndCall, chatId, onAgentSpekingChange }) {

  const [isOpen, setIsOpen] = useState(true);
  const [videosPreloaded, setVideosPreloaded] = useState(false);
  const [voiceMessages, setVoiceMessages] = useState([]);
  const [transcriptionPreview, setTranscriptionPreview] = useState("Escuchando... habla ahora");
  const [sessionId, setSessionId] = useState(null);
  const [conversationCount, setConversationCount] = useState(0);
  const [sessionDuration, setSessionDuration] = useState("00:00");
  const [isBackendReady, setIsBackendReady] = useState(false);
  //const [recordingTimeoutId, setRecordingTimeoutId] = useState(null);
  const [currentPlayingId, setCurrentPlayingId] = useState(null);
  const [ttsProgress, setTtsProgress] = useState(0);

  const [currentUtteranceId, setCurrentUtteranceId] = useState(null);
  const [lastProcessedUtteranceId, setLastProcessedUtteranceId] = useState(null);

  // NUEVO: Flag para saber si el audio ya está listo
  const [isAudioReady, setIsAudioReady] = useState(false);

  const currentUtteranceIdRef = useRef(null);
  const lastProcessedUtteranceIdRef = useRef(null);
  const messagesEndRef = useRef(null);
  const recordingTimeoutIdRef = useRef(null);
  const sessionStartTime = useRef(null);

  // NUEVO: Flag para evitar inicialización duplicada
  const isInitializingRef = useRef(false);

  const {
        micMuted,
        ttsVolume,
        audioLevel,
        vadState,
        isUserSpeaking,
        isAgentSpeaking,
        audioDevice,
        bufferUsage,
        toggleMicrophone,
        setTtsVolume: updateTtsVolume,
        setupAudioControls,
        cleanupAudioControls,
        startStreaming,
        stopStreaming,
        playTTS,
        stopTTS,
        audioContextRef,
        getBufferMetrics,
        isStreamingRef,
  } = useAudioControls();

  // Exponer la ref global del streaming para uso del WebSocket
    /*useEffect(() => {
      if (typeof window !== "undefined") {
        window.isStreamingRef = isStreamingRef;
      }
    }, [isStreamingRef]);*/

  const clearRecordingAnimation = useCallback(() => {
    if (recordingTimeoutIdRef.current) {
      clearTimeout(recordingTimeoutIdRef.current);
      //setRecordingTimeoutId(null);
      recordingTimeoutIdRef.current = null;
    }

    setVoiceMessages((prev) => prev.filter((m) => m.type !== "recording"));
  }, [/*recordingTimeoutId*/]);

  const handleSocketMessageRef = useRef(null);

  const {
        connectionStatus,
        latency,
        connect,
        disconnect,
        sendMessage,
        sendBinaryData,
        sendRawMessage,
        startKeepAlive,
        stopKeepAlive,
    } = useWebSocket({
            audioContextRef,
            onTTSAudio: (audioBlob) => {
                const audioUrl = URL.createObjectURL(audioBlob);

                const updateProgress = (progress) => {
                  setTtsProgress(progress);
                  if (progress >= 100) {
                      setTimeout(() => {
                          setTtsProgress(0);
                          setCurrentPlayingId(null);
                      }, 500);
                  }
                };

                startKeepAlive();

                playTTS(audioUrl, updateProgress).finally(() => {
                    URL.revokeObjectURL(audioUrl);
                    setCurrentPlayingId(null);
                    stopKeepAlive();
                });
            },

            //onMessage: handleSocketMessage,
            onMessage: (data) => handleSocketMessageRef.current?.(data),
        }
  );

  // CAMBIO: Exponer variables globales de forma más segura
  useEffect(() => {
    if (typeof window === "undefined") return;

    // asignaciones solo si existen para evitar errores
    window.isStreamingRef = isStreamingRef;
    window.sendRawMessage = sendRawMessage;
    window.connectionStatus = connectionStatus;
    window.isBackendReady = isBackendReady;
    // opcional: window.sendBinaryData = sendBinaryData;

    return () => {
      try {
        delete window.isStreamingRef;
        delete window.sendRawMessage;
        delete window.connectionStatus;
        delete window.isBackendReady;
        // delete window.sendBinaryData;
      } catch (e) { /* noop */ }
    };
  }, [isStreamingRef, sendRawMessage, connectionStatus, isBackendReady /*, sendBinaryData */]);

  // useCallback para memoizar la función de manejo de mensajes
  const handleSocketMessage = useCallback(
    (data) => {
      console.log('[SOCKET IN]', data.type, data);
      switch (data.type) {
        case "user_transcript":
          const utteranceId = data.utterance_id || Date.now();

          if (
            !currentUtteranceIdRef.current ||
            utteranceId >= currentUtteranceIdRef.current
          ) {
            setCurrentUtteranceId(utteranceId);
            currentUtteranceIdRef.current = utteranceId;
            console.log("Primer mensaje detectado, conversación iniciada");

            setVoiceMessages((prev) =>
              prev.filter((msg) => {
                if (
                  msg.utteranceId &&
                  msg.utteranceId !== utteranceId &&
                  (msg.type === "user" || msg.type === "agent")
                )
                  return true;

                if (
                  !msg.utteranceId &&
                  (msg.type === "user" || msg.type === "agent")
                )
                  return true;

                if (
                  msg.utteranceId === utteranceId &&
                  (msg.type === "user_progressive" || msg.type === "recording")
                )
                  return false;

                if (msg.utteranceId === utteranceId && msg.type === "user")
                  return false;

                return true;
              })
            );

            const userMessage = {
              id: `user_${Date.now()}_${Math.random()
                .toString(36)
                .substr(2, 9)}`,
              type: "user",
              content: data.text,
              timestamp: new Date().toLocaleTimeString("es-ES", {
                hour: "2-digit",
                minute: "2-digit",
                hour12: false,
              }),
              utteranceId: utteranceId,
            };
            setVoiceMessages((prev) => [...prev, userMessage]);
            setConversationCount((prev) => prev + 1);

            console.log(`Utterance #${utteranceId}: Transcripción procesada`);
          } else {
            console.log(
              `Utterance #${utteranceId}: Ignorada (obsoleta, actual: #${currentUtteranceIdRef.current})`
            );
            return;
          }

          const timeoutId = setTimeout(() => {
            setVoiceMessages((prev) => [
              ...prev,
              {
                id: `recording_${Date.now()}_${Math.random()
                  .toString(36)
                  .substr(2, 9)}`,
                type: "recording",
                content: "respondiendo",
                timestamp: new Date().toLocaleTimeString("es-ES", {
                  hour: "2-digit",
                  minute: "2-digit",
                  hour12: false,
                }),
                isRecording: true,
                utteranceId: utteranceId,
              },
            ]);
          }, 2000);

          recordingTimeoutIdRef.current = timeoutId;
          break;
        
        case "word_detected":
          console.log(`[DEPRECATED] word_detected ignorado - usar user_progressive`);
          break;

        case "user_progressive":
          console.log(
            `[USER_PROGRESSIVE] "${data.content}" (utterance: ${data.utterance_id}, final: ${data.is_final})`
          );

          const userProgressiveUtteranceId = data.utterance_id || Date.now();

          if (
            !currentUtteranceIdRef.current ||
            userProgressiveUtteranceId >= currentUtteranceIdRef.current
          ) {
            setCurrentUtteranceId(userProgressiveUtteranceId);
            currentUtteranceIdRef.current = userProgressiveUtteranceId;

            setVoiceMessages((prev) => {
              const filteredMessages = prev.filter((msg) => {
                if (
                  msg.type === "user" &&
                  msg.utteranceId !== userProgressiveUtteranceId
                )
                  return true;
                if (msg.type === "agent") return true;
                if (
                  msg.type === "user_progressive" &&
                  msg.utteranceId === userProgressiveUtteranceId
                )
                  return false;
                if (
                  msg.type === "recording" &&
                  msg.utteranceId === userProgressiveUtteranceId
                )
                  return false;
                return true;
              });

              const progressiveMessage = {
                id: `user_progressive_${userProgressiveUtteranceId}_${Date.now()}`,
                type: "user_progressive",
                content: data.content,
                timestamp: new Date().toLocaleTimeString("es-ES", {
                  hour: "2-digit",
                  minute: "2-digit",
                  hour12: false,
                }),
                utteranceId: userProgressiveUtteranceId,
                isProgressive: true,
                confidence: data.confidence,
              };

              console.log(`[RETELL] Actualizando línea usuario: "${data.content}"`);
              return [...filteredMessages, progressiveMessage];
            });
          } else {
            console.log(
              `[USER_PROGRESSIVE] Ignorado - utterance obsoleta ${userProgressiveUtteranceId} vs actual ${currentUtteranceIdRef.current}`
            );
          }
          break;
        
        case "partial_transcript":
          console.log(`[PARTIAL] ${data.text} (confidence: ${data.confidence})`);
          break;

        case "final_transcript":
          console.log(`[FINAL] ${data.text} (${data.latency_ms}ms)`);
          break;

        case "agent_message":
          const responseUtteranceId =
            data.utterance_id ?? currentUtteranceIdRef.current;

          const shouldProcessMessage =
            !responseUtteranceId ||
            !lastProcessedUtteranceIdRef.current ||
            responseUtteranceId >= lastProcessedUtteranceIdRef.current;

          if (shouldProcessMessage) {
            clearRecordingAnimation();

            const agentMessage = {
              id: `agent_${Date.now()}_${Math.random()
                .toString(36)
                .substr(2, 9)}`,
              type: "agent",
              content: data.text,
              timestamp: new Date().toLocaleTimeString("es-ES", {
                hour: "2-digit",
                minute: "2-digit",
                hour12: false,
              }),
              isPlaying: data.tts_ready,
              playProgress: 0,
              utteranceId: responseUtteranceId,
            };
            setVoiceMessages((prev) => [...prev, agentMessage]);
            setConversationCount((prev) => prev + 1);
            if (responseUtteranceId) {
              setLastProcessedUtteranceId(responseUtteranceId);
              lastProcessedUtteranceIdRef.current = responseUtteranceId;
            }

            if (data.tts_ready) {
              setCurrentPlayingId(agentMessage.id);
            }

            console.log(
              `Utterance #${responseUtteranceId || "null"}: Respuesta de COSMOS procesada`
            );
          } else {
            console.log(
              `Utterance #${responseUtteranceId || "null"}: Respuesta ignorada (más antigua que #${lastProcessedUtteranceIdRef.current})`
            );
          }
          break;

        case "speech_start":
          setTranscriptionPreview("Escuchando...");
          break;

        case "speech_end":
          console.log("[SOCKET] speech_end recibido - reiniciando streaming flag");
          setTranscriptionPreview("Procesando...");
          // IMPORTANTE: NO resetear isStreamingRef aquí, lo hace el VAD
          break;

        case "interruption_detected":
          console.log("[VAD] Interrupción detectada - Parando audio");
          clearRecordingAnimation();
          stopTTS();
          stopKeepAlive();
          console.log("[VAD] Audio pausado por interrupción");
          break;

        case "tts_interrupted":
          console.log("[VAD] TTS interrumpido por backend");
          stopTTS();
          setVoiceMessages((prev) =>
            prev.map((msg) => ({
              ...msg,
              isPlaying: false,
            }))
          );
          break;

        case "streaming_ready":
          console.log("✅ Backend confirmó streaming_ready - iniciando audio");
          setIsBackendReady(true);
          break;

        default:
          console.log(`[WS] Tipo desconocido: ${data.type}`);
          break;
      }
    },
    [
      setVoiceMessages,
      setCurrentUtteranceId,
      setConversationCount,
      setLastProcessedUtteranceId,
      setCurrentPlayingId,
      setTranscriptionPreview,
      clearRecordingAnimation,
      stopTTS,
      stopKeepAlive,
      setIsBackendReady,
    ]
  );

  // Actualizamos el ref para que useWebSocket siempre tenga la última versión
  handleSocketMessageRef.current = handleSocketMessage;

  useEffect(() => {
        currentUtteranceIdRef.current = currentUtteranceId;
  }, [currentUtteranceId]);

  useEffect(() => {
        lastProcessedUtteranceIdRef.current = lastProcessedUtteranceId;
  }, [lastProcessedUtteranceId]);

  // Paso 1: Generar session ID
  useEffect(() => {
    if (!isOpen || sessionId) return;

    const navEntries = performance.getEntriesByType("navigation");
    const isPageReload = navEntries[0]?.type === "reload";
    const voiceSessionActive = sessionStorage.getItem("cosmos_voice_active");

    if (isPageReload || !voiceSessionActive) {
      console.log("RECARGA DETECTADA - Generando nueva session ID");
      sessionStorage.removeItem("cosmos_voice_session_id");
    } else {
      console.log("Reconexión temporal - manteniendo session");
    }

    sessionStorage.setItem("cosmos_voice_active", Date.now().toString());

    let existingSessionId = sessionStorage.getItem("cosmos_voice_session_id");

    if (!existingSessionId) {
      const timestamp = Date.now();
      const random = Math.random().toString(36).substr(2, 8);
      existingSessionId = `voice_session_${timestamp}_${random}`;
      sessionStorage.setItem("cosmos_voice_session_id", existingSessionId);
      console.log(`NUEVA session_id: ${existingSessionId}`);
    } else {
      console.log(`Session_id reutilizada: ${existingSessionId}`);
    }

    setSessionId(existingSessionId);
    sessionStartTime.current = new Date();
  }, [isOpen, sessionId]);

  // Paso 2: Configurar audio (solo cuando hay sessionId)
  useEffect(() => {
    if (!isOpen || !sessionId || isAudioReady || isInitializingRef.current) return;

    isInitializingRef.current = true;

    const initAudio = async () => {
      try {
        console.log("Configurando controles de audio...");
        await setupAudioControls();
        setIsAudioReady(true);
        console.log("Audio configurado correctamente");
      } catch (error) {
        console.error("Error configurando audio:", error);
      } finally {
        isInitializingRef.current = false;
      }
    };

    initAudio();

    return () => {
      if (isAudioReady) {
        cleanupAudioControls();
        setIsAudioReady(false);
      }
    };
  }, [isOpen, sessionId, isAudioReady, setupAudioControls, cleanupAudioControls]);

  // Paso 3: Conectar WebSocket (solo cuando audio esté listo)
  useEffect(() => {
    if (!isOpen || !sessionId || !isAudioReady) return;

    // Solo conectar si NO está conectado/conectando
    if (connectionStatus === 'disconnected') {
      console.log("Conectando WebSocket...");
      connect();
    }

    // NO llamar disconnect() en el cleanup
    // El WebSocket se limpia automáticamente cuando isOpen=false
  }, [isOpen, sessionId, isAudioReady, connectionStatus, connect]);

  // Paso 4: Enviar handshake (solo cuando WebSocket esté conectado)
  useEffect(() => {
    if (connectionStatus !== "connected" || !sessionId) return;

    console.log("Enviando handshake al backend...");
    setIsBackendReady(false);

    // Enviar session_id
    sendMessage({
      type: "set_session_id",
      session_id: sessionId,
    });

    // Enviar start_streaming
    sendRawMessage("start_streaming");
    console.log("Esperando confirmación streaming_ready del backend...");
  }, [connectionStatus, sessionId, sendMessage, sendRawMessage]);

  // Paso 5: Iniciar streaming de audio (solo cuando backend esté listo)
  useEffect(() => {
    if (!isBackendReady || connectionStatus !== "connected" || isStreamingRef.current) return;

    const handleAudioData = (audioData) => {
      if (micMuted) return;
      if (connectionStatus !== "connected") return;

      sendBinaryData(audioData);
    };

    if (startStreaming(handleAudioData)) {
      console.log("Streaming de audio iniciado - listo para recibir voz");
    }

    return () => {
      if (isStreamingRef.current) {
        stopStreaming();
      }
    };
  }, [isBackendReady, connectionStatus, sendBinaryData, startStreaming, stopStreaming, micMuted, isStreamingRef]);

  useEffect(() => {
    if (videosPreloaded) {
      console.log("VoiceAgent recibió: videos ya precargados desde NuevoChatMainPanel.jsx");
    } else {
      console.log("VoiceAgent recibió: videos aún no precargados");
    }
  }, [videosPreloaded]);

  /**/
  const scrollToBottom = useCallback(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({
        behavior: "smooth",
        block: "end",
      });
    }
  }, []);

  /**/
  useEffect(() => {
    scrollToBottom();
  }, [voiceMessages, scrollToBottom]);

  /**/
  useEffect(() => {
    if (vadState === "processing") {
      setTimeout(() => scrollToBottom(), 100);
    }
  }, [vadState, scrollToBottom]);

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.ctrlKey) {
        switch (e.key) {
          case "m":
            e.preventDefault();
            toggleMicrophone();
            break;
          case "ArrowUp":
            e.preventDefault();
            updateTtsVolume(Math.min(1, ttsVolume + 0.1));
            break;
          case "ArrowDown":
            e.preventDefault();
            updateTtsVolume(Math.max(0, ttsVolume - 0.1));
            break;
        }
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [toggleMicrophone, updateTtsVolume, ttsVolume]);

  useEffect(() => {
    // Cleanup cuando el componente se desmonta o isOpen cambia a false
    return () => {
      if (!isOpen) {
        console.log('[VoiceAgentPage] Limpieza completa al cerrar');
        disconnect();
        cleanupAudioControls();
        sessionStorage.removeItem("cosmos_voice_active");
      }
    };
  }, [isOpen, disconnect, cleanupAudioControls]);

  // Notificar al layout (para que se lo pase al sidebar) cuando cambia isAgentSpeking para que se muestre un video u otro
  useEffect(() => {
    if (onAgentSpekingChange){
      onAgentSpekingChange(isAgentSpeaking)
    }
  }, [isAgentSpeaking, onAgentSpekingChange])

  const handleClearConversation = () => {
    setVoiceMessages([]);
    setConversationCount(0);
  };

  const handleExportConversation = () => {
    const content = voiceMessages
      .map(
        (msg) =>
          `[${msg.timestamp}] ${
            msg.type === "user" || msg.type === "user_progressive"
              ? "Usuario"
              : "COSMOS"
          }: ${msg.content}`
      )
      .join("\n");

    const blob = new Blob([content], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `cosmos_conversacion_${sessionId}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const handleEndCall = useCallback(() => {
    console.log("[VOICE] Finalizando llamada - limpieza completa");

    // 1. Limpieza de recursos
    clearRecordingAnimation();
    stopStreaming();
    stopTTS();
    stopKeepAlive();
    cleanupAudioControls();
    disconnect();

    // 2. Limpieza de sesión
    sessionStorage.removeItem("cosmos_voice_session_id");
    sessionStorage.removeItem("cosmos_voice_active");
    console.log("[VOICE] Session ID limpiada - próxima apertura generará nueva");

    // 3. Callback opcional
    if (onEndCall) {
      onEndCall();
    }

    // 4. Ocultar interfaz
    setIsOpen(false);

    // 5. INTENTAR CERRAR LA PESTAÑA
    console.log("[VOICE] Intentando cerrar pestaña...");
    try {
        window.close();
    } catch (e) {
        console.warn("El navegador bloqueó el cierre automático de la pestaña.");
    }

    // 6. FALLBACK (Plan B): Si la pestaña sigue abierta después de 100ms, redirigir
    setTimeout(() => {
        if (!window.closed) {
            console.log("[VOICE] Redirigiendo al inicio porque no se pudo cerrar la pestaña.");
            // Opción A: Redirigir a la home de la app
            window.location.href = '/'; 
            
            // Opción B: Si usas useNavigate de react-router (más suave)
            // navigate('/'); 
            
            // Opción C: Volver a la página anterior
            // window.history.back();
        }
    }, 100);
  }, [
    clearRecordingAnimation,
    stopStreaming,
    stopTTS,
    stopKeepAlive,
    cleanupAudioControls,
    disconnect,
    onEndCall,
  ]);

  // FUNCIÓN AUXILIAR: Parsea markdown inline (**bold**, *italic*, `code`)
  const parseInlineMarkdown = (text) => {
    const parts = [];
    let currentIndex = 0;
    let keyCounter = 0;

    // Regex para detectar **bold**, *italic*, o `code`
    // IMPORTANTE: Debe detectar **bold** ANTES que *italic* para evitar conflictos
    const markdownRegex = /(\*\*.*?\*\*|\*(?!\*).*?\*(?!\*)|`.*?`)/g;
    let match;

    while ((match = markdownRegex.exec(text)) !== null) {
      const matchText = match[0];
      const matchIndex = match.index;

      // Añadir texto antes del match
      if (matchIndex > currentIndex) {
        parts.push(
          <span key={`text-${keyCounter++}`}>
            {text.substring(currentIndex, matchIndex)}
          </span>
        );
      }

      // Procesar el match según su tipo
      if (matchText.startsWith("**") && matchText.endsWith("**")) {
        const boldText = matchText.slice(2, -2);
        parts.push(
          <strong
            key={`bold-${keyCounter++}`}
            style={{ fontWeight: 700, color: "#1e40af" }}
          >
            {boldText}
          </strong>
        );
      } else if (
        matchText.startsWith("*") &&
        matchText.endsWith("*") &&
        matchText.length > 2
      ) {
        // *Italic* (solo si no es ** bold)
        const italicText = matchText.slice(1, -1);
        parts.push(
          <em key={`italic-${keyCounter++}`} style={{ fontStyle: "italic" }}>
            {italicText}
          </em>
        );
      } else if (matchText.startsWith("`") && matchText.endsWith("`")) {
        // `Code`
        const codeText = matchText.slice(1, -1);
        parts.push(
          <code
            key={`code-${keyCounter++}`}
            style={{
              backgroundColor: "#f3f4f6",
              padding: "2px 6px",
              borderRadius: "4px",
              fontFamily: "monospace",
              fontSize: "0.9em",
              color: "#dc2626",
            }}
          >
            {codeText}
          </code>
        );
      }
      currentIndex = matchIndex + matchText.length;
    }

    // Añadir texto restante después del último match
    if (currentIndex < text.length) {
      parts.push(
        <span key={`text-${keyCounter++}`}>{text.substring(currentIndex)}</span>
      );
    }

    return parts.length > 0 ? parts : text;
  };

  // FUNCIÓN MEJORADA: Formatea texto de COSMOS con soporte Markdown
  const formatCosmosText = (text) => {
    const lines = text.split("\n");

    return lines.map((line, lineIndex) => {
      const trimmedLine = line.trim();

      // Líneas vacías - mantener salto de línea
      if (!trimmedLine) return <br key={`br-${lineIndex}`} />;

      // TÍTULOS: Líneas que terminan en ':' (en negrita y azul)
      if (trimmedLine.endsWith(":")) {
        return (
          <div
            key={`title-${lineIndex}`}
            style={{
              fontWeight: "bold",
              color: "#2563eb",
              marginBottom: "0.5rem",
            }}
          >
            {parseInlineMarkdown(trimmedLine)}
          </div>
        );
      }

      // LISTAS CON BULLETS: Líneas que empiezan con '-' o '•'
      if (trimmedLine.startsWith("- ") || trimmedLine.startsWith("• ")) {
        return (
          <div
            key={`bullet-${lineIndex}`}
            style={{ marginLeft: "1rem", marginBottom: "0.25rem" }}
          >
            • {parseInlineMarkdown(trimmedLine.substring(2))}
          </div>
        );
      }

      // LISTAS NUMERADAS: Detecta patrones como '1. ', '2. ', etc.
      const numberedMatch = trimmedLine.match(/^(\d+)\.\s+(.+)$/);
      if (numberedMatch) {
        return (
          <div
            key={`numbered-${lineIndex}`}
            style={{ marginLeft: "1rem", marginBottom: "0.25rem" }}
          >
            {numberedMatch[1]}. {parseInlineMarkdown(numberedMatch[2])}
          </div>
        );
      }

      // TEXTO NORMAL (con soporte para markdown inline)
      return (
        <div key={`text-${lineIndex}`} style={{ marginBottom: "0.25rem" }}>
          {parseInlineMarkdown(trimmedLine)}
        </div>
      );
    });
  };

  // Función para capitalizar la primera letra
  const capitalizeFirstLetter = (text) => {
    if (!text) return text;
    return text.charAt(0).toUpperCase() + text.slice(1);
  };


  if (!isOpen) return null;

  return (
    <div
      className={`flex-1 flex flex-col h-full
        ${
          isDarkMode
            ? "bg-gray-900 border-gray-700 shadow-blue-900/20"
            : "bg-white border-gray-200 shadow-blue-200/40"
        }`}
      data-testid="voice-agent-page"
    >
      {/* Panel principal */}
      <div className="flex-1 flex flex-col relative h-full" data-testid="panel-conversation">
        
        {/* Botones de accion sobre el chat */}
        <div className="absolute top-4 right-6 flex gap-2 z-20">
          <Button
            onClick={handleClearConversation}
            variant="ghost"
            className={cn(
              "flex items-center gap-2 rounded-xl text-sm font-medium transition-all duration-200 backdrop-blur-sm border",
              isDarkMode
                ? "bg-red-500/10 border-red-500/20 text-red-300 hover:bg-red-500/20 hover:border-red-500/40 hover:text-red-200 shadow-sm"
                : "bg-red-500/10 border-red-400/30 text-red-600 hover:bg-red-500/20 hover:border-red-500/40 hover:text-red-700 shadow-sm"
            )}
            data-testid="button-clear-conversation"
          >
            <FiTrash2 className="w-3 h-3 md:w-4 md:h-4" />
            <span className="hidden sm:inline">Limpiar</span>
          </Button>

          <Button
            onClick={handleExportConversation}
            variant="ghost"
            className={cn(
              "flex items-center gap-2 rounded-xl text-sm font-medium transition-all duration-200 backdrop-blur-sm border",
              isDarkMode
                ? "bg-blue-500/10 border-blue-500/20 text-blue-300 hover:bg-blue-500/20 hover:border-blue-500/40 hover:text-blue-200 shadow-sm"
                : "bg-blue-500/10 border-blue-400/30 text-blue-600 hover:bg-blue-500/20 hover:border-blue-500/40 hover:text-blue-700 shadow-sm"
            )}
            data-testid="button-export-conversation"
          >
            <FiDownload className="w-3 h-3 md:w-4 md:h-4" />
            <span className="hidden sm:inline">Exportar</span>
          </Button>
        </div>

        {/* Contenedor de mensajes */}
        <div
          className={`flex-1 overflow-y-auto min-h-0 px-6 py-10 space-y-2 sm:space-y-3 md:space-y-4 hide-scrollbar`}
          style={{ paddingTop: "4.5rem" }}
          data-testid="container-messages"
        >
          {/* Mensajes */}
          {voiceMessages.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <FaRobot
                  className={`w-10 h-10 mx-auto mb-3 ${
                    isDarkMode ? "text-gray-500" : "text-gray-400"
                  }`}
                />
                <p className={`text-sm ${isDarkMode ? "text-gray-400" : "text-gray-500"}`}>
                  Comienza a hablar para iniciar tu conversación con COSMOS
                </p>
              </div>
            </div>
          ) : (
            voiceMessages.map((message) => (
              <div
                key={message.id}
                className={`flex ${
                  message.type === "user" || message.type === "user_progressive"
                    ? "justify-end"
                    : "justify-start items-start space-x-2 sm:space-x-3"
                }`}
              >
                {message.type === "agent" && (
                  <div className="w-6 h-6 sm:w-7 sm:h-7 md:w-8 md:h-8 rounded-lg flex items-center justify-center flex-shrink-0 mt-1 overflow-hidden">
                    <img
                      src={cosmosIconGradient}
                      alt="COSMOS"
                      className="w-full h-full object-cover"
                    />
                  </div>
                )}

                <div
                  className={`voice-message-bubble max-w-[95%] sm:max-w-[90%] md:max-w-[85%] lg:max-w-[70%]
                    p-2 sm:p-2.5 md:p-3 lg:p-4 rounded-2xl shadow-sm
                    text-[11px] sm:text-[12px] md:text-[11.5px] lg:text-[13px] xl:text-[15px] leading-snug
                    ${
                      message.type === "user" || message.type === "user_progressive"
                        ? isDarkMode
                          ? "bg-green-700 text-green-100 rounded-br-md border border-white"
                          : "bg-green-100 text-black rounded-br-md"
                        : isDarkMode
                        ? "bg-gray-500 border border-white-600 text-white self-start rounded-bl-md"
                        : "bg-blue-100 text-black rounded-bl-md"
                    }`}
                  data-testid={`message-${message.type}-${message.id}`}
                >
                  <div
                    style={{
                      whiteSpace:
                        message.type === "user" || message.type === "user_progressive"
                          ? "nowrap"
                          : "pre-wrap",
                      overflow: message.type === "user_progressive" ? "hidden" : "visible",
                      textOverflow: message.type === "user_progressive" ? "clip" : "unset"
                    }}
                  >
                    {message.type === "agent" ? (
                      formatCosmosText(message.content)
                    ) : message.type === "user_progressive" && message.progressiveWords ? (
                      /* Renderizado de palabra por palabra del user según habla y con efecto de gris a negro */
                      <span className="inline-flex items-center">
                        {message.progressiveWords.map((word) => (
                          <span
                            key={`${message.id}-${word.timestamp}`}
                            className={`whitespace-pre mr-1 transition-colors duration-150 ${
                              word.isNew
                                ? isDarkMode
                                  ? "text-green-300 opacity-50"
                                  : "text-gray-400"
                                : "inherit"
                            }`}
                          >
                            {word.word}
                          </span>
                        ))}
                        {/* Barra que parpadea según se escribe el mensaje para que parezca que se está escribiendo a mano */}
                        <span className={`inline-block w-1 h-3 animate-pulse ${isDarkMode ? "bg-green-200" : "bg-blue-400"}`} />
                      </span>
                    ) : (
                      capitalizeFirstLetter(message.content)
                    )}
                  </div>

                  {/* Timestamp */}
                  <div
                    className={`text-[10px] sm:text-[11px] mt-1 flex items-center ${
                      message.type === "user" || message.type === "user_progressive"
                        ? "justify-end space-x-1 opacity-75"
                        : "justify-start"
                    } ${
                      message.type === "agent"
                        ? isDarkMode
                          ? "text-gray-400"
                          : "text-gray-500"
                        : isDarkMode
                        ? "text-gray-300"
                        : "text-gray-600"
                    }`}
                  >
                    {message.type === "user" && (
                      <>
                        <FiMic className="text-[9px]" />
                        <span>{message.timestamp}</span>
                      </>
                    )}
                    {message.type === "agent" &&
                      message.isPlaying &&
                      ttsProgress > 0 &&
                      ttsProgress < 100 && (
                        <div className="flex items-center space-x-2">
                          <FiVolume2 className="text-[9px]" />
                          <span>TTS Reproduciendo</span>
                          <div className="w-10 bg-gray-300 rounded-full h-1">
                            <div
                              className="bg-blue-500 h-1 rounded-full"
                              style={{ width: `${ttsProgress}%` }}
                            />
                          </div>
                        </div>
                      )}
                  </div>
                </div>
              </div>
            ))
          )}

          {/* Indicador de "pensando" */}
          {vadState === "processing" && (
            <div className="flex justify-start items-start space-x-3">
              <div className="w-7 h-7 md:w-8 md:h-8 rounded-lg flex items-center justify-center mt-1 overflow-hidden">
                <img
                  src={cosmosIconGradient}
                  alt="COSMOS"
                  className="w-full h-full object-cover"
                />
              </div>
              <div
                className={`border p-2 sm:p-3 rounded-2xl rounded-bl-md ${
                  isDarkMode
                    ? "bg-gray-800 border-gray-700"
                    : "bg-gray-100 border-gray-300"
                }`}
              >
                <div className="flex items-center space-x-2">
                  <div className="flex space-x-1">
                    <div
                      className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                      style={{ animationDelay: "0ms" }}
                    />
                    <div
                      className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                      style={{ animationDelay: "150ms" }}
                    />
                    <div
                      className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"
                      style={{ animationDelay: "300ms" }}
                    />
                  </div>
                  <span
                    className={`text-[11px] sm:text-xs ${
                      isDarkMode ? "text-gray-400" : "text-gray-600"
                    }`}
                  >
                    COSMOS está pensando...
                  </span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Controles de voz */}
        <VoiceControls
          micMuted={micMuted}
          ttsVolume={ttsVolume}
          toggleMicrophone={toggleMicrophone}
          updateTtsVolume={updateTtsVolume}
          handleEndCall={handleEndCall}
          isDarkMode={isDarkMode}
        />
      </div>
    </div>
  );
}