import { useState, useRef, useCallback } from 'react';

export function useAudioControls() {
  // Métricas de buffer
  const totalChunksSentRef = useRef(0);
  const bufferOverrunsRef = useRef(0);

  //Para solucionar el error de VoiceAgent.jsx de getBufferMetric
  const getBufferMetrics = () => ({
    totalChunksSent: totalChunksSentRef.current || 0,
    bufferOverruns: bufferOverrunsRef.current || 0,
    isStreaming: isStreamingRef.current || false,
  });

  const [state, setState] = useState({
    micMuted: false,
    ttsVolume: 0.8,
    audioLevel: -45,
    vadState: 'listening',
    isUserSpeaking: false,
    isAgentSpeaking: false,
    audioDevice: null,
    bufferUsage: 30
  });

  // Audio context and nodes
  const audioContextRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const sourceNodeRef = useRef(null);
  const volumeGainNodeRef = useRef(null);
  const muteGainNodeRef = useRef(null);
  const analyserNodeRef = useRef(null);
  //const mediaRecorderRef = useRef(null);
  const audioRef = useRef(null);

  // Audio level monitoring
  const audioLevelIntervalRef = useRef(null);
  
  // Audio streaming - PCM16 capture
  const isStreamingRef = useRef(false);
  const onAudioDataRef = useRef(null);
  const scriptProcessorRef = useRef(null);
  const pcmBufferRef = useRef(new Int16Array(0));
  const pcmBufferSizeRef = useRef(0);

  //const lastStartAttemptRef = useRef(0);

  const setupAudioControls = useCallback(async () => {
    try {
      console.log('[UseAudio] Solicitando permisos de micrófono...');
      // Get user media
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          sampleRate: 16000,
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });

      console.log('[UseAudio] Permisos concedidos')
      mediaStreamRef.current = stream;

      // Create audio context
      audioContextRef.current = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 16000
      });

      // Resume audio context if suspended
      if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
      }

      // Create audio nodes
      sourceNodeRef.current = audioContextRef.current.createMediaStreamSource(stream);
      volumeGainNodeRef.current = audioContextRef.current.createGain();
      muteGainNodeRef.current = audioContextRef.current.createGain();
      analyserNodeRef.current = audioContextRef.current.createAnalyser();

      // Configue analyser for audio level monitoring
      analyserNodeRef.current.fftSize = 256;
      analyserNodeRef.current.smoothingTimeConstant = 0.8;

      // Conect audio basic notes
      sourceNodeRef.current
        .connect(volumeGainNodeRef.current)
        .connect(muteGainNodeRef.current)
        .connect(analyserNodeRef.current);

      // Set initial volumes
      volumeGainNodeRef.current.gain.setValueAtTime(1, audioContextRef.current.currentTime);
      muteGainNodeRef.current.gain.setValueAtTime(1, audioContextRef.current.currentTime);

      // Setup ScriptProcessorNode for PCM16 16kHz mono capture
      // Using 4096 buffer size for ~256ms chunks at 16kHz (4096/16000 = 0.256s)
      scriptProcessorRef.current = audioContextRef.current.createScriptProcessor(4096, 1, 1);
      
      // Initialize PCM buffer for 250ms chunks at 16kHz mono
      const chunkSize = Math.floor(16000 * 0.25); // 250ms a 16kHz (y este el comentario del codifgo de fer -> 4000 samples for 250ms)
      pcmBufferRef.current = new Int16Array(chunkSize);
      pcmBufferSizeRef.current = 0;

      // Process audio in real-time and convert to PCM16
      scriptProcessorRef.current.onaudioprocess = (event) => {
        if (!isStreamingRef.current || !onAudioDataRef.current) return;

        const inputBuffer = event.inputBuffer.getChannelData(0);
        
        // Convert Float32 to Int16 PCM
        for (let i = 0; i < inputBuffer.length; i++) {
          if (pcmBufferSizeRef.current >= pcmBufferRef.current.length) {

             /*// Buffer is full, send it and reset
            const audioData = new ArrayBuffer(pcmBufferRef.current.length * 2); // 2 bytes per Int16
            const view = new DataView(audioData);
            
            for (let j = 0; j < pcmBufferRef.current.length; j++) {
              view.setInt16(j * 2, pcmBufferRef.current[j], true); // little-endian
            }
            
            onAudioDataRef.current(audioData);
            pcmBufferSizeRef.current = 0;*/    // Codigo que teni fer que he corregido para quitar errores (por si acaso)

            // El buffer está lleno, lo envíamos y reseteamos
            // Contador de chunks enviados
            totalChunksSentRef.current += 1;

            // Detectar buffer overrun (si algo ya estaba en 0)
            if (pcmBufferSizeRef.current > pcmBufferRef.current.length) {
              bufferOverrunsRef.current += 1;
              console.warn('[UseAudio] Buffer overrun detectado');
            }

            // Código existente de envío
            const audioData = new ArrayBuffer(pcmBufferRef.current.length * 2); 
            const view = new DataView(audioData);
            for (let j = 0; j < pcmBufferRef.current.length; j++) {
              view.setInt16(j * 2, pcmBufferRef.current[j], true);
            }
            onAudioDataRef.current(audioData);
            pcmBufferSizeRef.current = 0;
          }
          
          // Convert Float32 (-1.0 to 1.0) to Int16 (-32768 to 32767)
          const sample = Math.max(-1, Math.min(1, inputBuffer[i]));
          pcmBufferRef.current[pcmBufferSizeRef.current++] = Math.round(sample * 32767);
        }
      };

      // Connect ScriptProcessorNode to audio graph for PCM16 capture
      // Must connect to destination to ensure onaudioprocess fires
      muteGainNodeRef.current.connect(scriptProcessorRef.current);
      scriptProcessorRef.current.connect(audioContextRef.current.destination);

      // Check sample rate compatibility with COSMOS backend
      const actualSampleRate = audioContextRef.current.sampleRate;
      if (actualSampleRate !== 16000) {
        console.warn(`[UseAudio] Sample rate es ${actualSampleRate}Hz, pero COSMOS espera 16kHz`);
        console.warn('[UseAudio] El audio puede no funcionar correctamente. Considera implementar resampling.');
        // TODO: Implement resampling from actualSampleRate → 16kHz for full compatibility
        // For now, we continue with the browser's native rate but warn about potential issues
      } else {
        console.log('[UseAudio] AudioContext configurado correctamente a 16kHz');
      }

      // Get audio device info
      const devices = await navigator.mediaDevices.enumerateDevices();
      const audioInputs = devices.filter(device => device.kind === 'audioinput');
      const currentDevice = audioInputs.find(device => 
        stream.getAudioTracks()[0].getSettings().deviceId === device.deviceId
      );
      
      setState(prev => ({
        ...prev,
        audioDevice: currentDevice?.label || 'Default Audio Device'
      }));

      console.log('[UseAudio] Dispositivo:', currentDevice?.label || 'Default');

      // Start audio level monitoring
      startAudioLevelMonitoring();

      // Setup TTS audio element
      audioRef.current = new Audio();
      audioRef.current.volume = state.ttsVolume;

      console.log('[UseAudio] Configuración completa');

    } catch (error) {
      console.error('[UseAudio] Error configurando controles:', error);
      throw error; // Re-throw para que VoiceAgentPage lo maneje
    }
  }, []);

  const startAudioLevelMonitoring = useCallback(() => {
    if (!analyserNodeRef.current) return;

    const dataArray = new Uint8Array(analyserNodeRef.current.frequencyBinCount);
    
    const updateAudioLevel = () => {
      if (!analyserNodeRef.current) return;

      analyserNodeRef.current.getByteFrequencyData(dataArray);
      
      // Calculate RMS (Root Mean Square) for audio level
      let sum = 0;
      for (let i = 0; i < dataArray.length; i++) {
        sum += dataArray[i] * dataArray[i];
      }
      const rms = Math.sqrt(sum / dataArray.length);
      
      // Convert to decibels
      const decibels = rms > 0 ? 20 * Math.log10(rms / 255) : -Infinity;
      
      setState(prev => ({
        ...prev,
        audioLevel: Math.max(-60, Math.min(0, decibels)),
        isUserSpeaking: decibels > -40,
        vadState: decibels > -40 ? 'speaking' : 'listening'
      }));

      // CAMBIO CRÍTICO: Ya NO intentamos enviar start_streaming desde aquí
      // El flujo completo está controlado desde VoiceAgentPage
      // Este VAD solo actualiza el estado visual (isUserSpeaking, vadState)
    };

    audioLevelIntervalRef.current = setInterval(updateAudioLevel, 100);
    console.log('[UseAudio] Monitoreo de nivel iniciado (VAD activo)');
  }, []);

  const cleanupAudioControls = useCallback(() => {
    console.log('[UseAudio] Limpiando controles...');

    // Stop audio level monitoring
    if (audioLevelIntervalRef.current) {
      clearInterval(audioLevelIntervalRef.current);
      audioLevelIntervalRef.current = null;
    }

    // Stop streaming if active
    if (isStreamingRef.current) {
      isStreamingRef.current = false;
      onAudioDataRef.current = null;
    }

    // Disconnect and cleanup ScriptProcessorNode
    if (scriptProcessorRef.current) {
      scriptProcessorRef.current.disconnect();
      scriptProcessorRef.current = null;
    }

    // Close media stream
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(track => track.stop());
      mediaStreamRef.current = null;
    }

    // Close audio context
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
      audioContextRef.current = null;
    }

    // Cleanup refs
    sourceNodeRef.current = null;
    volumeGainNodeRef.current = null;
    muteGainNodeRef.current = null;
    analyserNodeRef.current = null;
    scriptProcessorRef.current = null;
    pcmBufferRef.current = new Int16Array(0);
    pcmBufferSizeRef.current = 0;

    console.log('[UseAudio] Limpieza completada');
  }, []);

  const toggleMicrophone = useCallback(() => {
    const newMutedState = !state.micMuted;

    console.log(`[UseAudio] Micrófono ${newMutedState ? 'silenciado' : 'activado'}`);
    
    // Method 1: Direct track control
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getAudioTracks().forEach(track => {
        track.enabled = !newMutedState;
      });
    }
    
    // Method 2: Gain node control (hardware-independent)
    if (muteGainNodeRef.current && audioContextRef.current) {
      muteGainNodeRef.current.gain.setValueAtTime(
        newMutedState ? 0 : 1,
        audioContextRef.current.currentTime
      );
    }
    
    setState(prev => ({ ...prev, micMuted: newMutedState }));
  }, [state.micMuted]);

  const setTtsVolume = useCallback((volume) => {
    const clampedVolume = Math.max(0, Math.min(1, volume));
    
    if (audioRef.current) {
      audioRef.current.volume = clampedVolume;
      audioRef.current.muted = false; // TTS never muted
    }
    
    setState(prev => ({ ...prev, ttsVolume: clampedVolume }));
  }, []);

  const playTTS = useCallback(async (audioUrl, onProgress) => {
    if (!audioRef.current) return;
    
    try {
      console.log('[UseAudio] Reproduciendo TTS...');
      setState(prev => ({ ...prev, isAgentSpeaking: true }));
      audioRef.current.src = audioUrl;
      audioRef.current.volume = state.ttsVolume;
      
      audioRef.current.ontimeupdate = () => {
        if (audioRef.current && onProgress) {
          const progress = audioRef.current.duration > 0 
            ? (audioRef.current.currentTime / audioRef.current.duration) * 100 
            : 0;
          onProgress(Math.min(progress, 100));
        }
      };
      
      // Event listener para progreso del audio
      audioRef.current.onended = () => {
        console.log('[UseAudio] TTS finalizado');
        setState(prev => ({ ...prev, isAgentSpeaking: false }));
        if (onProgress) onProgress(100); // Completar al 100%
      };
      
      await audioRef.current.play();
    } catch (error) {
      console.error('[UseAudio] Error reproduciendo TTS:', error);
      setState(prev => ({ ...prev, isAgentSpeaking: false }));
    }
  }, [state.ttsVolume]);

  const stopTTS = useCallback(() => {
    if (audioRef.current) {
      console.log('[UseAudio] Deteniendo TTS...');
      // Detener completamente el audio, no solo pausar
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
      audioRef.current.src = ''; // Cancelar la fuente de audio completamente
      audioRef.current.load(); // Forzar la carga del estado vacío
      setState(prev => ({ ...prev, isAgentSpeaking: false }));
    }
  }, []);

  const startStreaming = useCallback((onAudioData) => {
    if (!scriptProcessorRef.current || !audioContextRef.current) {
      console.error('[UseAudio] ScriptProcessorNode no inicializado');
      return false;
    }

    console.log('[UseAudio] Iniciando streaming PCM16...');

    totalChunksSentRef.current = 0;
    bufferOverrunsRef.current = 0;
    onAudioDataRef.current = onAudioData;
    
    try {
      // Reset PCM buffer
      pcmBufferSizeRef.current = 0;

      // Start PCM16 streaming (ScriptProcessorNode auto-processes)
      isStreamingRef.current = true;
      console.log('[UseAudio] Streaming PCM16 activo (16kHz mono)');
      return true;
    } catch (error) {
      console.error('[UseAudio] Error iniciando streaming:', error);
      return false;
    }
  }, []);

  const stopStreaming = useCallback(() => {
    if (isStreamingRef.current) {
      console.log('[UseAudio] Deteniendo streaming...');

      try {
        // Flush any remaining audio data before stopping
        if (pcmBufferSizeRef.current > 0 && onAudioDataRef.current) {
          const audioData = new ArrayBuffer(pcmBufferSizeRef.current * 2); // 2 bytes per Int16
          const view = new DataView(audioData);
          
          for (let j = 0; j < pcmBufferSizeRef.current; j++) {
            view.setInt16(j * 2, pcmBufferRef.current[j], true); // little-endian
          }
          
          onAudioDataRef.current(audioData);
          console.log(`[UseAudio] Flushed final: ${pcmBufferSizeRef.current} PCM16 samples`);
        }
        
        isStreamingRef.current = false;
        onAudioDataRef.current = null;
        pcmBufferSizeRef.current = 0; // Reset buffer
        console.log('[UseAudio] PCM16 Audio streaming detenido');
      } catch (error) {
        console.error('[UseAudio] Error deteniendo streaming PCM16:', error);
      }
    }
  }, []);

  return {
    ...state,
    setupAudioControls,
    cleanupAudioControls,
    toggleMicrophone,
    setTtsVolume,
    playTTS,
    stopTTS,
    startStreaming,
    stopStreaming,
    isStreamingRef,
    isStreaming: isStreamingRef.current,
    audioContextRef, // Exportar para sistema de keep-alive
    getBufferMetrics,
  };
}