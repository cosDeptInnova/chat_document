import { useState, useCallback } from 'react';

export function useVoiceAgent() {
  const [state, setState] = useState({
    isConnected: false,
    sessionId: null,
    vadState: 'listening', // puede ser 'listening' | 'speaking' | 'processing'
    isUserSpeaking: false,
    isAgentSpeaking: false
  });

  const updateVadState = useCallback((newState) => {
    setState(prev => ({ ...prev, vadState: newState }));
  }, []);

  const setUserSpeaking = useCallback((speaking) => {
    setState(prev => ({ ...prev, isUserSpeaking: speaking }));
  }, []);

  const setAgentSpeaking = useCallback((speaking) => {
    setState(prev => ({ ...prev, isAgentSpeaking: speaking }));
  }, []);

  const setConnected = useCallback((connected) => {
    setState(prev => ({ ...prev, isConnected: connected }));
  }, []);

  const setSessionId = useCallback((id) => {
    setState(prev => ({ ...prev, sessionId: id }));
  }, []);

  return {
    ...state,
    updateVadState,
    setUserSpeaking,
    setAgentSpeaking,
    setConnected,
    setSessionId
  };
}