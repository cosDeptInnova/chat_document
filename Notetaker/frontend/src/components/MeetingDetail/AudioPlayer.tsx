import { useState, useEffect } from 'react';
import { meetingsApi } from '../../services/api';
import { TranscriptionView } from './TranscriptionView';
import type { Transcription } from '../../types';

interface AudioPlayerProps {
  meetingId: string;
  userEmail?: string;
  transcription?: Transcription | null;
}

export const AudioPlayer: React.FC<AudioPlayerProps> = ({ meetingId, userEmail, transcription }) => {
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadAudio = async () => {
      try {
        const url = meetingsApi.getAudioUrl(meetingId, userEmail);
        setAudioUrl(url);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Error al cargar el audio');
      }
    };

    loadAudio();
  }, [meetingId, userEmail]);

  if (error) {
    return (
      <div className="text-center py-12 text-red-600 dark:text-red-400">
        <p>{error}</p>
      </div>
    );
  }

  if (!audioUrl) {
    return (
      <div className="text-center py-12 text-gray-500 dark:text-slate-400">
        <p>Cargando audio...</p>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Sección fija: Reproductor de audio (compacto) */}
      <div className="flex-shrink-0 space-y-1 pb-3">
        <h3 className="text-sm font-semibold text-gray-900 dark:text-slate-50">Reproductor de audio</h3>
        <audio controls preload="metadata" className="w-full h-10" src={audioUrl}>
          Tu navegador no soporta el elemento de audio.
        </audio>
      </div>
      
      {/* Sección con scroll: Transcripción - altura calculada */}
      {transcription && (
        <div className="flex-1 overflow-y-auto border-t border-gray-200 dark:border-slate-700 pt-4" style={{ maxHeight: 'calc(100vh - 380px)' }}>
          <TranscriptionView transcription={transcription} showHeaderInfo={false} />
        </div>
      )}
    </div>
  );
};
