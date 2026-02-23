import { useState, useEffect } from 'react';
import { meetingsApi } from '../../services/api';
import { TranscriptionView } from './TranscriptionView';
import type { Transcription } from '../../types';

interface VideoPlayerProps {
  meetingId: string;
  userEmail?: string;
  transcription?: Transcription | null;
}

export const VideoPlayer: React.FC<VideoPlayerProps> = ({ meetingId, userEmail, transcription }) => {
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadVideo = async () => {
      try {
        const url = meetingsApi.getVideoUrl(meetingId, userEmail);
        setVideoUrl(url);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Error al cargar el vídeo');
      }
    };

    loadVideo();
  }, [meetingId, userEmail]);

  if (error) {
    return (
      <div className="text-center py-12 text-red-600 dark:text-red-400">
        <p>{error}</p>
      </div>
    );
  }

  if (!videoUrl) {
    return (
      <div className="text-center py-12 text-gray-500 dark:text-slate-400">
        <p>Cargando vídeo...</p>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col">
      {/* Sección fija: Reproductor de video (compacto) */}
      <div className="flex-shrink-0 space-y-1 pb-3">
        <h3 className="text-sm font-semibold text-gray-900 dark:text-slate-50">Reproductor de vídeo</h3>
        <video controls className="w-full max-h-[200px] rounded-lg bg-black object-contain" src={videoUrl}>
          Tu navegador no soporta el elemento de vídeo.
        </video>
      </div>
      
      {/* Sección con scroll: Transcripción - altura calculada */}
      {transcription && (
        <div className="flex-1 overflow-y-auto border-t border-gray-200 dark:border-slate-700 pt-4" style={{ maxHeight: 'calc(100vh - 480px)' }}>
          <TranscriptionView transcription={transcription} showHeaderInfo={false} />
        </div>
      )}
    </div>
  );
};

