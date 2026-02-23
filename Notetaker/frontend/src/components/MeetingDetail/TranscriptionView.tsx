import { useMemo } from 'react';
import { ArrowDownTrayIcon } from '@heroicons/react/24/outline';
import type { Transcription } from '../../types';
import { processRecallTranscript } from '../../utils/transcriptProcessor';
import { getSpeakerColor } from '../../utils/speakerColors';

interface TranscriptionViewProps {
  transcription: Transcription | null;
  showHeaderInfo?: boolean;
  /** Duracion real reunion en segundos (bot entra -> bot sale). Opcional. */
  totalMeetingDurationSeconds?: number | null;
}

// Función para convertir segundos a formato MM:SS (como en convert.py)
const secondsToMMSS = (seconds: number): string => {
  const minutes = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
};

// Agrupar segmentos consecutivos del mismo speaker (similar a convert.py)
interface GroupedSegment {
  speaker: string;
  start_time: number;
  texts: string[];
  end_time: number;
}

const groupConsecutiveSegments = (
  segments: Array<{ speaker: string; text: string; start_time: number; end_time: number }>
): GroupedSegment[] => {
  if (segments.length === 0) return [];

  const grouped: GroupedSegment[] = [];
  const PAUSE_THRESHOLD = 1.2; // segundos (igual que en convert.py)

  let currentGroup: GroupedSegment | null = null;

  for (const segment of segments) {
    if (!currentGroup) {
      // Primer segmento
      currentGroup = {
        speaker: segment.speaker,
        start_time: segment.start_time,
        texts: [segment.text],
        end_time: segment.end_time,
      };
    } else if (
      currentGroup.speaker === segment.speaker &&
      segment.start_time - currentGroup.end_time < PAUSE_THRESHOLD
    ) {
      // Mismo speaker y pausa corta: agregar al grupo actual
      currentGroup.texts.push(segment.text);
      currentGroup.end_time = segment.end_time;
    } else {
      // Nuevo speaker o pausa larga: guardar grupo anterior y crear uno nuevo
      grouped.push(currentGroup);
      currentGroup = {
        speaker: segment.speaker,
        start_time: segment.start_time,
        texts: [segment.text],
        end_time: segment.end_time,
      };
    }
  }

  // Agregar el último grupo
  if (currentGroup) {
    grouped.push(currentGroup);
  }

  return grouped;
};

export const TranscriptionView: React.FC<TranscriptionViewProps> = ({ transcription, showHeaderInfo = true, totalMeetingDurationSeconds }) => {
  // Procesar el JSON raw de Recall.ai si está disponible
  const processedData = useMemo(() => {
    if (!transcription || !transcription.has_transcription) {
      return null;
    }

    // Si ya viene procesado (compatibilidad hacia atrás)
    if (transcription.conversation && transcription.conversation.length > 0) {
      return {
        conversation: transcription.conversation,
        speakers: transcription.speakers || [],
        total_segments: transcription.total_segments || 0,
        total_duration_seconds: transcription.total_duration_seconds || 0,
        full_text: transcription.full_text || '',
      };
    }

    // Procesar desde JSON raw
    if (transcription.raw_transcript_json) {
      return processRecallTranscript(transcription.raw_transcript_json);
    }

    return null;
  }, [transcription]);

  // Agrupar segmentos consecutivos del mismo speaker
  const groupedSegments = useMemo(() => {
    if (!processedData || !processedData.conversation) {
      return [];
    }
    return groupConsecutiveSegments(processedData.conversation);
  }, [processedData]);

  // Lista unica de participantes a partir de los segmentos mostrados, ordenada alfabeticamente.
  // Asi cada participante tiene un indice estable y un color distinto (evita colisiones).
  const uniqueSpeakerNames = useMemo(() => {
    const names = new Set<string>();
    for (const g of groupedSegments) {
      if (g.speaker?.trim()) names.add(g.speaker.trim());
    }
    return Array.from(names).sort((a, b) => a.localeCompare(b));
  }, [groupedSegments]);

  // Función para generar texto de transcripción para descarga
  const generateTranscriptText = (): string => {
    if (!processedData || groupedSegments.length === 0) {
      return processedData?.full_text || '';
    }

    let text = '';
    
    // Información de la reunión
    text += 'TRANSCRIPCIÓN DE REUNIÓN\n';
    text += '========================\n\n';
    text += `Segmentos: ${processedData.total_segments}\n`;
    text += `Tiempo de habla: ${Math.floor(processedData.total_duration_seconds / 60)} minutos\n`;
    text += `Participantes: ${processedData.speakers.length}\n`;
    if (processedData.speakers.length > 0) {
      text += `Lista de participantes: ${processedData.speakers.map(s => s.name).join(', ')}\n`;
    }
    text += '\n\n';

    // Transcripción
    text += 'TRANSCRIPCIÓN COMPLETA\n';
    text += '======================\n\n';
    
    groupedSegments.forEach((group) => {
      const timeStr = secondsToMMSS(group.start_time);
      text += `[${timeStr}] ${group.speaker}:\n`;
      text += `${group.texts.join(' ')}\n\n`;
    });

    return text;
  };

  // Función para descargar transcripción
  const handleDownload = () => {
    const text = generateTranscriptText();
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `transcripcion-${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  if (!transcription || !transcription.has_transcription || !processedData) {
    return (
      <div className="text-center py-12 text-gray-500 dark:text-slate-400">
        <p>No hay transcripción disponible para esta reunión.</p>
        <p className="text-sm mt-2">La transcripción aparecerá aquí una vez que la reunión haya terminado.</p>
      </div>
    );
  }

  // Si solo se muestra la transcripción (sin header)
  if (!showHeaderInfo) {
    return (
      <div className="flex flex-col h-full overflow-hidden">
        {/* Sección con scroll: solo transcripción */}
        <div className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden space-y-4">
          {groupedSegments.length === 0 ? (
            <div className="text-center py-8 text-gray-500 dark:text-slate-400">
              <p>No hay segmentos de transcripción disponibles.</p>
            </div>
          ) : (
            groupedSegments.map((group, index) => {
              const color = getSpeakerColor(group.speaker, uniqueSpeakerNames);
              
              return (
                <div
                  key={index}
                  className="border-l-4 pl-4 py-3 rounded-r-lg transition-colors bg-white dark:bg-slate-800"
                  style={{
                    borderLeftColor: color.main,
                  }}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center space-x-3">
                      <span
                        className="font-semibold text-lg"
                        style={{ color: color.main }}
                      >
                        {group.speaker}
                      </span>
                      <span className="text-xs text-gray-500 dark:text-slate-400 font-mono">
                        {secondsToMMSS(group.start_time)}
                      </span>
                    </div>
                  </div>
                  <div className="text-gray-800 dark:text-slate-200 leading-relaxed break-words">
                    {group.texts.map((text, textIndex) => (
                      <span key={textIndex}>
                        {text}
                        {textIndex < group.texts.length - 1 && ' '}
                      </span>
                    ))}
                  </div>
                </div>
              );
            })
          )}
          
          {/* Texto completo (opcional, colapsable) */}
          {processedData.full_text && (
            <details className="bg-gray-50 dark:bg-slate-700 rounded-lg p-4 mt-4">
              <summary className="cursor-pointer text-sm font-semibold text-gray-700 dark:text-slate-300 hover:text-gray-900 dark:hover:text-slate-50">
                Ver texto completo sin formato
              </summary>
              <div className="mt-3 text-sm text-gray-600 dark:text-slate-400 whitespace-pre-wrap">
                {processedData.full_text}
              </div>
            </details>
          )}
        </div>
      </div>
    );
  }

  // Modo completo: sección fija arriba + scroll en transcripción
  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Sección fija: Estadísticas y Participantes */}
      <div className="flex-shrink-0 space-y-3 pb-4">
        {/* Estadísticas */}
        <div className={`grid gap-3 ${totalMeetingDurationSeconds != null ? 'grid-cols-4' : 'grid-cols-3'}`}>
          <div className="bg-gray-50 dark:bg-slate-700 rounded-lg p-3">
            <p className="text-xs text-gray-600 dark:text-slate-400">Segmentos</p>
            <p className="text-xl font-bold text-gray-900 dark:text-slate-50">{processedData.total_segments}</p>
          </div>
          <div className="bg-gray-50 dark:bg-slate-700 rounded-lg p-3">
            <p className="text-xs text-gray-600 dark:text-slate-400">Tiempo de habla</p>
            <p className="text-xl font-bold text-gray-900 dark:text-slate-50">
              {Math.floor(processedData.total_duration_seconds / 60)} min
            </p>
          </div>
          {totalMeetingDurationSeconds != null && (
            <div className="bg-gray-50 dark:bg-slate-700 rounded-lg p-3">
              <p className="text-xs text-gray-600 dark:text-slate-400">Duración total reunión</p>
              <p className="text-xl font-bold text-gray-900 dark:text-slate-50">
                {Math.floor(totalMeetingDurationSeconds / 60)} min
              </p>
            </div>
          )}
          <div className="bg-gray-50 dark:bg-slate-700 rounded-lg p-3">
            <p className="text-xs text-gray-600 dark:text-slate-400">Participantes</p>
            <p className="text-xl font-bold text-gray-900 dark:text-slate-50">
              {processedData.speakers?.length ?? uniqueSpeakerNames.length}
            </p>
          </div>
        </div>

        {/* Participantes: siempre visible despues de la info basica si hay segmentos o lista de speakers */}
        {(uniqueSpeakerNames.length > 0 || (processedData.speakers && processedData.speakers.length > 0)) && (
          <div className="bg-white dark:bg-slate-800 rounded-lg p-3 border border-gray-200 dark:border-slate-700">
            <h4 className="text-xs font-semibold text-gray-900 dark:text-slate-50 mb-1.5">Participantes:</h4>
            <div className="flex flex-wrap gap-1.5">
              {(processedData.speakers && processedData.speakers.length > 0
                ? processedData.speakers.map((s) => s.name)
                : uniqueSpeakerNames
              ).map((name, index) => {
                const color = getSpeakerColor(name, uniqueSpeakerNames);
                return (
                  <span
                    key={`${name}-${index}`}
                    className="px-2 py-0.5 rounded-full text-xs font-medium border-2 bg-white dark:bg-slate-800"
                    style={{
                      borderColor: color.main,
                      color: color.main,
                    }}
                  >
                    {name}
                  </span>
                );
              })}
            </div>
          </div>
        )}
      </div>

      {/* Header fijo: Título y botón de descarga */}
      <div className="flex-shrink-0 flex items-center justify-between pb-3 border-b border-gray-200 dark:border-slate-700 mb-3">
        <h3 className="text-base font-semibold text-gray-900 dark:text-slate-50">Transcripción completa</h3>
        <button
          onClick={handleDownload}
          className="inline-flex items-center px-3 py-1.5 text-sm font-medium text-gray-700 dark:text-slate-300 bg-white dark:bg-slate-700 border border-gray-300 dark:border-slate-600 rounded-lg hover:bg-gray-50 dark:hover:bg-slate-600 transition-colors"
          title="Descargar transcripción"
        >
          <ArrowDownTrayIcon className="h-4 w-4 mr-1.5" />
          Descargar
        </button>
      </div>

      {/* Sección con scroll: Transcripción completa */}
      <div className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden">
        <div className="space-y-4">
            {groupedSegments.length === 0 ? (
              <div className="text-center py-8 text-gray-500 dark:text-slate-400">
                <p>No hay segmentos de transcripción disponibles.</p>
              </div>
            ) : (
              groupedSegments.map((group, index) => {
                const color = getSpeakerColor(group.speaker, uniqueSpeakerNames);
                
                return (
                  <div
                    key={index}
                    className="border-l-4 pl-4 py-3 rounded-r-lg transition-colors bg-white dark:bg-slate-800"
                    style={{
                      borderLeftColor: color.main,
                    }}
                  >
                    {/* Header con speaker y timestamp */}
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center space-x-3">
                        <span
                          className="font-semibold text-lg"
                          style={{ color: color.main }}
                        >
                          {group.speaker}
                        </span>
                        <span className="text-xs text-gray-500 dark:text-slate-400 font-mono">
                          {secondsToMMSS(group.start_time)}
                        </span>
                      </div>
                    </div>
                    {/* Texto agrupado */}
                    <div className="text-gray-800 dark:text-slate-200 leading-relaxed break-words">
                      {group.texts.map((text, textIndex) => (
                        <span key={textIndex}>
                          {text}
                          {textIndex < group.texts.length - 1 && ' '}
                        </span>
                      ))}
                    </div>
                  </div>
                );
              })
            )}
        </div>

        {/* Texto completo (opcional, colapsable) */}
        {processedData.full_text && (
          <details className="bg-gray-50 dark:bg-slate-700 rounded-lg p-4 mt-4">
            <summary className="cursor-pointer text-sm font-semibold text-gray-700 dark:text-slate-300 hover:text-gray-900 dark:hover:text-slate-50">
              Ver texto completo sin formato
            </summary>
            <div className="mt-3 text-sm text-gray-600 dark:text-slate-400 whitespace-pre-wrap">
              {processedData.full_text}
            </div>
          </details>
        )}
      </div>
    </div>
  );
};

