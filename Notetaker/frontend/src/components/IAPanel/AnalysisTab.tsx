import { ClockIcon, ExclamationTriangleIcon } from '@heroicons/react/24/outline';
import { getSpeakerColor } from '../../utils/speakerColors';

interface AnalysisTabProps {
  summaryData: any;
  loading: boolean;
}

export const AnalysisTab: React.FC<AnalysisTabProps> = ({ summaryData, loading }) => {
  const formatTime = (seconds: number) => {
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}:${String(secs).padStart(2, '0')}`;
  };

  // Estado de carga inicial
  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[200px]">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  // Estados sin datos
  if (!summaryData || summaryData.status === 'not_available') {
    return (
      <div className="text-center py-8">
        <p className="text-gray-600 dark:text-slate-400">{summaryData?.message || 'Esta reunión no tiene análisis disponible aún'}</p>
      </div>
    );
  }

  if (summaryData.status === 'pending') {
    return (
      <div className="text-center py-8">
        <ClockIcon className="h-12 w-12 text-gray-400 dark:text-slate-500 mx-auto mb-4" />
        <p className="text-gray-700 dark:text-slate-300 font-medium mb-2">El análisis está en cola</p>
        <p className="text-sm text-gray-600 dark:text-slate-400">{summaryData.message || 'Se procesará en breve...'}</p>
      </div>
    );
  }

  if (summaryData.status === 'processing') {
    return (
      <div className="text-center py-8">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
        <p className="text-gray-700 dark:text-slate-300 font-medium mb-2">Analizando la reunión</p>
        <p className="text-sm text-gray-600 dark:text-slate-400">
          {summaryData.message || 'Este proceso puede tardar unos minutos, los resultados aparecerán automáticamente.'}
        </p>
        <div className="mt-4 w-full bg-gray-200 dark:bg-slate-700 rounded-full h-2">
          <div className="bg-primary-600 h-2 rounded-full animate-pulse" style={{ width: '60%' }}></div>
        </div>
      </div>
    );
  }

  if (summaryData.status === 'failed') {
    return (
      <div className="text-center py-8">
        <ExclamationTriangleIcon className="h-12 w-12 text-red-500 dark:text-red-400 mx-auto mb-4" />
        <p className="text-gray-700 dark:text-slate-300 font-medium mb-2">Error al procesar</p>
        <p className="text-sm text-gray-600 dark:text-slate-400">
          {summaryData.message || 'No se pudo completar el análisis. Inténtalo de nuevo más tarde.'}
        </p>
      </div>
    );
  }

  // Estado completado - mostrar datos
  const insights = summaryData.insights || {};
  const normalized = summaryData.normalized || [];
  
  // Calcular estadísticas
  const participationPercent = insights.participation_percent || {};
  const talkTimeSeconds = insights.talk_time_seconds || {};
  const speakers = Object.keys(participationPercent);
  const allSpeakers = speakers; // Lista de todos los speakers para asignar colores
  const totalDuration = Math.max(...Object.values(talkTimeSeconds as Record<string, number>)) || 0;
  const averageSpeakingTime = speakers.length > 0 
    ? Object.values(talkTimeSeconds as Record<string, number>).reduce((a, b) => a + b, 0) / speakers.length 
    : 0;
  
  // Calcular engagement score basado en colaboración y decisividad
  const collaborationScore = insights.collaboration?.score_0_100 || 0;
  const decisivenessScore = insights.decisiveness?.score_0_100 || 0;
  const engagementScore = Math.round((collaborationScore + decisivenessScore) / 2);

  // Preparar decisiones
  const decisions = insights.decisions || [];
  const decisionPoints = decisions
    .filter((d: string) => d && d !== 'topic' && d !== 'summary')
    .map((decision: string, index: number) => {
      // Intentar extraer timestamp si existe en normalized
      let timestamp = index * 300; // Timestamp estimado por defecto
      if (normalized.length > 0 && index < normalized.length) {
        timestamp = normalized[index].start || timestamp;
      }
      return {
        timestamp,
        description: decision,
      };
    });

  return (
    <div className="space-y-6">
      {/* Métricas principales */}
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-gray-50 dark:bg-slate-700 rounded-lg p-4">
          <p className="text-sm text-gray-600 dark:text-slate-400">Tiempo máximo de habla</p>
          <p className="text-2xl font-bold text-gray-900 dark:text-slate-50">
            {formatTime(totalDuration)}
          </p>
        </div>
        <div className="bg-gray-50 dark:bg-slate-700 rounded-lg p-4">
          <p className="text-sm text-gray-600 dark:text-slate-400">Participantes</p>
          <p className="text-2xl font-bold text-gray-900 dark:text-slate-50">{speakers.length}</p>
        </div>
        <div className="bg-gray-50 dark:bg-slate-700 rounded-lg p-4">
          <p 
            className="text-sm text-gray-600 dark:text-slate-400 tooltip-trigger cursor-help"
            data-tooltip-text="Tiempo medio de habla por participante. Si alguien habló mucho más que el promedio, puede indicar que dominó la reunión."
          >
            Tiempo promedio
          </p>
          <p className="text-2xl font-bold text-gray-900 dark:text-slate-50">
            {formatTime(averageSpeakingTime)}
          </p>
        </div>
        <div className="bg-gray-50 dark:bg-slate-700 rounded-lg p-4">
          <p 
            className="text-sm text-gray-600 dark:text-slate-400 tooltip-trigger-right cursor-help"
            data-tooltip-text="Promedio de Colaboración y Decisividad. Cuanto más alto, mejor: indica una reunión productiva con buena participación y toma de decisiones."
          >
            Engagement
          </p>
          <p className="text-2xl font-bold text-gray-900 dark:text-slate-50">{engagementScore}%</p>
        </div>
      </div>

      {/* Métricas adicionales */}
      {(insights.collaboration || insights.decisiveness || insights.conflict_level_0_100 !== undefined) && (
        <div className="grid grid-cols-3 gap-4">
          {insights.collaboration && (
            <div className="bg-blue-50 dark:bg-blue-900/30 rounded-lg p-4">
              <p className="text-sm text-blue-600 dark:text-blue-400">Colaboración</p>
              <p className="text-xl font-bold text-blue-900 dark:text-blue-50">
                {insights.collaboration.score_0_100?.toFixed(0) || 'N/A'}
              </p>
            </div>
          )}
          {insights.decisiveness && (
            <div className="bg-green-50 dark:bg-green-900/30 rounded-lg p-4">
              <p className="text-sm text-green-600 dark:text-green-400">Decisividad</p>
              <p className="text-xl font-bold text-green-900 dark:text-green-50">
                {insights.decisiveness.score_0_100?.toFixed(0) || 'N/A'}
              </p>
            </div>
          )}
          {insights.conflict_level_0_100 !== undefined && (
            <div className="bg-red-50 dark:bg-red-900/30 rounded-lg p-4">
              <p className="text-sm text-red-600 dark:text-red-400">Conflicto</p>
              <p className="text-xl font-bold text-red-900 dark:text-red-50">
                {insights.conflict_level_0_100.toFixed(0)}%
              </p>
            </div>
          )}
        </div>
      )}

      {/* Puntos de decisión */}
      {decisionPoints.length > 0 && (
        <div>
          <h4 className="text-md font-semibold text-gray-900 dark:text-slate-50 mb-3">Decisiones y puntos clave</h4>
          <div className="space-y-3">
            {decisionPoints.map((point: { timestamp: number; description: string }, index: number) => (
              <div key={index} className="border-l-4 border-primary-500 dark:border-primary-400 pl-4 py-2">
                <div className="flex items-center justify-between mb-1">
                  <span className="font-semibold text-gray-900 dark:text-slate-50">{point.description}</span>
                  <span className="text-xs text-gray-500 dark:text-slate-400">{formatTime(point.timestamp)}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Participantes y tiempo de habla */}
      {speakers.length > 0 && (
        <div>
          <h4 className="text-md font-semibold text-gray-900 dark:text-slate-50 mb-3">Tiempo de habla por participante</h4>
          <div className="space-y-2">
            {speakers
              .sort((a: string, b: string) => {
                // Ordenar por tiempo de habla de mayor a menor
                const timeA = talkTimeSeconds[a] || 0;
                const timeB = talkTimeSeconds[b] || 0;
                return timeB - timeA;
              })
              .map((speaker: string) => {
                const percentage = participationPercent[speaker] || 0;
                const time = talkTimeSeconds[speaker] || 0;
                const speakerColor = getSpeakerColor(speaker, allSpeakers);
                return (
                  <div key={speaker} className="bg-gray-50 dark:bg-slate-700 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span 
                        className="font-medium"
                        style={{ color: speakerColor.main }}
                      >
                        {speaker}
                      </span>
                      <span className="text-sm text-gray-600 dark:text-slate-400">{percentage.toFixed(1)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-slate-600 rounded-full h-2">
                      <div
                        className="h-2 rounded-full"
                        style={{ width: `${percentage}%`, backgroundColor: speakerColor.main }}
                      ></div>
                    </div>
                    <p className="text-xs text-gray-500 dark:text-slate-400 mt-1">
                      {formatTime(time)} de tiempo de habla
                    </p>
                  </div>
                );
              })}
          </div>
        </div>
      )}

      {/* Mensaje si no hay datos */}
      {speakers.length === 0 && decisionPoints.length === 0 && (
        <div className="text-center py-8">
          <p className="text-gray-600 dark:text-slate-400">No hay datos de análisis disponibles</p>
        </div>
      )}
    </div>
  );
};
