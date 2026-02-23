import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip } from 'recharts';
import { ClockIcon, ExclamationTriangleIcon } from '@heroicons/react/24/outline';
import { getSpeakerColor, getSpeakerColorMain } from '../../utils/speakerColors';

interface InsightsTabProps {
  summaryData: any;
  loading: boolean;
}

export const InsightsTab: React.FC<InsightsTabProps> = ({ summaryData, loading }) => {
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
  
  // Preparar datos de participación
  const participationPercent = insights.participation_percent || {};
  const allSpeakers = Object.keys(participationPercent);
  const participationData = Object.entries(participationPercent).map(([speaker, percentage]) => ({
    name: speaker,
    value: percentage as number,
    color: getSpeakerColorMain(speaker, allSpeakers),
  }));

  // Preparar datos de temas (topics)
  const topics = insights.topics || [];
  const topicsData = topics.map((topic: string | { name: string; weight?: number }, index: number) => {
    // Manejar tanto objetos {name, weight} como strings
    if (typeof topic === 'string') {
      return {
        name: topic || `Tema ${index + 1}`,
        importancia: 100, // Valor por defecto como porcentaje
      };
    } else {
      // Convertir weight (0-1) a porcentaje (0-100)
      const weight = topic.weight || 1;
      return {
        name: topic.name || `Tema ${index + 1}`,
        importancia: weight * 100, // Convertir a porcentaje
      };
    }
  });

  // Determinar sentimiento basado en atmosphere (valores de -1 a 1, donde 0 es neutro)
  const atmosphere = insights.atmosphere || {};
  const valence = atmosphere.valence ?? 0; // Default 0 (neutro) si no hay valor
  let sentiment: 'positive' | 'neutral' | 'negative' = 'neutral';
  if (valence > 0.3) sentiment = 'positive';
  else if (valence < -0.3) sentiment = 'negative';

  const sentimentColor =
    sentiment === 'positive'
      ? 'text-green-600 dark:text-green-400 bg-green-100 dark:bg-green-900/30'
      : sentiment === 'negative'
      ? 'text-red-600 dark:text-red-400 bg-red-100 dark:bg-red-900/30'
      : 'text-yellow-600 dark:text-yellow-400 bg-yellow-100 dark:bg-yellow-900/30';

  return (
    <div className="space-y-6">
      {/* Sentimiento */}
      <div>
        <h3 className="text-lg font-semibold text-gray-900 dark:text-slate-50 mb-3">Sentimiento</h3>
        <div className={`inline-flex items-center px-4 py-2 rounded-lg ${sentimentColor}`}>
          <span className="font-semibold capitalize">{sentiment}</span>
        </div>
        {atmosphere.labels && atmosphere.labels.length > 0 && (
          <div className="mt-2 flex flex-wrap gap-2">
            {atmosphere.labels.map((label: string, index: number) => (
              <span
                key={index}
                className="px-2 py-1 text-xs rounded bg-gray-200 dark:bg-slate-700 text-gray-700 dark:text-slate-300"
              >
                {label}
              </span>
            ))}
          </div>
        )}
      </div>

      {/* Participación por hablante */}
      {participationData.length > 0 && (
        <div>
          <h4 className="text-md font-semibold text-gray-900 dark:text-slate-50 mb-3">Participación por hablante</h4>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={participationData}
                cx="50%"
                cy="50%"
                labelLine={false}
                label={({ percent }) => `${percent ? (percent * 100).toFixed(0) : 0}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="value"
              >
                {participationData.map((item, index) => (
                  <Cell key={`cell-${index}`} fill={item.color} />
                ))}
              </Pie>
              <Tooltip 
                formatter={(value: number | undefined) => value !== undefined ? `${value.toFixed(1)}%` : ''}
                content={({ active, payload }: any) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload as { name: string; value: number };
                    return (
                      <div className="bg-white dark:bg-slate-800 p-3 border border-gray-200 dark:border-slate-700 rounded-lg shadow-lg">
                        <p className="font-semibold text-gray-900 dark:text-slate-50">{data.name}</p>
                        <p className="text-primary-600 dark:text-primary-400">{data.value.toFixed(1)}%</p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
            </PieChart>
          </ResponsiveContainer>
          {/* Leyenda con nombres completos */}
          <div className="mt-4 flex flex-wrap gap-3 justify-center">
            {participationData.map((item, index) => {
              const speakerColor = getSpeakerColor(item.name, allSpeakers);
              return (
                <div key={index} className="flex items-center gap-2">
                  <div 
                    className="w-4 h-4 rounded-full" 
                    style={{ backgroundColor: item.color }}
                  />
                  <span 
                    className="text-sm font-medium"
                    style={{ color: speakerColor.main }}
                  >
                    {item.name}
                  </span>
                  <span className="text-sm font-semibold text-gray-900 dark:text-slate-50">{item.value.toFixed(1)}%</span>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Temas más mencionados */}
      {topicsData.length > 0 && (
        <div>
          <h4 className="text-md font-semibold text-gray-900 dark:text-slate-50 mb-3">Temas tratados</h4>
          <ResponsiveContainer width="100%" height={Math.max(200, topicsData.length * 50)}>
            <BarChart data={topicsData} layout="vertical" margin={{ left: 10, right: 20, top: 5, bottom: 5 }}>
              <XAxis 
                type="number" 
                domain={[0, 100]}
                tickFormatter={(value) => `${value.toFixed(0)}%`}
              />
              <YAxis 
                dataKey="name" 
                type="category" 
                width={200}
                tick={{ fontSize: 11, fill: 'currentColor' }}
                interval={0}
              />
              <Tooltip 
                formatter={(value: number | undefined) => value !== undefined ? `${value.toFixed(1)}%` : ''}
                labelFormatter={(label) => label}
                content={({ active, payload }: any) => {
                  if (active && payload && payload.length) {
                    const data = payload[0].payload as { name: string; importancia: number };
                    return (
                      <div className="bg-white dark:bg-slate-800 p-3 border border-gray-200 dark:border-slate-700 rounded-lg shadow-lg">
                        <p className="font-semibold text-gray-900 dark:text-slate-50">{data.name}</p>
                        <p className="text-primary-600 dark:text-primary-400">Importancia: {data.importancia.toFixed(1)}%</p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Bar dataKey="importancia" fill="#3b82f6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Métricas adicionales */}
      <div className="grid grid-cols-2 gap-4">
        {insights.collaboration && (
          <div className="bg-blue-50 dark:bg-blue-900/30 rounded-lg p-4">
            <p 
              className="text-sm text-blue-600 dark:text-blue-400 tooltip-trigger cursor-help" 
              data-tooltip-text="Nivel de colaboración entre participantes (rango: 0-100). Valores altos indican trabajo en equipo efectivo, coordinación y apoyo mutuo durante la reunión."
            >
              Colaboración
            </p>
            <p className="text-2xl font-bold text-blue-900 dark:text-blue-50">
              {insights.collaboration.score_0_100?.toFixed(0) || 'N/A'}
            </p>
          </div>
        )}
        {insights.decisiveness && (
          <div className="bg-green-50 dark:bg-green-900/30 rounded-lg p-4">
            <p 
              className="text-sm text-green-600 dark:text-green-400 tooltip-trigger-right cursor-help" 
              data-tooltip-text="Capacidad de la reunión para tomar decisiones (rango: 0-100). Valores altos indican reuniones más decisivas con resultados claros y acciones concretas."
            >
              Decisividad
            </p>
            <p className="text-2xl font-bold text-green-900 dark:text-green-50">
              {insights.decisiveness.score_0_100?.toFixed(0) || 'N/A'}
            </p>
          </div>
        )}
        {insights.conflict_level_0_100 !== undefined && (
          <div className="bg-red-50 dark:bg-red-900/30 rounded-lg p-4">
            <p 
              className="text-sm text-red-600 dark:text-red-400 tooltip-trigger cursor-help" 
              data-tooltip-text="Nivel de conflicto o tensión detectado durante la reunión (rango: 0-100%). Valores altos indican más desacuerdos, tensiones o discusiones acaloradas."
            >
              Nivel de conflicto
            </p>
            <p className="text-2xl font-bold text-red-900 dark:text-red-50">
              {insights.conflict_level_0_100.toFixed(0)}%
            </p>
          </div>
        )}
      </div>

      {/* Mensaje si no hay datos */}
      {participationData.length === 0 && topicsData.length === 0 && (
        <div className="text-center py-8">
          <p className="text-gray-600 dark:text-slate-400">No hay datos de insights disponibles</p>
        </div>
      )}
    </div>
  );
};
