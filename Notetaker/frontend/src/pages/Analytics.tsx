import { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { analyticsApi } from '../services/api';
import type { UserAnalytics, UserAnalyticsHistory } from '../services/api';
import { 
  LineChart, 
  Line, 
  AreaChart, 
  Area, 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from 'recharts';
import { 
  ChartBarIcon, 
  ClockIcon, 
  UserGroupIcon, 
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
  LightBulbIcon,
  ChatBubbleLeftRightIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  SparklesIcon,
  CalendarDaysIcon,
  UserIcon
} from '@heroicons/react/24/outline';

export const Analytics: React.FC = () => {
  const { user } = useAuth();
  const [analytics, setAnalytics] = useState<UserAnalytics | null>(null);
  const [history, setHistory] = useState<UserAnalyticsHistory | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchAnalytics = async () => {
      if (!user?.id) {
        console.log('Analytics: No user.id available', user);
        return;
      }
      
      try {
        setLoading(true);
        setError(null);
        console.log('Analytics: Fetching data for user.id:', user.id);
        const data = await analyticsApi.getUserAnalytics(user.id);
        console.log('Analytics: Data received:', data);
        setAnalytics(data);
      } catch (err: any) {
        console.error('Error fetching analytics:', err);
        const errorMessage = err?.response?.data?.detail || err?.message || 'No se pudieron cargar las métricas';
        setError(errorMessage);
      } finally {
        setLoading(false);
      }
    };

    fetchAnalytics();
  }, [user?.id]);

  // Cargar histórico
  useEffect(() => {
    const fetchHistory = async () => {
      if (!user?.id) return;
      
      try {
        const data = await analyticsApi.getUserAnalyticsHistory(user.id, 12);
        setHistory(data);
      } catch (err) {
        console.error('Error fetching analytics history:', err);
        // No mostrar error, simplemente no mostrar gráficas si falla
      }
    };

    fetchHistory();
  }, [user?.id]);

  const formatHours = (hours: number) => {
    if (hours < 1) {
      return `${Math.round(hours * 60)} min`;
    }
    return `${hours.toFixed(1)} h`;
  };

  const formatTalkTime = (seconds: number) => {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    if (hours > 0) {
      return `${hours}h ${minutes}m`;
    }
    return `${minutes} min`;
  };

  const getChangeIcon = (change: number) => {
    if (change > 0) {
      return <ArrowTrendingUpIcon className="h-4 w-4 text-green-500" />;
    } else if (change < 0) {
      return <ArrowTrendingDownIcon className="h-4 w-4 text-red-500" />;
    }
    return null;
  };

  const getChangeColor = (change: number) => {
    if (change > 0) return 'text-green-600 dark:text-green-400';
    if (change < 0) return 'text-red-600 dark:text-red-400';
    return 'text-gray-500 dark:text-slate-400';
  };

  const getQualityColor = (value: number, isConflict: boolean = false) => {
    if (isConflict) {
      // Para conflicto, bajo es bueno
      if (value < 30) return 'text-green-600 dark:text-green-400';
      if (value < 50) return 'text-yellow-600 dark:text-yellow-400';
      return 'text-red-600 dark:text-red-400';
    }
    // Para otros, alto es bueno
    if (value >= 70) return 'text-green-600 dark:text-green-400';
    if (value >= 50) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-red-600 dark:text-red-400';
  };

  const getQualityBgColor = (value: number, isConflict: boolean = false) => {
    if (isConflict) {
      if (value < 30) return 'bg-green-50 dark:bg-green-900/30';
      if (value < 50) return 'bg-yellow-50 dark:bg-yellow-900/30';
      return 'bg-red-50 dark:bg-red-900/30';
    }
    if (value >= 70) return 'bg-green-50 dark:bg-green-900/30';
    if (value >= 50) return 'bg-yellow-50 dark:bg-yellow-900/30';
    return 'bg-red-50 dark:bg-red-900/30';
  };

  // Funciones helper para formatear datos de gráficas
  const formatMonthLabel = (year: number, month: number): string => {
    const monthNames = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic'];
    return `${monthNames[month - 1]} ${year}`;
  };

  const prepareChartData = () => {
    if (!history?.monthly_metrics || history.monthly_metrics.length === 0) return [];
    
    return history.monthly_metrics.map(metric => ({
      month: formatMonthLabel(metric.year, metric.month),
      year: metric.year,
      monthNum: metric.month,
      meetings: metric.meetings_count,
      hours: parseFloat(metric.total_hours.toFixed(1)),
      participation: parseFloat(metric.average_participation_percent.toFixed(1)),
      collaboration: parseFloat(metric.average_collaboration.toFixed(1)),
      decisiveness: parseFloat(metric.average_decisiveness.toFixed(1)),
      conflict: parseFloat(metric.average_conflict.toFixed(1)),
      engagement: parseFloat(metric.average_engagement.toFixed(1)),
    }));
  };

  const chartData = prepareChartData();

  if (loading) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-slate-50">Analytics</h1>
          <p className="mt-2 text-gray-600 dark:text-slate-400">Tus métricas personales de reuniones</p>
        </div>
        <div className="flex items-center justify-center min-h-[400px]">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
        </div>
      </div>
    );
  }

  if (error || !analytics) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-slate-50">Analytics</h1>
          <p className="mt-2 text-gray-600 dark:text-slate-400">Tus métricas personales de reuniones</p>
        </div>
        <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-12 text-center transition-colors">
          <ExclamationTriangleIcon className="h-12 w-12 text-gray-400 dark:text-slate-500 mx-auto mb-4" />
          <p className="text-gray-500 dark:text-slate-400">{error || 'No hay datos disponibles'}</p>
        </div>
      </div>
    );
  }

  const hasData = analytics.meetings_this_month > 0 || analytics.meetings_last_month > 0;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-slate-50">Analytics</h1>
        <p className="mt-2 text-gray-600 dark:text-slate-400">Tus métricas personales de reuniones</p>
      </div>

      {!hasData ? (
      <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-12 text-center transition-colors">
          <ChartBarIcon className="h-12 w-12 text-gray-400 dark:text-slate-500 mx-auto mb-4" />
          <p className="text-gray-700 dark:text-slate-300 font-medium mb-2">Aún no tienes reuniones analizadas</p>
          <p className="text-sm text-gray-500 dark:text-slate-400">
            Las métricas aparecerán cuando completes reuniones con transcripción y análisis de IA.
          </p>
        </div>
      ) : (
        <>
          {/* Resumen Mensual - Cards principales */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {/* Reuniones este mes */}
            <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6 transition-colors">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 dark:text-slate-400">Reuniones este mes</p>
                  <p className="text-3xl font-bold text-gray-900 dark:text-slate-50 mt-1">
                    {analytics.meetings_this_month}
                  </p>
                  {analytics.comparison && (
                    <div className={`flex items-center gap-1 mt-1 text-sm ${getChangeColor(analytics.comparison.meetings_change_percent)}`}>
                      {getChangeIcon(analytics.comparison.meetings_change_percent)}
                      <span>{Math.abs(analytics.comparison.meetings_change_percent).toFixed(0)}% vs mismo periodo mes anterior</span>
                    </div>
                  )}
                </div>
                <div className="p-3 bg-primary-100 dark:bg-primary-900/30 rounded-full">
                  <CalendarDaysIcon className="h-6 w-6 text-primary-600 dark:text-primary-400" />
                </div>
              </div>
            </div>

            {/* Horas en reuniones */}
            <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6 transition-colors">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 dark:text-slate-400">Horas en reuniones</p>
                  <p className="text-3xl font-bold text-gray-900 dark:text-slate-50 mt-1">
                    {formatHours(analytics.total_hours_this_month)}
                  </p>
                  {analytics.comparison && (
                    <div className={`flex items-center gap-1 mt-1 text-sm ${getChangeColor(analytics.comparison.hours_change_percent)}`}>
                      {getChangeIcon(analytics.comparison.hours_change_percent)}
                      <span>{Math.abs(analytics.comparison.hours_change_percent).toFixed(0)}% vs mismo periodo mes anterior</span>
                    </div>
                  )}
                </div>
                <div className="p-3 bg-blue-100 dark:bg-blue-900/30 rounded-full">
                  <ClockIcon className="h-6 w-6 text-blue-600 dark:text-blue-400" />
                </div>
              </div>
            </div>

            {/* Participación promedio */}
            <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6 transition-colors">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 dark:text-slate-400">Tu participación promedio</p>
                  <p className="text-3xl font-bold text-gray-900 dark:text-slate-50 mt-1">
                    {analytics.participation?.average_participation_percent.toFixed(0) || 0}%
                  </p>
                  <p className="text-xs text-gray-500 dark:text-slate-400 mt-1">
                    del tiempo de habla
                  </p>
                </div>
                <div className="p-3 bg-purple-100 dark:bg-purple-900/30 rounded-full">
                  <ChatBubbleLeftRightIcon className="h-6 w-6 text-purple-600 dark:text-purple-400" />
                </div>
              </div>
            </div>

            {/* Engagement promedio */}
            <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6 transition-colors">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-600 dark:text-slate-400">Engagement promedio</p>
                  <p className={`text-3xl font-bold mt-1 ${getQualityColor(analytics.quality?.average_engagement || 0)}`}>
                    {analytics.quality?.average_engagement.toFixed(0) || 0}%
                  </p>
                  <p className="text-xs text-gray-500 dark:text-slate-400 mt-1">
                    de tus reuniones
                  </p>
                </div>
                <div className="p-3 bg-green-100 dark:bg-green-900/30 rounded-full">
                  <SparklesIcon className="h-6 w-6 text-green-600 dark:text-green-400" />
                </div>
              </div>
            </div>
          </div>

          {/* Segunda fila: Participación y Calidad */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Tu Participación */}
            {analytics.participation && (
              <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6 transition-colors">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-slate-50 mb-4 flex items-center gap-2">
                  <UserIcon className="h-5 w-5 text-primary-600 dark:text-primary-400" />
                  Tu Participación
                </h2>
                
                <div className="space-y-4">
                  {/* Rol más frecuente */}
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 dark:text-slate-400">Tu rol más frecuente</span>
                    <div className="flex items-center gap-2">
                      {analytics.participation.driver_count >= analytics.participation.contributor_count ? (
                        <span className="px-3 py-1 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-400 rounded-full text-sm font-medium">
                          Driver
                        </span>
                      ) : (
                        <span className="px-3 py-1 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 rounded-full text-sm font-medium">
                          Contributor
                        </span>
                      )}
                      <span className="text-xs text-gray-500 dark:text-slate-400">
                        ({analytics.participation.driver_count}D / {analytics.participation.contributor_count}C)
                      </span>
                    </div>
                  </div>

                  {/* Tiempo total hablando */}
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 dark:text-slate-400">Tiempo total hablando</span>
                    <span className="text-lg font-semibold text-gray-900 dark:text-slate-50">
                      {formatTalkTime(analytics.participation.total_talk_time_seconds)}
                    </span>
                  </div>

                  {/* Responsividad */}
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-gray-600 dark:text-slate-400">Responsividad promedio</span>
                    <div className="flex items-center gap-2">
                      <span className={`text-lg font-semibold ${getQualityColor(analytics.participation.average_responsivity)}`}>
                        {analytics.participation.average_responsivity.toFixed(0)}/100
                      </span>
                    </div>
                  </div>

                  {/* Barra de participación visual */}
                  <div className="mt-4">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs text-gray-500 dark:text-slate-400">Participación</span>
                      <span className="text-xs text-gray-500 dark:text-slate-400">{analytics.participation.average_participation_percent.toFixed(0)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-slate-700 rounded-full h-2">
                      <div
                        className="bg-primary-600 dark:bg-primary-500 h-2 rounded-full transition-all"
                        style={{ width: `${Math.min(analytics.participation.average_participation_percent, 100)}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Calidad de tus Reuniones */}
            {analytics.quality && (
              <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6 transition-colors">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-slate-50 mb-4 flex items-center gap-2">
                  <CheckCircleIcon className="h-5 w-5 text-primary-600 dark:text-primary-400" />
                  Calidad de tus Reuniones
                </h2>
                
                <div className="grid grid-cols-2 gap-4">
                  {/* Colaboración */}
                  <div className={`rounded-lg p-4 ${getQualityBgColor(analytics.quality.average_collaboration)}`}>
                    <p className="text-sm text-gray-600 dark:text-slate-400">Colaboración</p>
                    <p className={`text-2xl font-bold ${getQualityColor(analytics.quality.average_collaboration)}`}>
                      {analytics.quality.average_collaboration.toFixed(0)}
                    </p>
                  </div>

                  {/* Decisividad */}
                  <div className={`rounded-lg p-4 ${getQualityBgColor(analytics.quality.average_decisiveness)}`}>
                    <p className="text-sm text-gray-600 dark:text-slate-400">Decisividad</p>
                    <p className={`text-2xl font-bold ${getQualityColor(analytics.quality.average_decisiveness)}`}>
                      {analytics.quality.average_decisiveness.toFixed(0)}
                    </p>
                  </div>

                  {/* Conflicto */}
                  <div className={`rounded-lg p-4 ${getQualityBgColor(analytics.quality.average_conflict, true)}`}>
                    <p className="text-sm text-gray-600 dark:text-slate-400">Conflicto</p>
                    <p className={`text-2xl font-bold ${getQualityColor(analytics.quality.average_conflict, true)}`}>
                      {analytics.quality.average_conflict.toFixed(0)}%
                    </p>
                  </div>

                  {/* Engagement */}
                  <div className={`rounded-lg p-4 ${getQualityBgColor(analytics.quality.average_engagement)}`}>
                    <p className="text-sm text-gray-600 dark:text-slate-400">Engagement</p>
                    <p className={`text-2xl font-bold ${getQualityColor(analytics.quality.average_engagement)}`}>
                      {analytics.quality.average_engagement.toFixed(0)}%
                    </p>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Tercera fila: Colaboradores y Patrones */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Top Colaboradores */}
            {analytics.top_collaborators.length > 0 && (
              <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6 transition-colors">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-slate-50 mb-4 flex items-center gap-2">
                  <UserGroupIcon className="h-5 w-5 text-primary-600 dark:text-primary-400" />
                  Con quién te reúnes más
                </h2>
                
                <div className="space-y-3">
                  {analytics.top_collaborators.map((collaborator, index) => (
                    <div key={index} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-slate-700 rounded-lg">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 bg-primary-100 dark:bg-primary-900/30 rounded-full flex items-center justify-center">
                          <span className="text-sm font-bold text-primary-600 dark:text-primary-400">
                            {index + 1}
                          </span>
                        </div>
                        <div>
                          <p className="font-medium text-gray-900 dark:text-slate-50">{collaborator.name}</p>
                          <p className="text-xs text-gray-500 dark:text-slate-400">
                            {collaborator.meeting_count} reuniones
                          </p>
                        </div>
                      </div>
                      {collaborator.average_collaboration > 0 && (
                        <div
                          className="text-right cursor-help"
                          title="Promedio del nivel de colaboracion (0-100) en las reuniones que has tenido con esta persona. Valor que asigna la IA al analizar cada reunion."
                        >
                          <p className={`text-sm font-semibold ${getQualityColor(collaborator.average_collaboration)}`}>
                            {collaborator.average_collaboration.toFixed(0)}
                          </p>
                          <p className="text-xs text-gray-500 dark:text-slate-400">collab.</p>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Patrones */}
            {analytics.patterns.length > 0 && (
              <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6 transition-colors">
                <h2 className="text-lg font-semibold text-gray-900 dark:text-slate-50 mb-4 flex items-center gap-2">
                  <ChartBarIcon className="h-5 w-5 text-primary-600 dark:text-primary-400" />
                  Tus Patrones
                </h2>
                
                <div className="space-y-4">
                  {analytics.patterns.map((pattern, index) => (
                    <div key={index} className="border-l-4 border-primary-500 dark:border-primary-400 pl-4 py-2">
                      <div className="flex items-center gap-2 mb-1">
                        {pattern.type === 'best_hour' && <ClockIcon className="h-4 w-4 text-primary-600 dark:text-primary-400" />}
                        {pattern.type === 'busiest_day' && <CalendarDaysIcon className="h-4 w-4 text-primary-600 dark:text-primary-400" />}
                        <span className="font-semibold text-gray-900 dark:text-slate-50">{pattern.value}</span>
                      </div>
                      <p className="text-sm text-gray-600 dark:text-slate-400">{pattern.detail}</p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Sugerencias */}
          {analytics.suggestions.length > 0 && (
            <div className="bg-gradient-to-r from-primary-50 to-blue-50 dark:from-primary-900/20 dark:to-blue-900/20 rounded-lg shadow p-6 transition-colors">
              <h2 className="text-lg font-semibold text-gray-900 dark:text-slate-50 mb-4 flex items-center gap-2">
                <LightBulbIcon className="h-5 w-5 text-yellow-500" />
                Sugerencias para ti
              </h2>
              
              <div className="space-y-3">
                {analytics.suggestions.map((suggestion, index) => (
                  <div key={index} className="flex items-start gap-3 p-3 bg-white/60 dark:bg-slate-800/60 rounded-lg">
                    <SparklesIcon className="h-5 w-5 text-primary-600 dark:text-primary-400 flex-shrink-0 mt-0.5" />
                    <p className="text-gray-700 dark:text-slate-300">{suggestion}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Evolución Temporal - Gráficas */}
          {chartData.length > 0 && (
            <div className="space-y-6 mt-6">
              <h2 className="text-xl font-bold text-gray-900 dark:text-slate-50 flex items-center gap-2">
                <ChartBarIcon className="h-6 w-6 text-primary-600 dark:text-primary-400" />
                Evolución Temporal
              </h2>

              {/* Gráfico 1: Reuniones y Horas por mes */}
              <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6 transition-colors">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-slate-50 mb-4">
                  Reuniones y Horas por Mes
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-slate-600" />
                    <XAxis 
                      dataKey="month" 
                      className="text-xs text-gray-600 dark:text-slate-400"
                      tick={{ fill: 'currentColor' }}
                    />
                    <YAxis 
                      yAxisId="left"
                      className="text-xs text-gray-600 dark:text-slate-400"
                      tick={{ fill: 'currentColor' }}
                    />
                    <YAxis 
                      yAxisId="right" 
                      orientation="right"
                      className="text-xs text-gray-600 dark:text-slate-400"
                      tick={{ fill: 'currentColor' }}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'var(--tw-bg-slate-800)', 
                        border: '1px solid var(--tw-border-slate-600)',
                        borderRadius: '8px',
                        color: 'var(--tw-text-slate-50)'
                      }}
                    />
                    <Legend />
                    <Bar yAxisId="left" dataKey="meetings" fill="#3b82f6" name="Reuniones" />
                    <Bar yAxisId="right" dataKey="hours" fill="#10b981" name="Horas" />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Gráfico 2: Participación promedio */}
              <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6 transition-colors">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-slate-50 mb-4">
                  Participación Promedio
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-slate-600" />
                    <XAxis 
                      dataKey="month" 
                      className="text-xs text-gray-600 dark:text-slate-400"
                      tick={{ fill: 'currentColor' }}
                    />
                    <YAxis 
                      domain={[0, 100]}
                      className="text-xs text-gray-600 dark:text-slate-400"
                      tick={{ fill: 'currentColor' }}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'var(--tw-bg-slate-800)', 
                        border: '1px solid var(--tw-border-slate-600)',
                        borderRadius: '8px',
                        color: 'var(--tw-text-slate-50)'
                      }}
                    />
                    <Legend />
                    <Area 
                      type="monotone" 
                      dataKey="participation" 
                      stroke="#8b5cf6" 
                      fill="#8b5cf6" 
                      fillOpacity={0.3}
                      name="Participación %"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>

              {/* Gráfico 3: Calidad de reuniones */}
              <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6 transition-colors">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-slate-50 mb-4">
                  Calidad de Reuniones
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-slate-600" />
                    <XAxis 
                      dataKey="month" 
                      className="text-xs text-gray-600 dark:text-slate-400"
                      tick={{ fill: 'currentColor' }}
                    />
                    <YAxis 
                      domain={[0, 100]}
                      className="text-xs text-gray-600 dark:text-slate-400"
                      tick={{ fill: 'currentColor' }}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'var(--tw-bg-slate-800)', 
                        border: '1px solid var(--tw-border-slate-600)',
                        borderRadius: '8px',
                        color: 'var(--tw-text-slate-50)'
                      }}
                    />
                    <Legend />
                    <Line 
                      type="monotone" 
                      dataKey="collaboration" 
                      stroke="#3b82f6" 
                      strokeWidth={2}
                      dot={{ r: 4 }}
                      name="Colaboración"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="decisiveness" 
                      stroke="#10b981" 
                      strokeWidth={2}
                      dot={{ r: 4 }}
                      name="Decisividad"
                    />
                    <Line 
                      type="monotone" 
                      dataKey="engagement" 
                      stroke="#f59e0b" 
                      strokeWidth={2}
                      dot={{ r: 4 }}
                      name="Engagement"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>

              {/* Gráfico 4: Engagement acumulado */}
              <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6 transition-colors">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-slate-50 mb-4">
                  Tendencias de Engagement
                </h3>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={chartData}>
                    <defs>
                      <linearGradient id="engagementGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.4}/>
                        <stop offset="95%" stopColor="#f59e0b" stopOpacity={0}/>
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" className="stroke-gray-300 dark:stroke-slate-600" />
                    <XAxis 
                      dataKey="month" 
                      className="text-xs text-gray-600 dark:text-slate-400"
                      tick={{ fill: 'currentColor' }}
                    />
                    <YAxis 
                      domain={[0, 100]}
                      className="text-xs text-gray-600 dark:text-slate-400"
                      tick={{ fill: 'currentColor' }}
                    />
                    <Tooltip 
                      contentStyle={{ 
                        backgroundColor: 'var(--tw-bg-slate-800)', 
                        border: '1px solid var(--tw-border-slate-600)',
                        borderRadius: '8px',
                        color: 'var(--tw-text-slate-50)'
                      }}
                    />
                    <Legend />
                    <Area 
                      type="monotone" 
                      dataKey="engagement" 
                      stroke="#f59e0b" 
                      fillOpacity={1}
                      fill="url(#engagementGradient)"
                      strokeWidth={2}
                      name="Engagement %"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
      </div>
          )}
        </>
      )}
    </div>
  );
};
