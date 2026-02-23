import { useState, useEffect, useMemo, useCallback } from 'react';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { meetingsApi } from '../services/api';
import { formatLocalDateTime } from '../utils/dateUtils';
import { ArrowLeftIcon, ArrowPathIcon } from '@heroicons/react/24/outline';
import type { Meeting, Transcription, MeetingAccess } from '../types';
import { IAPanel } from '../components/IAPanel/IAPanel';
import { TranscriptionView } from '../components/MeetingDetail/TranscriptionView';
import { AudioPlayer } from '../components/MeetingDetail/AudioPlayer';
import { VideoPlayer } from '../components/MeetingDetail/VideoPlayer';
import { MailingTab } from '../components/MeetingDetail/MailingTab';

export const MeetingDetail: React.FC = () => {
  const { meetingId } = useParams<{ meetingId: string }>();
  const { user } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [meeting, setMeeting] = useState<Meeting | null>(null);
  const [transcription, setTranscription] = useState<Transcription | null>(null);
  const [meetingAccess, setMeetingAccess] = useState<MeetingAccess[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'transcript' | 'audio' | 'video' | 'mailing'>('transcript');
  const [regenerating, setRegenerating] = useState(false);
  const [summaryStatus, setSummaryStatus] = useState<string | null>(null);
  const [hasSummaryData, setHasSummaryData] = useState(false);


  // Obtener la ruta de retorno del estado de navegación, o usar fallback
  const returnPath = (location.state as { returnPath?: string })?.returnPath || '/meetings/past';

  // Determinar si estamos en la vista de administración
  const isAdminView = useMemo(() => {
    return returnPath === '/admin/meetings' || location.pathname.includes('/admin/meetings');
  }, [returnPath, location.pathname]);

  // Verificar si el usuario tiene MeetingAccess para esta reunión
  const userMeetingAccess = useMemo(() => {
    if (!user?.email || !meetingAccess.length) return null;
    return meetingAccess.find(access => access.user_email === user.email) || null;
  }, [user?.email, meetingAccess]);

  // Determinar permisos según MeetingAccess y licencia
  const { canViewTranscript, canViewAudio, canViewVideo, hasAccess } = useMemo(() => {
    // Si no tiene MeetingAccess, no puede ver nada
    if (!userMeetingAccess) {
      return {
        canViewTranscript: false,
        canViewAudio: false,
        canViewVideo: false,
        hasAccess: false,
      };
    }

    // Si tiene MeetingAccess, verificar permisos específicos
    const license = user?.license || 'basic';
    const canAudio = license === 'advanced' || license === 'pro';
    const canVideo = license === 'pro';

    return {
      canViewTranscript: userMeetingAccess.can_view_transcript,
      canViewAudio: userMeetingAccess.can_view_audio && canAudio,
      canViewVideo: userMeetingAccess.can_view_video && canVideo,
      hasAccess: true,
    };
  }, [user?.license, userMeetingAccess]);

  useEffect(() => {
    const fetchData = async () => {
      if (!meetingId) return;

      try {
        setLoading(true);
        setError(null);

        // Cargar siempre usuarios asignados (no solo para admins)
        const promises: Promise<any>[] = [
          meetingsApi.get(meetingId),
          meetingsApi.getAccess(meetingId).catch(() => []),
          meetingsApi.getTranscription(meetingId, user?.email).catch(() => null),
          meetingsApi.getSummary(meetingId, user?.email).catch(() => null)
        ];

        const [meetingData, accessData, transcriptionData, summaryData] = await Promise.all(promises);

        setMeeting(meetingData);
        setTranscription(transcriptionData);
        setMeetingAccess(accessData);

        if (summaryData) {
          setSummaryStatus(summaryData.status);
          const hasData = !!(summaryData.toon || (summaryData.insights && Object.keys(summaryData.insights).length > 0));
          setHasSummaryData(hasData);
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Error al cargar la reunión');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [meetingId, user?.email]);

  // Si el usuario está en una pestaña que no puede ver, redirigir a transcript
  useEffect(() => {
    if (activeTab === 'audio' && !canViewAudio) {
      setActiveTab('transcript');
    } else if (activeTab === 'video' && !canViewVideo) {
      setActiveTab('transcript');
    } else if (activeTab === 'transcript' && !canViewTranscript && !hasAccess) {
      // Si no tiene acceso, mantener en transcript pero mostrar mensaje
      setActiveTab('transcript');
    }
  }, [activeTab, canViewAudio, canViewVideo, canViewTranscript, hasAccess]);

  // Función para cambiar de pestaña con validación de permisos
  const handleTabChange = (tab: 'transcript' | 'audio' | 'video' | 'mailing') => {
    if (tab === 'transcript' && !canViewTranscript && !hasAccess) {
      return; // No permitir cambiar a transcript si no tiene acceso
    }
    if (tab === 'audio' && !canViewAudio) {
      return; // No permitir cambiar a audio si no tiene permiso
    }
    if (tab === 'video' && !canViewVideo) {
      return; // No permitir cambiar a video si no tiene permiso
    }
    setActiveTab(tab);
  };

  // Función para regenerar el resumen
  const handleRegenerateSummary = useCallback(async () => {
    if (!user?.email || !user?.is_admin || !meetingId) {
      return;
    }

    if (!confirm('¿Estás seguro de que quieres regenerar el resumen de esta reunión? El proceso puede tardar varios minutos.')) {
      return;
    }

    setRegenerating(true);
    try {
      await meetingsApi.regenerateSummary(meetingId, user.email);
      // Recargar después de un breve delay
      setTimeout(() => {
        // Forzar recarga del panel de IA si es necesario
        window.location.reload();
      }, 1000);
    } catch (error: any) {
      console.error('Error regenerando resumen:', error);
      alert(error?.response?.data?.detail || 'Error al regenerar el resumen. Solo los administradores pueden realizar esta acción.');
    } finally {
      setRegenerating(false);
    }
  }, [meetingId, user]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  if (error || !meeting) {
    return (
      <div className="bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg p-4 text-red-700 dark:text-red-400">
        {error || 'Reunión no encontrada'}
      </div>
    );
  }

  return (
    <div className="flex flex-col h-[calc(100vh-120px)]">
      {/* Header fijo */}
      <div className="flex-shrink-0 space-y-4 mb-4">
        <button
          onClick={() => navigate(returnPath)}
          className="inline-flex items-center text-sm text-gray-600 dark:text-slate-400 hover:text-gray-900 dark:hover:text-slate-50"
        >
          <ArrowLeftIcon className="h-4 w-4 mr-2" />
          Volver a reuniones
        </button>

        <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-4 transition-colors">
          <h1 className="text-2xl font-bold text-gray-900 dark:text-slate-50 mb-2">
            {meeting.title || 'Reunión sin título'}
          </h1>
          <div className="flex items-center space-x-4 text-sm text-gray-600 dark:text-slate-400">
            <span>
              {formatLocalDateTime(meeting.scheduled_start_time)}
            </span>
            <span
              className={`px-2 py-1 text-xs font-semibold rounded ${meeting.status === 'completed'
                ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-400'
                : 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-400'
                }`}
            >
              {meeting.status}
            </span>
          </div>

          {/* Usuarios asignados a la reunión - Solo mostrar en vista de administración */}
          {isAdminView && user?.is_admin && meetingAccess.length > 0 && (
            <div className="mt-3 pt-3 border-t border-gray-200 dark:border-slate-700">
              <h3 className="text-sm font-semibold text-gray-700 dark:text-slate-300 mb-2">
                Usuarios asignados ({meetingAccess.length})
              </h3>
              <div className="flex flex-wrap gap-2">
                {meetingAccess.map((access) => (
                  <div
                    key={access.id}
                    className="inline-flex items-center px-3 py-1 rounded-full text-sm bg-gray-100 dark:bg-slate-700 text-gray-700 dark:text-slate-300"
                  >
                    <span className="font-medium">{access.user_email}</span>
                    <span className="ml-2 text-xs text-gray-500 dark:text-slate-400">
                      ({access.can_view_transcript ? 'T' : ''}
                      {access.can_view_audio ? 'A' : ''}
                      {access.can_view_video ? 'V' : ''})
                    </span>
                  </div>
                ))}
              </div>
              <p className="text-xs text-gray-500 dark:text-slate-400 mt-2">
                T = Transcripción, A = Audio, V = Vídeo
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Grid que ocupa el espacio restante */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1 min-h-0">
        {/* Contenido principal */}
        <div className="lg:col-span-2 flex flex-col min-h-0">
          {/* Mostrar mensaje de no acceso SIEMPRE cuando no hay acceso, incluso antes de las tabs */}
          {!hasAccess && (
            <div className="bg-yellow-50 dark:bg-yellow-900/30 border border-yellow-200 dark:border-yellow-800 rounded-lg p-6 text-yellow-800 dark:text-yellow-400 mb-6">
              <h3 className="text-lg font-semibold mb-2">No tiene acceso a los datos de la reunión</h3>
              <p>No tienes acceso asignado a esta reunión. Solo puedes ver la información básica y los usuarios asignados.</p>
            </div>
          )}

          {/* Reunion fallida: mostrar motivo del fallo para poder diagnosticar */}
          {meeting?.status === 'failed' && (
            <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6 text-red-800 dark:text-red-200 mb-6">
              <h3 className="text-lg font-semibold mb-2">No hay transcripcion disponible para esta reunion</h3>
              <p className="mb-2">Esta reunion termino en estado fallido. El bot no pudo unirse o grabar.</p>
              {meeting.error_message ? (
                <>
                  <p className="font-medium mt-3 mb-1">Motivo del fallo:</p>
                  <p className="text-sm whitespace-pre-wrap break-words bg-red-100/50 dark:bg-red-900/30 p-3 rounded border border-red-200 dark:border-red-700">
                    {meeting.error_message}
                  </p>
                </>
              ) : (
                <p className="text-sm opacity-90">
                  No se registro un motivo especifico. Posibles causas: nadie invito al bot a la reunion,
                  nadie se presento y el bot salio a los X minutos, problemas de conexion o error del servicio VEXA.
                </p>
              )}
              {meeting.recall_status && (
                <p className="text-xs mt-2 text-red-600 dark:text-red-300">Estado VEXA: {meeting.recall_status}</p>
              )}
            </div>
          )}

          {/* Tabs - Solo mostrar si hay acceso */}
          {hasAccess && (
            <div className="bg-white dark:bg-slate-800 rounded-lg shadow transition-colors flex flex-col h-full">
              <div className="flex-shrink-0 border-b border-gray-200 dark:border-slate-700 relative">
                {/* Botón de regenerar resumen (admins o fallo de generación) */}
                {(user?.is_admin || (summaryStatus === 'failed' && !hasSummaryData)) && (
                  <button
                    onClick={handleRegenerateSummary}
                    disabled={regenerating || loading}
                    className="absolute right-4 top-1/2 -translate-y-1/2 px-3 py-1.5 text-xs font-medium text-gray-700 dark:text-slate-300 bg-gray-100 dark:bg-slate-700 hover:bg-gray-200 dark:hover:bg-slate-600 rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-1.5 z-10"
                    title={user?.is_admin ? "Regenerar resumen de IA" : "Reintentar generación de resumen"}
                  >
                    <ArrowPathIcon className={`h-4 w-4 ${regenerating ? 'animate-spin' : ''}`} />
                    {regenerating ? 'Regenerando...' : (user?.is_admin ? 'Regenerar resumen' : 'Reintentar resumen')}
                  </button>
                )}
                <nav className="flex -mb-px">
                  <button
                    onClick={() => handleTabChange('transcript')}
                    className={`px-6 py-4 text-sm font-medium border-b-2 transition-colors ${activeTab === 'transcript'
                      ? 'border-primary-500 dark:border-primary-400 text-primary-600 dark:text-primary-400'
                      : 'border-transparent text-gray-500 dark:text-slate-400 hover:text-gray-700 dark:hover:text-slate-300 hover:border-gray-300 dark:hover:border-slate-600'
                      }`}
                  >
                    Transcripción
                  </button>
                  {canViewAudio && (
                    <button
                      onClick={() => handleTabChange('audio')}
                      className={`px-6 py-4 text-sm font-medium border-b-2 transition-colors ${activeTab === 'audio'
                        ? 'border-primary-500 dark:border-primary-400 text-primary-600 dark:text-primary-400'
                        : 'border-transparent text-gray-500 dark:text-slate-400 hover:text-gray-700 dark:hover:text-slate-300 hover:border-gray-300 dark:hover:border-slate-600'
                        }`}
                    >
                      Audio
                    </button>
                  )}
                  {canViewVideo && (
                    <button
                      onClick={() => handleTabChange('video')}
                      className={`px-6 py-4 text-sm font-medium border-b-2 transition-colors ${activeTab === 'video'
                        ? 'border-primary-500 dark:border-primary-400 text-primary-600 dark:text-primary-400'
                        : 'border-transparent text-gray-500 dark:text-slate-400 hover:text-gray-700 dark:hover:text-slate-300 hover:border-gray-300 dark:hover:border-slate-600'
                        }`}
                    >
                      Vídeo
                    </button>
                  )}
                  <button
                    onClick={() => handleTabChange('mailing')}
                    className={`px-6 py-4 text-sm font-medium border-b-2 transition-colors ${activeTab === 'mailing'
                      ? 'border-primary-500 dark:border-primary-400 text-primary-600 dark:text-primary-400'
                      : 'border-transparent text-gray-500 dark:text-slate-400 hover:text-gray-700 dark:hover:text-slate-300 hover:border-gray-300 dark:hover:border-slate-600'
                      }`}
                  >
                    Mailing
                  </button>
                </nav>
              </div>

              <div className="flex-1 min-h-0 p-6 overflow-hidden">
                {activeTab === 'transcript' && (
                  <div className="h-full">
                    <TranscriptionView transcription={transcription} totalMeetingDurationSeconds={meeting.total_meeting_duration_seconds} />
                  </div>
                )}
                {activeTab === 'audio' && (
                  canViewAudio ? (
                    <AudioPlayer meetingId={meeting.id} userEmail={user?.email} transcription={transcription} />
                  ) : (
                    <div className="bg-yellow-50 dark:bg-yellow-900/30 border border-yellow-200 dark:border-yellow-800 rounded-lg p-6 text-yellow-800 dark:text-yellow-400">
                      <h3 className="text-lg font-semibold mb-2">Acceso restringido</h3>
                      <p>No tienes permiso para ver el audio de esta reunión.</p>
                    </div>
                  )
                )}
                {activeTab === 'video' && (
                  canViewVideo ? (
                    <VideoPlayer meetingId={meeting.id} userEmail={user?.email} transcription={transcription} />
                  ) : (
                    <div className="bg-yellow-50 dark:bg-yellow-900/30 border border-yellow-200 dark:border-yellow-800 rounded-lg p-6 text-yellow-800 dark:text-yellow-400">
                      <h3 className="text-lg font-semibold mb-2">Acceso restringido</h3>
                      <p>No tienes permiso para ver el video de esta reunión.</p>
                    </div>
                  )
                )}
                {activeTab === 'mailing' && (
                  <div className="h-full overflow-y-auto">
                    <MailingTab meeting={meeting} meetingAccess={meetingAccess} transcription={transcription} />
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Información básica de transcripción - Siempre visible cuando no hay acceso */}
          {!hasAccess && (
            <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6 transition-colors">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-slate-50 mb-4">Información básica</h3>
              {meeting.transcription_basic_info ? (
                <>
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <div>
                      <p className="text-sm text-gray-500 dark:text-slate-400">Segmentos</p>
                      <p className="text-lg font-semibold text-gray-900 dark:text-slate-50">
                        {meeting.transcription_basic_info.total_segments ?? 0}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500 dark:text-slate-400">Tiempo de habla</p>
                      <p className="text-lg font-semibold text-gray-900 dark:text-slate-50">
                        {meeting.transcription_basic_info.total_duration_seconds != null
                          ? `${Math.floor(meeting.transcription_basic_info.total_duration_seconds / 60)} min`
                          : '—'}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500 dark:text-slate-400">Duración total reunión</p>
                      <p className="text-lg font-semibold text-gray-900 dark:text-slate-50">
                        {meeting.total_meeting_duration_seconds != null
                          ? `${Math.floor(meeting.total_meeting_duration_seconds / 60)} min`
                          : '—'}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm text-gray-500 dark:text-slate-400">Participantes</p>
                      <p className="text-lg font-semibold text-gray-900 dark:text-slate-50">
                        {meeting.transcription_basic_info.participants?.length ?? 0}
                      </p>
                    </div>
                  </div>
                  {meeting.transcription_basic_info.participants && meeting.transcription_basic_info.participants.length > 0 && (
                    <div className="mt-4">
                      <p className="text-sm font-semibold text-gray-700 dark:text-slate-300 mb-2">Participantes identificados:</p>
                      <div className="flex flex-wrap gap-2">
                        {meeting.transcription_basic_info.participants.map((participant, idx) => (
                          <span
                            key={idx}
                            className="inline-flex items-center px-3 py-1 rounded-full text-sm bg-gray-100 dark:bg-slate-700 text-gray-700 dark:text-slate-300"
                          >
                            {participant}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              ) : (
                <p className="text-sm text-gray-500 dark:text-slate-400">No hay datos de transcripción disponibles.</p>
              )}
            </div>
          )}
        </div>

        {/* Panel lateral de IA - Solo mostrar si hay acceso */}
        {hasAccess && (
          <div className="lg:col-span-1 flex flex-col h-full">
            <IAPanel meetingId={meeting.id} userEmail={user?.email} />
          </div>
        )}
      </div>
    </div>
  );
};

