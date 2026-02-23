import { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { profileApi, integrationsApi } from '../services/api';
import type { IntegrationStatus } from '../services/api';
import { useSearchParams } from 'react-router-dom';
import { IntegrationCard } from '../components/IntegrationCard';

// Importar iconos SVG
// IMPORTANTE: Coloca tus archivos SVG en frontend/src/assets/ con estos nombres:
import googleCalendarIcon from '../assets/google-calendar-icon.svg';
import outlookCalendarIcon from '../assets/outlook-calendar-icon.svg';

export const Settings: React.FC = () => {
  const { user, refreshUser } = useAuth();
  const [displayName, setDisplayName] = useState(user?.display_name || '');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [searchParams, setSearchParams] = useSearchParams();
  const tabParam = searchParams.get('tab');
  const [activeTab, setActiveTab] = useState<'profile' | 'integrations'>(
    (tabParam === 'profile' || tabParam === 'integrations' ? tabParam : 'profile')
  );
  const [integrationStatus, setIntegrationStatus] = useState<IntegrationStatus | null>(null);
  const [loadingIntegrations, setLoadingIntegrations] = useState(false);

  // Actualizar displayName cuando cambie el usuario
  useEffect(() => {
    if (user?.display_name) {
      setDisplayName(user.display_name);
    }
  }, [user?.display_name]);

  // Cargar estado de integraciones
  useEffect(() => {
    const loadIntegrationStatus = async () => {
      if (!user?.email) return;

      try {
        const status = await integrationsApi.getStatus(user.email);
        setIntegrationStatus(status);
      } catch (err: any) {
        console.error('Error cargando estado de integraciones:', err);
      }
    };

    loadIntegrationStatus();
  }, [user?.email]);

  // Manejar callback de OAuth
  useEffect(() => {
    const connected = searchParams.get('connected');
    const error = searchParams.get('error');

    if (connected) {
      setSuccess(`Integración con ${connected === 'google' ? 'Google Calendar' : 'Outlook Calendar'} conectada exitosamente`);
      // Recargar estado de integraciones
      if (user?.email) {
        integrationsApi.getStatus(user.email).then(setIntegrationStatus);
      }
      // Limpiar parámetros de URL
      setSearchParams({ tab: 'integrations' });
    }

    if (error) {
      // Mapear errores a mensajes amigables
      let errorMessage = '';
      if (error === 'authorization_cancelled') {
        errorMessage = 'El usuario no aceptó los permisos. La conexión fue cancelada.';
      } else if (error === 'no_code') {
        errorMessage = 'No se recibió el código de autorización. Por favor, intenta de nuevo.';
      } else if (error === 'invalid_state') {
        errorMessage = 'Error de seguridad en la conexión. Por favor, intenta de nuevo.';
      } else if (error === 'callback_failed') {
        errorMessage = 'Error al procesar la conexión. Por favor, intenta de nuevo.';
      } else if (error === 'invalid_provider') {
        errorMessage = 'Proveedor de calendario no válido.';
      } else {
        errorMessage = `Error en la conexión: ${error}`;
      }
      setError(errorMessage);
      setSearchParams({ tab: 'integrations' });
    }
  }, [searchParams, user?.email, setSearchParams]);

  // Manejar cambio de pestaña
  const handleTabChange = (tab: 'profile' | 'integrations') => {
    setActiveTab(tab);
    setSearchParams({ tab });
    setError(null);
    setSuccess(null);
  };

  const handleUpdateDisplayName = async () => {
    if (!user?.email) {
      setError('No se pudo obtener la información del usuario');
      return;
    }

    if (displayName === user?.display_name) {
      return; // No hacer nada si no cambió
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      await profileApi.updateMyProfile(user.email, displayName);
      setSuccess('Nombre actualizado exitosamente');
      await refreshUser();
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Error al actualizar nombre');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-slate-50">Configuración</h1>
        <p className="mt-2 text-gray-600 dark:text-slate-400">Ajusta tu perfil y preferencias</p>
      </div>

      {/* Tabs */}
      <div className="bg-white dark:bg-slate-800 rounded-lg shadow transition-colors">
        <div className="border-b border-gray-200 dark:border-slate-700">
          <nav className="flex -mb-px">
            <button
              onClick={() => handleTabChange('profile')}
              className={`px-6 py-4 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'profile'
                  ? 'border-primary-500 dark:border-primary-400 text-primary-600 dark:text-primary-400'
                  : 'border-transparent text-gray-500 dark:text-slate-400 hover:text-gray-700 dark:hover:text-slate-300 hover:border-gray-300 dark:hover:border-slate-600'
              }`}
            >
              Perfil
            </button>
            <button
              onClick={() => handleTabChange('integrations')}
              className={`px-6 py-4 text-sm font-medium border-b-2 transition-colors ${
                activeTab === 'integrations'
                  ? 'border-primary-500 dark:border-primary-400 text-primary-600 dark:text-primary-400'
                  : 'border-transparent text-gray-500 dark:text-slate-400 hover:text-gray-700 dark:hover:text-slate-300 hover:border-gray-300 dark:hover:border-slate-600'
              }`}
            >
              Integraciones
            </button>
          </nav>
        </div>

        <div className="p-6">
          {activeTab === 'profile' && (
            <div className="space-y-6">
              <div>
                <h2 className="text-xl font-semibold text-gray-900 dark:text-slate-50 mb-4">
                  Información del Perfil
                </h2>
                
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
                      Email
                    </label>
                    <input
                      type="email"
                      value={user?.email || ''}
                      disabled
                      className="w-full px-4 py-2 border border-gray-300 dark:border-slate-600 rounded-lg bg-gray-50 dark:bg-slate-700 text-gray-500 dark:text-slate-400 cursor-not-allowed"
                    />
                    <p className="mt-1 text-xs text-gray-500 dark:text-slate-400">
                      El email no se puede cambiar
                    </p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
                      Nombre a mostrar
                    </label>
                    <input
                      type="text"
                      value={displayName}
                      onChange={(e) => setDisplayName(e.target.value)}
                      className="w-full px-4 py-2 border border-gray-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700 text-gray-900 dark:text-slate-50 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                      placeholder="Tu nombre"
                    />
                  </div>

                  {error && (
                    <div className="p-4 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-400">
                      {error}
                    </div>
                  )}

                  {success && (
                    <div className="p-4 bg-green-50 dark:bg-green-900/30 border border-green-200 dark:border-green-800 rounded-lg text-green-700 dark:text-green-400">
                      {success}
                    </div>
                  )}

                  <button
                    onClick={handleUpdateDisplayName}
                    disabled={loading || displayName === user?.display_name}
                    className="px-4 py-2 bg-primary-600 text-white rounded-lg font-semibold hover:bg-primary-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {loading ? 'Guardando...' : 'Guardar cambios'}
                  </button>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'integrations' && (
            <div className="space-y-6">
              <div>
                <h2 className="text-xl font-semibold text-gray-900 dark:text-slate-50 mb-2">
                  Integraciones de Calendario
                </h2>
                <p className="text-sm text-gray-600 dark:text-slate-400 mb-6">
                  Conecta tus calendarios para sincronizar automáticamente tus reuniones.
                  Las reuniones de tu calendario aparecerán automáticamente en Cosmos NoteTaker.
                </p>
              </div>

              {error && (
                <div className="p-4 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-400">
                  {error}
                </div>
              )}

              {success && (
                <div className="p-4 bg-green-50 dark:bg-green-900/30 border border-green-200 dark:border-green-800 rounded-lg text-green-700 dark:text-green-400">
                  {success}
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <IntegrationCard
                  name="Google Calendar"
                  logo={googleCalendarIcon}
                  description="Sincroniza tus eventos de Google Calendar. Las reuniones con enlaces de video se crearán automáticamente en NoteTaker."
                  isConnected={integrationStatus?.google_calendar?.connected || false}
                  connectedAt={integrationStatus?.google_calendar?.connected_at || null}
                  onConnect={() => {
                    if (!user?.email) {
                      setError('No se pudo obtener el email del usuario');
                      return;
                    }
                    const authUrl = integrationsApi.startOAuth('google', user.email);
                    window.location.href = authUrl;
                  }}
                  onDisconnect={async () => {
                    if (!user?.email) {
                      setError('No se pudo obtener el email del usuario');
                      return;
                    }
                    setLoadingIntegrations(true);
                    setError(null);
                    setSuccess(null);
                    try {
                      await integrationsApi.disconnect('google', user.email);
                      setSuccess('Google Calendar desconectado exitosamente');
                      const status = await integrationsApi.getStatus(user.email);
                      setIntegrationStatus(status);
                    } catch (err: any) {
                      setError(err.response?.data?.detail || err.message || 'Error al desconectar Google Calendar');
                    } finally {
                      setLoadingIntegrations(false);
                    }
                  }}
                  loading={loadingIntegrations}
                />

                <IntegrationCard
                  name="Outlook Calendar"
                  logo={outlookCalendarIcon}
                  description="Sincroniza tus eventos de Outlook Calendar. Las reuniones de Teams y otras plataformas se crearán automáticamente."
                  isConnected={integrationStatus?.outlook_calendar?.connected || false}
                  connectedAt={integrationStatus?.outlook_calendar?.connected_at || null}
                  onConnect={() => {
                    if (!user?.email) {
                      setError('No se pudo obtener el email del usuario');
                      return;
                    }
                    const authUrl = integrationsApi.startOAuth('outlook', user.email);
                    window.location.href = authUrl;
                  }}
                  onDisconnect={async () => {
                    if (!user?.email) {
                      setError('No se pudo obtener el email del usuario');
                      return;
                    }
                    setLoadingIntegrations(true);
                    setError(null);
                    setSuccess(null);
                    try {
                      await integrationsApi.disconnect('outlook', user.email);
                      setSuccess('Outlook Calendar desconectado exitosamente');
                      const status = await integrationsApi.getStatus(user.email);
                      setIntegrationStatus(status);
                    } catch (err: any) {
                      setError(err.response?.data?.detail || err.message || 'Error al desconectar Outlook Calendar');
                    } finally {
                      setLoadingIntegrations(false);
                    }
                  }}
                  loading={loadingIntegrations}
                />
              </div>

              {(integrationStatus?.google_calendar?.connected || integrationStatus?.outlook_calendar?.connected) && (
                <>
                  <div className="mt-6 p-4 bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-800 rounded-lg">
                    <div className="flex items-start">
                      <svg className="w-5 h-5 text-blue-600 dark:text-blue-400 mt-0.5 mr-3" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                      </svg>
                      <div className="flex-1">
                        <h3 className="text-sm font-medium text-blue-900 dark:text-blue-200">
                          Sincronización manual
                        </h3>
                        <p className="mt-1 text-sm text-blue-700 dark:text-blue-300">
                          Sincroniza manualmente los eventos de tus calendarios conectados.
                        </p>
                        <button
                          onClick={async () => {
                            if (!user?.email) return;
                            setLoadingIntegrations(true);
                            setError(null);
                            setSuccess(null);
                            try {
                              await integrationsApi.syncCalendars(user.email);
                              setSuccess('Sincronización completada exitosamente');
                            } catch (err: any) {
                              setError(err.response?.data?.detail || err.message || 'Error en la sincronización');
                            } finally {
                              setLoadingIntegrations(false);
                            }
                          }}
                          disabled={loadingIntegrations}
                          className="mt-3 px-4 py-2 text-sm font-medium text-blue-700 dark:text-blue-300 bg-blue-100 dark:bg-blue-900/50 border border-blue-300 dark:border-blue-700 rounded-lg hover:bg-blue-200 dark:hover:bg-blue-900/70 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          {loadingIntegrations ? 'Sincronizando...' : 'Sincronizar ahora'}
                        </button>
                      </div>
                    </div>
                  </div>

                  {integrationStatus?.google_calendar?.connected && !integrationStatus?.google_calendar?.push_notifications_active && (
                  <div className="mt-4 p-4 bg-amber-50 dark:bg-amber-900/30 border border-amber-200 dark:border-amber-800 rounded-lg">
                    <div className="flex items-start">
                      <svg className="w-5 h-5 text-amber-600 dark:text-amber-400 mt-0.5 mr-3" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                      </svg>
                      <div className="flex-1">
                        <h3 className="text-sm font-medium text-amber-900 dark:text-amber-200">
                          Activar sincronización automática
                        </h3>
                        <p className="mt-1 text-sm text-amber-700 dark:text-amber-300">
                          Activa las notificaciones push para que los nuevos eventos se sincronizen automáticamente cuando los crees en Google Calendar.
                        </p>
                        <button
                          onClick={async () => {
                            if (!user?.email) return;
                            setLoadingIntegrations(true);
                            setError(null);
                            setSuccess(null);
                            try {
                              await integrationsApi.enablePushNotifications(user.email, 'google');
                              setSuccess('Sincronización automática activada. Los nuevos eventos se sincronizarán automáticamente.');
                              // Recargar estado
                              const status = await integrationsApi.getStatus(user.email);
                              setIntegrationStatus(status);
                            } catch (err: any) {
                              setError(err.response?.data?.detail || err.message || 'Error activando sincronización automática');
                            } finally {
                              setLoadingIntegrations(false);
                            }
                          }}
                          disabled={loadingIntegrations}
                          className="mt-3 px-4 py-2 text-sm font-medium text-white bg-amber-600 dark:bg-amber-500 border border-amber-700 dark:border-amber-600 rounded-lg hover:bg-amber-700 dark:hover:bg-amber-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
                        >
                          {loadingIntegrations ? 'Activando...' : 'Activar sincronización automática'}
                        </button>
                      </div>
                    </div>
                  </div>
                )}

                {/* Mensaje de sincronización automática activa (Google, Outlook o ambos) */}
                {(integrationStatus?.google_calendar?.push_notifications_active || integrationStatus?.outlook_calendar?.push_notifications_active) && (
                  <div className="mt-4 p-4 bg-green-50 dark:bg-green-900/30 border border-green-200 dark:border-green-800 rounded-lg">
                    <div className="flex items-start">
                      <svg className="w-5 h-5 text-green-600 dark:text-green-400 mt-0.5 mr-3" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                      </svg>
                      <div className="flex-1">
                        <h3 className="text-sm font-medium text-green-900 dark:text-green-200">
                          Sincronización automática activa
                        </h3>
                        <p className="mt-1 text-sm text-green-700 dark:text-green-300">
                          {integrationStatus?.google_calendar?.push_notifications_active && integrationStatus?.outlook_calendar?.push_notifications_active ? (
                            <>Los nuevos eventos que crees en Google Calendar y Outlook se sincronizarán automáticamente en NoteTaker.</>
                          ) : integrationStatus?.google_calendar?.push_notifications_active ? (
                            <>Los nuevos eventos que crees en Google Calendar se sincronizarán automáticamente en NoteTaker.</>
                          ) : (
                            <>Los nuevos eventos que crees en Outlook se sincronizarán automáticamente en NoteTaker.</>
                          )}
                        </p>
                      </div>
                    </div>
                  </div>
                )}
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
