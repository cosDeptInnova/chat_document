import axios from 'axios';
import type { Meeting, Transcription, User, MeetingAccess } from '../types';

// Usar 127.0.0.1 en lugar de localhost para evitar IPv6 (::1) cuando el backend solo escucha en IPv4
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:7000';

// Integrations interfaces - exported early for use in other files
export interface IntegrationStatus {
  google_calendar: {
    connected: boolean;
    connected_at: string | null;
    calendar_id?: string;
    push_notifications_active?: boolean;
  };
  outlook_calendar: {
    connected: boolean;
    connected_at: string | null;
    calendar_id?: string;
    push_notifications_active?: boolean;
  };
}

export interface CalendarEvent {
  provider: string;
  id: string;
  title: string;
  start: any;
  end: any;
  description?: string;
  location?: string;
}

export interface Calendar {
  provider: string;
  id: string;
  name: string;
  primary?: boolean;
}

// Log de la URL base al cargar el módulo
console.log('[API] Configuración:', {
  VITE_API_BASE_URL: import.meta.env.VITE_API_BASE_URL,
  API_BASE_URL_USED: API_BASE_URL,
  allEnv: import.meta.env,
});

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000, // 10 segundos de timeout
});

// Importar función de verificación de token (síncrona)
import { isTokenExpired } from '../utils/tokenUtils';

// Interceptor para añadir token de autenticación si existe
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('auth_token');
  const userEmail = localStorage.getItem('user_email');
  
  // Verificar si el token está realmente expirado antes de usarlo
  if (token) {
    if (isTokenExpired(token)) {
      // Token realmente expirado, limpiar y redirigir
      localStorage.removeItem('auth_token');
      localStorage.removeItem('user_email');
      localStorage.removeItem('user');
      if (typeof window !== 'undefined') {
        window.location.href = '/login';
      }
      // Rechazar la petición
      return Promise.reject(new Error('Token expirado'));
    }
    
    config.headers.Authorization = `Bearer ${token}`;
  }
  
  // Añadir user_email como query param si está disponible (para compatibilidad con endpoints legacy)
  if (userEmail && config.params) {
    config.params.user_email = userEmail;
  } else if (userEmail && !config.params) {
    config.params = { user_email: userEmail };
  }
  
  // Log para debugging
  console.log(`[API] ${config.method?.toUpperCase()} ${config.url}`, {
    baseURL: config.baseURL,
    params: config.params,
    headers: config.headers,
  });
  
  return config;
});

// Interceptor de respuesta para logging de errores y manejo de tokens expirados
api.interceptors.response.use(
  (response) => {
    console.log(`[API] ✅ ${response.config.method?.toUpperCase()} ${response.config.url} - Status: ${response.status}`);
    
    // Verificar token después de cada respuesta exitosa (por si expiró durante la petición)
    const token = localStorage.getItem('auth_token');
    if (token && isTokenExpired(token)) {
      console.warn('[API] Token expirado detectado después de respuesta exitosa');
      localStorage.removeItem('auth_token');
      localStorage.removeItem('user_email');
      localStorage.removeItem('user');
      if (typeof window !== 'undefined') {
        window.location.href = '/login';
      }
    }
    
    return response;
  },
  (error) => {
    console.error(`[API] ❌ ${error.config?.method?.toUpperCase()} ${error.config?.url} - Error:`, {
      message: error.message,
      response: error.response?.data,
      status: error.response?.status,
      baseURL: error.config?.baseURL,
    });
    
    // Si el token está expirado (401), limpiar localStorage y redirigir a login
    // Verificar tanto si el mensaje menciona "Token" como si es cualquier 401 no autorizado
    const isTokenError = error.response?.status === 401 && (
      error.response?.data?.detail?.includes('Token') ||
      error.response?.data?.detail?.includes('expirado') ||
      error.response?.data?.detail?.includes('inválido') ||
      error.response?.data?.detail?.includes('autenticación') ||
      !error.response?.data?.detail // Si es 401 sin detalle, probablemente es token expirado
    );
    
    if (isTokenError) {
      console.warn('[API] Token expirado o inválido detectado en respuesta 401');
      localStorage.removeItem('auth_token');
      localStorage.removeItem('user_email');
      localStorage.removeItem('user');
      // Redirigir a login si estamos en el navegador
      if (typeof window !== 'undefined') {
        window.location.href = '/login';
      }
    }
    
    return Promise.reject(error);
  }
);

// Auth endpoints
export const authApi = {
  login: async (email: string, password: string, displayName?: string, rememberMe: boolean = false): Promise<User> => {
    try {
      const response = await api.post('/api/auth/simple-login', {
        email,
        password,
        display_name: displayName,
        remember_me: rememberMe,
      });
      
      // Guardar token JWT en localStorage
      if (response.data.token) {
        localStorage.setItem('auth_token', response.data.token);
      }
      
      // Guardar email en localStorage para usar en otras peticiones
      localStorage.setItem('user_email', email);
      
      return {
        id: response.data.user_id,
        email: response.data.email,
        display_name: response.data.display_name,
        is_admin: response.data.is_admin,
        must_change_password: response.data.must_change_password || false,
        // Usar license del backend directamente
        role: response.data.is_admin ? 'admin' : 'user',
        license: response.data.license || (response.data.is_premium ? 'advanced' : 'basic'),
      };
    } catch (error: any) {
      // Extraer mensaje de error del backend
      let errorMessage = 'Error al iniciar sesión';
      
      if (error.response?.data?.detail) {
        // El backend devuelve el mensaje en error.response.data.detail
        errorMessage = error.response.data.detail;
      } else if (error.response?.data?.message) {
        errorMessage = error.response.data.message;
      } else if (error.message) {
        errorMessage = error.message;
      }
      
      // Crear un error con el mensaje del backend
      const loginError = new Error(errorMessage);
      (loginError as any).status = error.response?.status;
      throw loginError;
    }
  },
  
  logout: () => {
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user_email');
    localStorage.removeItem('user');
  },
  
  getCurrentUser: async (userEmail: string): Promise<User> => {
    const response = await api.get('/api/auth/me', {
      params: { user_email: userEmail },
    });
    return {
      id: response.data.user_id,
      email: response.data.email,
      display_name: response.data.display_name,
      is_admin: response.data.is_admin,
      is_premium: response.data.is_premium,
      must_change_password: response.data.must_change_password || false,
      role: response.data.is_admin ? 'admin' : 'user',
      license: response.data.license,
    };
  },
  
  forgotPassword: async (email: string): Promise<{ message: string }> => {
    const response = await api.post('/api/auth/forgot-password', { email });
    return response.data;
  },
  
  resetPassword: async (token: string, newPassword: string): Promise<{ message: string }> => {
    const response = await api.post('/api/auth/reset-password', {
      token,
      new_password: newPassword,
    });
    return response.data;
  },
  
  changePassword: async (userEmail: string, currentPassword: string, newPassword: string): Promise<{ message: string }> => {
    const response = await api.post('/api/auth/change-password', {
      user_email: userEmail,
      current_password: currentPassword,
      new_password: newPassword,
    });
    return response.data;
  },

  ssoLogin: async (email: string, displayName?: string, cosmosToken?: string): Promise<User> => {
    const response = await api.post('/api/auth/sso-login', {
      email,
      display_name: displayName || undefined,
      cosmos_token: cosmosToken || undefined,
    });
    const data = response.data;
    if (data.token) {
      localStorage.setItem('auth_token', data.token);
    }
    localStorage.setItem('user_email', data.email);
    const user: User = {
      id: data.user_id,
      email: data.email,
      display_name: data.display_name,
      is_admin: data.is_admin,
      must_change_password: data.must_change_password || false,
      role: data.is_admin ? 'admin' : 'user',
      license: (data.license || (data.is_premium ? 'advanced' : 'basic')) as User['license'],
    };
    localStorage.setItem('user', JSON.stringify(user));
    return user;
  },
};

// Meetings endpoints
export const meetingsApi = {
  list: async (userEmail?: string, status?: string, upcomingOnly?: boolean, asAdmin?: boolean): Promise<Meeting[]> => {
    const params: Record<string, string | boolean> = {};
    if (userEmail) params.user_email = userEmail;
    if (status) params.status = status;
    if (upcomingOnly !== undefined) params.upcoming_only = upcomingOnly;
    if (asAdmin !== undefined) params.as_admin = asAdmin;
    
    const response = await api.get('/api/meetings/list', { params });
    return response.data;
  },
  
  get: async (meetingId: string): Promise<Meeting> => {
    const response = await api.get(`/api/meetings/${meetingId}`);
    return response.data;
  },
  
  getTranscription: async (meetingId: string, userEmail?: string): Promise<Transcription> => {
    const params: Record<string, string> = {};
    if (userEmail) params.user_email = userEmail;
    
    const response = await api.get(`/api/meetings/${meetingId}/transcription`, { params });
    return response.data;
  },

  getSummary: async (meetingId: string, userEmail?: string): Promise<any> => {
    const params: Record<string, string> = {};
    if (userEmail) params.user_email = userEmail;
    const response = await api.get(`/api/meetings/${meetingId}/summary`, { params });
    return response.data;
  },

  regenerateSummary: async (meetingId: string, userEmail: string): Promise<any> => {
    const response = await api.post(`/api/meetings/${meetingId}/regenerate-summary`, null, {
      params: { user_email: userEmail },
    });
    return response.data;
  },
  
  getAudioUrl: (meetingId: string, userEmail?: string): string => {
    const params = new URLSearchParams();
    if (userEmail) params.append('user_email', userEmail);
    return `${API_BASE_URL}/api/meetings/${meetingId}/audio?${params.toString()}`;
  },
  
  getVideoUrl: (meetingId: string, userEmail?: string): string => {
    const params = new URLSearchParams();
    if (userEmail) params.append('user_email', userEmail);
    return `${API_BASE_URL}/api/meetings/${meetingId}/video?${params.toString()}`;
  },
  
  getAccess: async (meetingId: string): Promise<MeetingAccess[]> => {
    const response = await api.get(`/api/meetings/${meetingId}/access`);
    return response.data;
  },
  
  grantAccess: async (
    meetingId: string,
    userEmail: string,
    permissions: {
      can_view_transcript: boolean;
      can_view_audio: boolean;
      can_view_video: boolean;
    }
  ): Promise<MeetingAccess> => {
    const response = await api.post(`/api/meetings/${meetingId}/access`, {
      user_email: userEmail,
      ...permissions,
    });
    return response.data;
  },
  
  revokeAccess: async (meetingId: string, userId: string): Promise<void> => {
    await api.delete(`/api/meetings/${meetingId}/access/${userId}`);
  },
  
  create: async (
    meetingUrl: string,
    scheduledStartTime: string,
    scheduledEndTime?: string,
    title?: string,
    userEmail?: string,
    nativeMeetingId?: string,
    passcode?: string
  ): Promise<Meeting> => {
    const body: Record<string, unknown> = {
      meeting_url: meetingUrl,
      scheduled_start_time: scheduledStartTime,
      scheduled_end_time: scheduledEndTime,
      title,
      user_email: userEmail,
    };
    if (nativeMeetingId != null && nativeMeetingId.trim()) {
      body.native_meeting_id = nativeMeetingId.trim();
    }
    if (passcode != null && passcode.trim()) {
      body.passcode = passcode.trim();
    }
    const response = await api.post('/api/meetings/create', body);
    return response.data;
  },
  
  delete: async (meetingId: string, userEmail?: string, deleteContent?: boolean, asAdmin?: boolean): Promise<{ success: boolean; message: string }> => {
    const params: Record<string, string | boolean> = {};
    if (userEmail) params.user_email = userEmail;
    if (deleteContent !== undefined) params.delete_content = deleteContent;
    if (asAdmin !== undefined) params.as_admin = asAdmin;
    
    const response = await api.delete(`/api/meetings/${meetingId}`, { params });
    return response.data;
  },
  
  sendSummaryEmail: async (
    meetingId: string,
    recipients: string[],
    subject: string,
    cc?: string[],
    additionalRecipients?: string[]
  ): Promise<{ success: boolean; message: string }> => {
    const response = await api.post(`/api/meetings/${meetingId}/send-summary-email`, {
      recipients,
      subject,
      cc: cc || [],
      additional_recipients: additionalRecipients || [],
    });
    return response.data;
  },
};

// Users endpoints (admin only)
export const usersApi = {
  list: async (userEmail: string, options?: { skipCache?: boolean }): Promise<User[]> => {
    const params: Record<string, string> = { user_email: userEmail };
    if (options?.skipCache) {
      params._ = String(Date.now());
    }
    const response = await api.get('/api/auth/users/list', { params });
    return response.data;
  },

  /** Sincronizar calendario de un usuario (solo admin). Misma accion que el usuario en Ajustes > Sincronizar. */
  syncUserCalendar: async (adminEmail: string, targetUserEmail: string): Promise<{ message: string; results: any }> => {
    const response = await api.post('/api/auth/admin/sync-user-calendar', {}, {
      params: { admin_email: adminEmail, target_user_email: targetUserEmail },
      timeout: 60000,
    });
    return response.data;
  },

  update: async (
    userId: string,
    userEmail: string,
    updates: {
      display_name?: string;
      email?: string;
      license?: string;
      is_premium?: boolean;
    }
  ): Promise<User> => {
    const response = await api.put(`/api/auth/users/${userId}`, updates, {
      params: { user_email: userEmail },
    });
    return response.data;
  },

  create: async (
    adminEmail: string,
    email: string,
    password: string,
    displayName?: string
  ): Promise<{ message: string; user_id: string; email: string; display_name: string }> => {
    const response = await api.post('/api/auth/admin/create-user', {
      email,
      password,
      display_name: displayName,
    }, {
      params: { admin_email: adminEmail },
    });
    return response.data;
  },

  resetPassword: async (
    adminEmail: string,
    userEmail: string,
    newPassword: string
  ): Promise<{ message: string }> => {
    const response = await api.post('/api/auth/admin/reset-password', {
      user_email: userEmail,
      new_password: newPassword,
    }, {
      params: { admin_email: adminEmail },
    });
    return response.data;
  },

  heartbeat: async (userEmail: string): Promise<{ success: boolean; timestamp: string }> => {
    const response = await api.post('/api/auth/heartbeat', {}, {
      params: { user_email: userEmail },
    });
    return response.data;
  },
};

// Profile endpoints (for authenticated users)
export const profileApi = {
  updateMyProfile: async (
    userEmail: string,
    displayName?: string
  ): Promise<User> => {
    const response = await api.put('/api/auth/users/me', {
      display_name: displayName,
    }, {
      params: { user_email: userEmail },
    });
    return {
      id: response.data.user_id,
      email: response.data.email,
      display_name: response.data.display_name,
      is_admin: response.data.is_admin,
      is_premium: response.data.is_premium,
      must_change_password: response.data.must_change_password || false,
      license: response.data.license,
    };
  },
};

// Integrations endpoints (calendar integrations)
export const integrationsApi = {
  getStatus: async (userEmail: string): Promise<IntegrationStatus> => {
    const response = await api.get('/api/integrations/status', {
      params: { user_email: userEmail },
    });
    return response.data;
  },

  startOAuth: (provider: 'google' | 'outlook', userEmail: string): string => {
    // Retornar URL para redirección (el backend redirige automáticamente)
    return `${API_BASE_URL}/api/integrations/oauth/start/${provider}?user_email=${encodeURIComponent(userEmail)}`;
  },

  disconnect: async (provider: 'google' | 'outlook', userEmail: string): Promise<{ message: string }> => {
    const response = await api.post(`/api/integrations/disconnect/${provider}`, {}, {
      params: { user_email: userEmail },
    });
    return response.data;
  },

  getCalendarEvents: async (
    userEmail: string,
    provider?: 'google' | 'outlook',
    calendarId?: string,
    daysAhead?: number
  ): Promise<{ events: CalendarEvent[] }> => {
    const params: Record<string, string | number> = { user_email: userEmail };
    if (provider) params.provider = provider;
    if (calendarId) params.calendar_id = calendarId;
    if (daysAhead) params.days_ahead = daysAhead;

    const response = await api.get('/api/integrations/calendar/events', { params });
    return response.data;
  },

  getCalendars: async (
    userEmail: string,
    provider?: 'google' | 'outlook'
  ): Promise<{ calendars: Calendar[] }> => {
    const params: Record<string, string> = { user_email: userEmail };
    if (provider) params.provider = provider;

    const response = await api.get('/api/integrations/calendar/calendars', { params });
    return response.data;
  },

  syncCalendars: async (userEmail: string): Promise<{ message: string; results: any }> => {
    const response = await api.post('/api/integrations/calendar/sync', {}, {
      params: { user_email: userEmail },
      timeout: 60000, // 60 s para sync de calendario (puede tardar con muchos eventos/Celery)
    });
    return response.data;
  },

  enablePushNotifications: async (
    userEmail: string,
    provider: 'google' | 'outlook'
  ): Promise<{ message: string; channel_id?: string; expiration?: string }> => {
    const response = await api.post('/api/integrations/calendar/enable-push-notifications', {}, {
      params: { user_email: userEmail, provider },
    });
    return response.data;
  },
};

// Analytics types
export interface ParticipationStats {
  total_talk_time_seconds: number;
  average_participation_percent: number;
  driver_count: number;
  contributor_count: number;
  average_responsivity: number;
}

export interface QualityMetrics {
  average_collaboration: number;
  average_decisiveness: number;
  average_conflict: number;
  average_engagement: number;
}

export interface MonthlyComparison {
  meetings_change_percent: number;
  hours_change_percent: number;
  participation_change_percent: number;
}

export interface TopCollaborator {
  name: string;
  meeting_count: number;
  average_collaboration: number;
}

export interface PatternInsight {
  type: string;
  value: string;
  detail: string;
}

export interface UserAnalytics {
  meetings_this_month: number;
  meetings_last_month: number;
  total_hours_this_month: number;
  total_hours_last_month: number;
  participation: ParticipationStats | null;
  quality: QualityMetrics | null;
  comparison: MonthlyComparison | null;
  top_collaborators: TopCollaborator[];
  patterns: PatternInsight[];
  suggestions: string[];
}

export interface MonthlyMetrics {
  year: number;
  month: number;
  meetings_count: number;
  total_hours: number;
  average_participation_percent: number;
  average_collaboration: number;
  average_decisiveness: number;
  average_conflict: number;
  average_engagement: number;
}

export interface UserAnalyticsHistory {
  monthly_metrics: MonthlyMetrics[];
  total_months: number;
}

// Analytics endpoints
export const analyticsApi = {
  getUserAnalytics: async (userId: string): Promise<UserAnalytics> => {
    const response = await api.get('/api/analytics/user', {
      params: { user_id: userId },
    });
    return response.data;
  },

  getUserAnalyticsHistory: async (userId: string, months: number = 12): Promise<UserAnalyticsHistory> => {
    const response = await api.get('/api/analytics/user/history', {
      params: { user_id: userId, months },
    });
    return response.data;
  },
};

// Security/SSL endpoints
export interface SSLCertificateInfo {
  subject_name: string;
  subject_alternative_names: string[];
  issuer_name: string;
  valid_from: string;
  valid_to: string;
  signature_algorithm: string;
  key_algorithm: string;
  key_size: number | null;
}

export interface SSLCertificateSummary {
  certificate: SSLCertificateInfo;
  chain?: SSLCertificateInfo[];
  imported_by?: string;
  imported_at?: string;
  has_private_key: boolean;
}

export interface SSLImportResponse {
  success: boolean;
  message: string;
  certificate_info: SSLCertificateInfo;
  nginx_instructions?: any;
  note?: string;
}

export const securityApi = {
  getSSLCertificate: async (): Promise<SSLCertificateSummary> => {
    const response = await api.get('/api/security/ssl/certificate');
    return response.data;
  },

  importSSLCertificate: async (
    certificate: File,
    privateKey?: File,
    keystorePassword?: string,
    intermediateMode: 'manual' | 'automatic' = 'manual',
    intermediateCertificates?: File[]
  ): Promise<SSLImportResponse> => {
    const formData = new FormData();
    formData.append('certificate', certificate);
    
    if (privateKey) {
      formData.append('private_key', privateKey);
    }
    
    if (keystorePassword) {
      formData.append('keystore_password', keystorePassword);
    }
    
    formData.append('intermediate_mode', intermediateMode);
    
    if (intermediateCertificates && intermediateMode === 'manual') {
      intermediateCertificates.forEach((cert) => {
        formData.append('intermediate_certificates', cert);
      });
    }
    
    const response = await api.post('/api/security/ssl/import', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  getSSLCertificateChain: async (): Promise<{
    certificate: SSLCertificateInfo;
    intermediate_chain: SSLCertificateInfo[];
    imported_by?: string;
    imported_at?: string;
  }> => {
    const response = await api.get('/api/security/ssl/chain');
    return response.data;
  },
};

export default api;

