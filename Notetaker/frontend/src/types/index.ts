// Tipos compartidos del frontend

export type LicenseLevel = 'basic' | 'advanced' | 'pro';
export type UserRole = 'admin' | 'user';
export type MeetingStatus = 'pending' | 'joining' | 'in_progress' | 'completed' | 'failed' | 'cancelled';

export interface User {
  id: string;
  email: string;
  display_name: string | null;
  is_admin: boolean;
  is_premium?: boolean;
  must_change_password?: boolean; // Indica si el usuario debe cambiar su contraseña
  // Campos calculados en el frontend basados en is_premium
  role?: UserRole;
  license?: LicenseLevel;
  is_online?: boolean; // Indica si el usuario está online (conectado)
  last_heartbeat?: string | null; // Último heartbeat del usuario
  // Estado de webhook de Outlook Calendar
  outlook_webhook_status?: string | null; // Estado del webhook (activo, expirado, no_configurado, etc.)
  outlook_webhook_expired?: boolean; // Indica si el webhook está expirado
}

export interface TranscriptionBasicInfo {
  total_segments: number;
  total_duration_seconds: number | null;
  participants: string[];
  has_transcription: boolean;
}

export interface Meeting {
  id: string;
  meeting_url: string;
  title: string | null;
  scheduled_start_time: string;
  scheduled_end_time: string | null;
  status: MeetingStatus;
  created_at: string;
  organizer_email: string | null;
  organizer_name: string | null;
  transcription_basic_info?: TranscriptionBasicInfo | null;
  /** Duracion real de la reunion en segundos (bot entra -> bot sale). */
  total_meeting_duration_seconds?: number | null;
  /** Motivo del fallo cuando status === 'failed' (VEXA/backend) */
  error_message?: string | null;
  /** Estado/codigo devuelto por VEXA (ej. reason del webhook), para diagnosticar fallos */
  recall_status?: string | null;
  /** True si reunion PENDING de Teams sin ID/passcode (bot no se unira; resincronizar calendario). Solo en listado admin. */
  necesita_resincronizar_calendario?: boolean | null;
}

export interface TranscriptionSegment {
  speaker: string;
  text: string;
  start_time: number;
  end_time: number;
  duration: number | null;
}

export interface Transcription {
  meeting_id: string;
  has_transcription: boolean;
  transcription_id?: string;
  language?: string;
  is_final?: boolean;
  raw_transcript_json?: any; // JSON raw de Recall.ai
  // Campos calculados en el frontend (no vienen del backend)
  total_segments?: number;
  total_duration_seconds?: number;
  speakers?: Array<{
    name: string;
    segments_count: number;
    total_duration: number;
  }>;
  conversation?: TranscriptionSegment[];
  full_text?: string;
}

export interface MeetingAccess {
  id: string;
  meeting_id: string;
  user_id: string;
  user_email: string;
  can_view_transcript: boolean;
  can_view_audio: boolean;
  can_view_video: boolean;
  created_at: string;
}

// Mock data para IA
export interface IASummary {
  summary: string;
  key_points: string[];
  action_items: Array<{
    id: string;
    text: string;
    assignee?: string;
    due_date?: string;
    completed: boolean;
  }>;
}

export interface IAInsights {
  sentiment: 'positive' | 'neutral' | 'negative';
  participation: Array<{
    speaker: string;
    percentage: number;
    speaking_time: number;
  }>;
  topics: Array<{
    topic: string;
    mentions: number;
  }>;
}

export interface IAAnalysis {
  total_duration: number;
  speakers_count: number;
  average_speaking_time: number;
  engagement_score: number;
  decision_points: Array<{
    timestamp: number;
    description: string;
  }>;
}

