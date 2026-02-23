import { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { useMeetings } from '../hooks/useMeetings';
import { Link } from 'react-router-dom';
import { formatLocalDateTime, isMeetingToday } from '../utils/dateUtils';
import { CalendarIcon, ClockIcon, DocumentTextIcon, MusicalNoteIcon, VideoCameraIcon, TrashIcon } from '@heroicons/react/24/outline';
import { meetingsApi } from '../services/api';
import { ConfirmDialog } from '../components/ConfirmDialog';
import type { Meeting } from '../types';

interface MeetingsListProps {
  type: 'upcoming' | 'past';
}

export const MeetingsList: React.FC<MeetingsListProps> = ({ type }) => {
  const { user } = useAuth();
  // Para próximas: filtrar por fecha futura (upcomingOnly=true) y opcionalmente por status pending
  // Para pasadas: incluir reuniones completadas O reuniones que ya pasaron (aunque estén en pending/failed)
  const upcomingOnly = type === 'upcoming';
  const status = type === 'upcoming' ? 'pending' : 'completed'; // 'completed' en backend ahora incluye pasadas aunque estén pending
  const { meetings, loading, error, refetch } = useMeetings(user?.email, status, upcomingOnly);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteConfirm, setDeleteConfirm] = useState<{ meetingId: string } | null>(null);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700">
        {error}
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-slate-50">
          {type === 'upcoming' ? 'Próximas reuniones' : 'Reuniones pasadas'}
        </h1>
        <p className="mt-2 text-gray-600 dark:text-slate-400">
          {meetings.length} {meetings.length === 1 ? 'reunión' : 'reuniones'}
        </p>
      </div>

      {meetings.length === 0 ? (
        <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-12 text-center transition-colors">
          <CalendarIcon className="mx-auto h-12 w-12 text-gray-400 dark:text-slate-500" />
          <h3 className="mt-4 text-lg font-medium text-gray-900 dark:text-slate-50">
            No hay reuniones {type === 'upcoming' ? 'próximas' : 'pasadas'}
          </h3>
          <p className="mt-2 text-gray-500 dark:text-slate-400">
            {type === 'upcoming'
              ? 'Las reuniones programadas aparecerán aquí'
              : 'Las reuniones completadas aparecerán aquí'}
          </p>
        </div>
      ) : (
        <div className="grid gap-6">
          {meetings.map((meeting) => {
            const isToday = type === 'upcoming' && isMeetingToday(meeting.scheduled_start_time);
            return (
              <MeetingCard 
                key={meeting.id} 
                meeting={meeting} 
                returnPath={type === 'upcoming' ? '/meetings/upcoming' : '/meetings/past'}
                showDelete={true}
                onDelete={() => {
                  setDeleteConfirm({ meetingId: meeting.id });
                }}
                isDeleting={deletingId === meeting.id}
                isToday={isToday}
              />
            );
          })}
        </div>
      )}

      {deleteConfirm && (
        <ConfirmDialog
          isOpen={!!deleteConfirm}
          onClose={() => setDeleteConfirm(null)}
          onConfirm={async () => {
            if (!deleteConfirm) return;
            setDeletingId(deleteConfirm.meetingId);
            try {
              await meetingsApi.delete(deleteConfirm.meetingId, user?.email);
              setDeleteConfirm(null);
              refetch();
            } catch (err: any) {
              alert(err.response?.data?.detail || 'Error al borrar la reunión');
            } finally {
              setDeletingId(null);
            }
          }}
          title="Borrar reunión"
          message="¿Estás seguro de que quieres borrar esta reunión?"
          confirmText="Sí, borrar"
          cancelText="Cancelar"
          confirmButtonColor="red"
          isLoading={deletingId === deleteConfirm.meetingId}
        />
      )}
    </div>
  );
};

const MeetingCard: React.FC<{ 
  meeting: Meeting; 
  returnPath: string;
  showDelete?: boolean;
  onDelete?: () => void;
  isDeleting?: boolean;
  isToday?: boolean;
}> = ({ meeting, returnPath, showDelete = false, onDelete, isDeleting = false, isToday = false }) => {
  const handleDeleteClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (onDelete) {
      onDelete();
    }
  };

  return (
    <div className={`bg-white dark:bg-slate-800 rounded-lg shadow hover:shadow-lg transition-shadow p-6 relative transition-colors ${
      isToday ? 'bg-amber-50 dark:bg-amber-900/20 border-l-4 border-l-amber-500 dark:border-l-amber-400' : ''
    }`}>
      <Link
        to={`/meetings/${meeting.id}`}
        state={{ returnPath }}
        className="block"
      >
        <div className="flex items-start justify-between pr-10">
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-slate-50 mb-2">
              {meeting.title || 'Reunión sin título'}
            </h3>
            <div className="flex items-center text-sm text-gray-500 dark:text-slate-400 space-x-4">
              <div className="flex items-center">
                <ClockIcon className="h-4 w-4 mr-1" />
                {formatLocalDateTime(meeting.scheduled_start_time)}
              </div>
              {(meeting.organizer_name || meeting.organizer_email) && (
                <div className="flex items-center">
                  <span>Organizada por: {meeting.organizer_name || meeting.organizer_email}</span>
                </div>
              )}
            </div>
          </div>
          <div className="flex items-center space-x-2 ml-4">
            <span
              className={`px-3 py-1 text-xs font-semibold rounded-full whitespace-nowrap ${
                meeting.status === 'completed'
                  ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-400'
                  : meeting.status === 'pending'
                  ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-400'
                  : 'bg-gray-100 dark:bg-slate-700 text-gray-800 dark:text-slate-300'
              }`}
            >
              {meeting.status}
            </span>
          </div>
        </div>
        <div className="mt-4 flex items-center space-x-4 text-sm text-gray-500 dark:text-slate-400">
          <div className="flex items-center">
            <DocumentTextIcon className="h-4 w-4 mr-1" />
            <span>Transcripción</span>
          </div>
          <div className="flex items-center">
            <MusicalNoteIcon className="h-4 w-4 mr-1" />
            <span>Audio</span>
          </div>
          <div className="flex items-center">
            <VideoCameraIcon className="h-4 w-4 mr-1" />
            <span>Vídeo</span>
          </div>
        </div>
      </Link>
      {showDelete && (
        <button
          onClick={handleDeleteClick}
          disabled={isDeleting}
          className="absolute top-4 right-4 p-2 text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-300 hover:bg-red-50 dark:hover:bg-red-900/30 rounded-md transition-colors disabled:opacity-50 disabled:cursor-not-allowed z-10"
          title="Borrar reunión"
        >
          {isDeleting ? (
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-red-600 dark:border-red-400"></div>
          ) : (
            <TrashIcon className="h-5 w-5" />
          )}
        </button>
      )}
    </div>
  );
};

