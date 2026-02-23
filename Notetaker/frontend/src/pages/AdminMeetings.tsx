import { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { useMeetings } from '../hooks/useMeetings';
import { Link } from 'react-router-dom';
import { formatLocalDateTime, isMeetingToday } from '../utils/dateUtils';
import { TrashIcon } from '@heroicons/react/24/outline';
import { meetingsApi } from '../services/api';
import { ConfirmDialog } from '../components/ConfirmDialog';
import type { Meeting } from '../types';

const ADMIN_MEETINGS_SORT_KEY = 'notetaker_admin_meetings_sort';

function loadSavedSort(): { field: 'title' | 'date' | 'status' | 'organizer'; dir: 'asc' | 'desc' } {
  try {
    const raw = localStorage.getItem(ADMIN_MEETINGS_SORT_KEY);
    if (raw) {
      const p = JSON.parse(raw) as { sortField?: string; sortDirection?: string } | null;
      if (p?.sortField && ['title', 'date', 'status', 'organizer'].includes(p.sortField) && p?.sortDirection && ['asc', 'desc'].includes(p.sortDirection)) {
        return { field: p.sortField as 'title' | 'date' | 'status' | 'organizer', dir: p.sortDirection as 'asc' | 'desc' };
      }
    }
  } catch {
    // ignore
  }
  return { field: 'date', dir: 'desc' };
}

export const AdminMeetings: React.FC = () => {
  const { user } = useAuth();
  // Admin puede ver todas las reuniones sin filtrar por usuario
  const { meetings, loading, error, refetch } = useMeetings(user?.email, undefined, undefined, true);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [deleteModal, setDeleteModal] = useState<{ meeting: Meeting } | null>(null);
  const [searchTitle, setSearchTitle] = useState<string>('');
  const [sort, setSort] = useState<{ field: 'title' | 'date' | 'status' | 'organizer'; dir: 'asc' | 'desc' }>(loadSavedSort);
  const sortField = sort.field;
  const sortDirection = sort.dir;

  // Persistir orden por defecto si no existe clave (asi la clave aparece y se aplica desde el primer uso)
  useEffect(() => {
    try {
      if (typeof localStorage === 'undefined') return;
      const existing = localStorage.getItem(ADMIN_MEETINGS_SORT_KEY);
      if (!existing) {
        localStorage.setItem(ADMIN_MEETINGS_SORT_KEY, JSON.stringify({ sortField: sort.field, sortDirection: sort.dir }));
      }
    } catch {
      // ignore
    }
  }, [sort.field, sort.dir]);

  const handleDeleteClick = (meeting: Meeting) => {
    setDeleteModal({ meeting });
  };

  const handleDeleteConfirm = async (deleteContent: boolean) => {
    if (!deleteModal) return;
    
    const meetingId = deleteModal.meeting.id;
    setDeletingId(meetingId);
    try {
      await meetingsApi.delete(meetingId, user?.email, deleteContent, true); // as_admin=true porque viene de la vista de admin
      setDeleteModal(null);
      refetch();
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Error al borrar la reunión');
    } finally {
      setDeletingId(null);
    }
  };

  const handleDeleteCancel = () => {
    setDeleteModal(null);
  };

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

  const handleSort = (field: 'title' | 'date' | 'status' | 'organizer') => {
    const nextField = sortField === field ? sortField : field;
    const nextDir = sortField === field ? (sortDirection === 'asc' ? 'desc' : 'asc') : 'asc';
    setSort({ field: nextField, dir: nextDir });
    try {
      if (typeof localStorage !== 'undefined') {
        localStorage.setItem(ADMIN_MEETINGS_SORT_KEY, JSON.stringify({ sortField: nextField, sortDirection: nextDir }));
      }
    } catch {
      // ignore
    }
  };

  const renderSortIndicator = (field: 'title' | 'date' | 'status' | 'organizer') => {
    if (sortField !== field) return null;
    return sortDirection === 'asc' ? ' ▲' : ' ▼';
  };

  const normalizedSearch = searchTitle.trim().toLowerCase();
  const filteredMeetings = normalizedSearch
    ? meetings.filter((meeting) =>
        (meeting.title || 'Reunión sin título').toLowerCase().includes(normalizedSearch)
      )
    : meetings;

  const sortedMeetings = [...filteredMeetings].sort((a, b) => {
    let cmp = 0;

    if (sortField === 'title') {
      const titleA = (a.title || 'Reunión sin título').toLowerCase();
      const titleB = (b.title || 'Reunión sin título').toLowerCase();
      cmp = titleA.localeCompare(titleB, 'es');
      if (cmp === 0) {
        const timeA = new Date(a.scheduled_start_time).getTime();
        const timeB = new Date(b.scheduled_start_time).getTime();
        cmp = timeA - timeB;
      }
    } else if (sortField === 'date') {
      const timeA = new Date(a.scheduled_start_time).getTime();
      const timeB = new Date(b.scheduled_start_time).getTime();
      cmp = timeA - timeB;
    } else if (sortField === 'status') {
      const statusA = (a.status || '').toLowerCase();
      const statusB = (b.status || '').toLowerCase();
      cmp = statusA.localeCompare(statusB, 'es');
      if (cmp === 0) {
        const timeA = new Date(a.scheduled_start_time).getTime();
        const timeB = new Date(b.scheduled_start_time).getTime();
        cmp = timeA - timeB;
      }
    } else if (sortField === 'organizer') {
      const orgA = (a.organizer_name || a.organizer_email || '').toLowerCase();
      const orgB = (b.organizer_name || b.organizer_email || '').toLowerCase();
      cmp = orgA.localeCompare(orgB, 'es');
      if (cmp === 0) {
        const timeA = new Date(a.scheduled_start_time).getTime();
        const timeB = new Date(b.scheduled_start_time).getTime();
        cmp = timeA - timeB;
      }
    }

    return sortDirection === 'asc' ? cmp : -cmp;
  });

  return (
    <div className="space-y-6">
      <div className="space-y-4 sm:space-y-0 sm:flex sm:items-end sm:justify-between sm:gap-4">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-slate-50">Todas las reuniones</h1>
          <p className="mt-2 text-gray-600 dark:text-slate-400">
            Vista global de todas las reuniones del sistema
          </p>
        </div>
        {meetings.length > 0 && (
          <div className="sm:w-80">
            <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1">
              Buscar por título
            </label>
            <input
              type="text"
              value={searchTitle}
              onChange={(e) => setSearchTitle(e.target.value)}
              placeholder="Escribe para filtrar por título..."
              className="block w-full rounded-md border-gray-300 dark:border-slate-600 dark:bg-slate-800 dark:text-slate-50 shadow-sm focus:border-primary-500 focus:ring-primary-500 text-sm"
            />
          </div>
        )}
      </div>

      {meetings.length === 0 ? (
        <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-12 text-center transition-colors">
          <p className="text-gray-500 dark:text-slate-400">No hay reuniones en el sistema</p>
        </div>
      ) : (
        <div className="bg-white dark:bg-slate-800 rounded-lg shadow overflow-hidden transition-colors">
          <table className="min-w-full divide-y divide-gray-200 dark:divide-slate-700">
            <thead className="bg-gray-50 dark:bg-slate-700">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-slate-300 uppercase tracking-wider">
                  <button
                    type="button"
                    onClick={() => handleSort('title')}
                    className="flex items-center space-x-1 hover:text-gray-900 dark:hover:text-slate-100 focus:outline-none"
                  >
                    <span>Título</span>
                    <span>{renderSortIndicator('title')}</span>
                  </button>
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-slate-300 uppercase tracking-wider">
                  <button
                    type="button"
                    onClick={() => handleSort('date')}
                    className="flex items-center space-x-1 hover:text-gray-900 dark:hover:text-slate-100 focus:outline-none"
                  >
                    <span>Fecha</span>
                    <span>{renderSortIndicator('date')}</span>
                  </button>
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-slate-300 uppercase tracking-wider">
                  <button
                    type="button"
                    onClick={() => handleSort('status')}
                    className="flex items-center space-x-1 hover:text-gray-900 dark:hover:text-slate-100 focus:outline-none"
                  >
                    <span>Estado</span>
                    <span>{renderSortIndicator('status')}</span>
                  </button>
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-slate-300 uppercase tracking-wider">
                  <button
                    type="button"
                    onClick={() => handleSort('organizer')}
                    className="flex items-center space-x-1 hover:text-gray-900 dark:hover:text-slate-100 focus:outline-none"
                  >
                    <span>Organizador</span>
                    <span>{renderSortIndicator('organizer')}</span>
                  </button>
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-slate-300 uppercase tracking-wider">
                  Acciones
                </th>
              </tr>
            </thead>
            <tbody className="bg-white dark:bg-slate-800 divide-y divide-gray-200 dark:divide-slate-700">
              {sortedMeetings.map((meeting) => {
                const today = isMeetingToday(meeting.scheduled_start_time);
                const necesitaResincronizar = meeting.necesita_resincronizar_calendario === true;
                const rowClassName = necesitaResincronizar
                  ? 'bg-orange-50 dark:bg-orange-900/20 border-l-4 border-l-orange-500 dark:border-l-orange-400'
                  : today
                    ? 'bg-amber-50 dark:bg-amber-900/20 border-l-4 border-l-amber-500 dark:border-l-amber-400'
                    : undefined;
                return (
                <tr key={meeting.id} className={rowClassName}>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="text-sm font-medium text-gray-900 dark:text-slate-50">
                      {meeting.title || 'Reunión sin título'}
                    </div>
                    {necesitaResincronizar && (
                      <div className="text-xs text-orange-700 dark:text-orange-300 mt-1">
                        El bot no se unirá; resincronizar calendario para obtener ID y código.
                      </div>
                    )}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-slate-400">
                    {formatLocalDateTime(meeting.scheduled_start_time)}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap">
                    <span
                      className={`px-2 py-1 text-xs font-semibold rounded ${
                        meeting.status === 'completed'
                          ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-400'
                          : meeting.status === 'pending'
                          ? 'bg-yellow-100 dark:bg-yellow-900/30 text-yellow-800 dark:text-yellow-400'
                          : 'bg-gray-100 dark:bg-slate-700 text-gray-800 dark:text-slate-300'
                      }`}
                    >
                      {meeting.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500 dark:text-slate-400">
                    {meeting.organizer_name || meeting.organizer_email || '-'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                    <div className="flex items-center space-x-3">
                      <Link
                        to={`/meetings/${meeting.id}`}
                        state={{ returnPath: '/admin/meetings' }}
                        className="text-primary-600 dark:text-primary-400 hover:text-primary-900 dark:hover:text-primary-300"
                      >
                        Ver detalles
                      </Link>
                      <button
                        onClick={() => handleDeleteClick(meeting)}
                        disabled={deletingId === meeting.id}
                        className="text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-1"
                        title="Borrar reunión"
                      >
                        {deletingId === meeting.id ? (
                          <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-red-600 dark:border-red-400"></div>
                        ) : (
                          <>
                            <TrashIcon className="h-4 w-4" />
                            <span>Borrar</span>
                          </>
                        )}
                      </button>
                    </div>
                  </td>
                </tr>
              );
              })}
            </tbody>
          </table>
        </div>
      )}

      {deleteModal && (
        <ConfirmDialog
          isOpen={true}
          onClose={handleDeleteCancel}
          onConfirm={() => handleDeleteConfirm(true)}
          title={deleteModal.meeting.status === 'completed' ? 'Borrar reunión completada' : 'Borrar reunión'}
          message={
            deleteModal.meeting.status === 'completed'
              ? 'Esta reunión está completada y contiene transcripción, audio y/o video. ¿Quieres borrar todo el contenido de esta reunión (transcripción, audio y video)?'
              : deleteModal.meeting.necesita_resincronizar_calendario
                ? 'Esta reunión no tiene ID/código extraídos; el bot no se unirá. Al borrarla, la próxima sincronización del calendario del usuario volverá a crear la reunión con ID y código. ¿Borrar definitivamente?'
                : '¿Estás seguro de que quieres borrar esta reunión? Esto la eliminará para todos los usuarios.'
          }
          confirmText="Sí, borrar"
          cancelText="Cancelar"
          confirmButtonColor="red"
          isLoading={deletingId === deleteModal.meeting.id}
        />
      )}
    </div>
  );
};

