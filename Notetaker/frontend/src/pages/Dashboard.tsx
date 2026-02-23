import { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { useMeetings } from '../hooks/useMeetings';
import { Link } from 'react-router-dom';
import { formatLocalDateTime } from '../utils/dateUtils';
import { CalendarIcon, ClockIcon, CheckCircleIcon, PlusIcon } from '@heroicons/react/24/outline';
import { CreateMeetingModal } from '../components/CreateMeetingModal';

export const Dashboard: React.FC = () => {
  const { user } = useAuth();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const { meetings: upcomingMeetings, loading: loadingUpcoming, refetch: refetchUpcoming } = useMeetings(
    user?.email,
    'pending',
    true // upcomingOnly=true para filtrar por fecha futura
  );
  const { meetings: pastMeetings, loading: loadingPast, refetch: refetchPast } = useMeetings(
    user?.email,
    'completed',
    false // upcomingOnly=false para pasadas
  );

  const upcomingCount = upcomingMeetings.length;
  const pastCount = pastMeetings.length;
  const recentMeetings = pastMeetings.slice(0, 5);

  const handleMeetingCreated = () => {
    refetchUpcoming();
    refetchPast();
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-slate-50">Dashboard</h1>
          <p className="mt-2 text-gray-600 dark:text-slate-400">Bienvenido de vuelta, {user?.display_name || user?.email}</p>
        </div>
        <button
          onClick={() => setIsModalOpen(true)}
          className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
        >
          <PlusIcon className="h-5 w-5" />
          Añadir Cosmos Notetaker a una reunión
        </button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <Link
          to="/meetings/upcoming"
          className="bg-white dark:bg-slate-800 rounded-lg shadow p-6 transition-colors hover:shadow-lg hover:scale-105 cursor-pointer"
        >
          <div className="flex items-center">
            <div className="p-3 bg-primary-100 dark:bg-primary-900/30 rounded-lg">
              <CalendarIcon className="h-6 w-6 text-primary-600 dark:text-primary-400" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600 dark:text-slate-400">Próximas</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-slate-50">
                {loadingUpcoming ? '...' : upcomingCount}
              </p>
            </div>
          </div>
        </Link>

        <Link
          to="/meetings/past"
          className="bg-white dark:bg-slate-800 rounded-lg shadow p-6 transition-colors hover:shadow-lg hover:scale-105 cursor-pointer"
        >
          <div className="flex items-center">
            <div className="p-3 bg-green-100 dark:bg-green-900/30 rounded-lg">
              <CheckCircleIcon className="h-6 w-6 text-green-600 dark:text-green-400" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600 dark:text-slate-400">Completadas</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-slate-50">
                {loadingPast ? '...' : pastCount}
              </p>
            </div>
          </div>
        </Link>

        <div className="bg-white dark:bg-slate-800 rounded-lg shadow p-6 transition-colors">
          <div className="flex items-center">
            <div className="p-3 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
              <ClockIcon className="h-6 w-6 text-blue-600 dark:text-blue-400" />
            </div>
            <div className="ml-4">
              <p className="text-sm font-medium text-gray-600 dark:text-slate-400">Total</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-slate-50">
                {loadingUpcoming || loadingPast ? '...' : upcomingCount + pastCount}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Meetings */}
      <div className="bg-white dark:bg-slate-800 rounded-lg shadow transition-colors">
        <div className="px-6 py-4 border-b border-gray-200 dark:border-slate-700">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-slate-50">Reuniones recientes</h2>
        </div>
        <div className="divide-y divide-gray-200 dark:divide-slate-700">
          {loadingPast ? (
            <div className="px-6 py-8 text-center text-gray-500 dark:text-slate-400">Cargando...</div>
          ) : recentMeetings.length === 0 ? (
            <div className="px-6 py-8 text-center text-gray-500 dark:text-slate-400">
              No hay reuniones recientes
            </div>
          ) : (
            recentMeetings.map((meeting) => (
              <Link
                key={meeting.id}
                to={`/meetings/${meeting.id}`}
                state={{ returnPath: '/dashboard' }}
                className="block px-6 py-4 hover:bg-gray-50 dark:hover:bg-slate-700 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-sm font-medium text-gray-900 dark:text-slate-50">
                      {meeting.title || 'Reunión sin título'}
                    </h3>
                    <p className="text-sm text-gray-500 dark:text-slate-400">
                      {formatLocalDateTime(meeting.scheduled_start_time)}
                    </p>
                  </div>
                  <span
                    className={`px-2 py-1 text-xs font-semibold rounded ${
                      meeting.status === 'completed'
                        ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-400'
                        : 'bg-gray-100 dark:bg-slate-700 text-gray-800 dark:text-slate-300'
                    }`}
                  >
                    {meeting.status}
                  </span>
                </div>
              </Link>
            ))
          )}
        </div>
      </div>

      {/* Modal de crear reunión */}
      <CreateMeetingModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onSuccess={handleMeetingCreated}
      />
    </div>
  );
};

