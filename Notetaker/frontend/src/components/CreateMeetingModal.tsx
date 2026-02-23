import { useState, useEffect, useRef } from 'react';
import { XMarkIcon, ChevronUpIcon, ChevronDownIcon } from '@heroicons/react/24/outline';
import { useAuth } from '../context/AuthContext';
import { meetingsApi } from '../services/api';
import { format } from 'date-fns';
import { es } from 'date-fns/locale';

interface CreateMeetingModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

export const CreateMeetingModal: React.FC<CreateMeetingModalProps> = ({
  isOpen,
  onClose,
  onSuccess,
}) => {
  const { user } = useAuth();
  const [meetingUrl, setMeetingUrl] = useState('');
  const [selectedDate, setSelectedDate] = useState('');
  // Estados separados para horas y minutos
  const [startHours, setStartHours] = useState('');
  const [startMinutes, setStartMinutes] = useState('');
  const [endHours, setEndHours] = useState('');
  const [endMinutes, setEndMinutes] = useState('');
  const [title, setTitle] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const endTimeManuallyModified = useRef(false);
  // Formato largo Teams corporativo: enlace con meetup-join (requiere ID y codigo manuales)
  const [meetingId, setMeetingId] = useState('');
  const [passcode, setPasscode] = useState('');
  const showLongFormatFields = Boolean(
    meetingUrl.trim() &&
    meetingUrl.includes('teams.microsoft.com') &&
    meetingUrl.includes('meetup-join')
  );

  // Inicializar con fecha y hora actuales cuando se abre el modal
  useEffect(() => {
    if (isOpen) {
      const now = new Date();
      const year = now.getFullYear();
      const month = String(now.getMonth() + 1).padStart(2, '0');
      const day = String(now.getDate()).padStart(2, '0');
      const hours = String(now.getHours()).padStart(2, '0');
      const minutes = String(now.getMinutes()).padStart(2, '0');

      setSelectedDate(`${year}-${month}-${day}`);
      setStartHours(hours);
      setStartMinutes(minutes);

      // Hora de fin: 1 hora despues por defecto
      const endDate = new Date(now);
      endDate.setHours(endDate.getHours() + 1);
      setEndHours(String(endDate.getHours()).padStart(2, '0'));
      setEndMinutes(String(endDate.getMinutes()).padStart(2, '0'));

      setMeetingUrl('');
      setTitle('');
      setMeetingId('');
      setPasscode('');
      setError(null);
      endTimeManuallyModified.current = false;
    }
  }, [isOpen]);

  // Actualizar hora de fin automaticamente cuando cambia la hora de inicio
  useEffect(() => {
    if (startHours && startMinutes && selectedDate && !endTimeManuallyModified.current) {
      const startDateTime = new Date(`${selectedDate}T${startHours}:${startMinutes}`);
      const endDateTime = new Date(startDateTime);
      endDateTime.setHours(endDateTime.getHours() + 1);

      setEndHours(String(endDateTime.getHours()).padStart(2, '0'));
      setEndMinutes(String(endDateTime.getMinutes()).padStart(2, '0'));
    }
  }, [startHours, startMinutes, selectedDate]);

  // Generar días del mes actual para el calendario
  const getDaysInMonth = () => {
    if (!selectedDate) return [];

    const [year, month] = selectedDate.split('-').map(Number);
    const firstDay = new Date(year, month - 1, 1);
    const lastDay = new Date(year, month, 0);
    const daysInMonth = lastDay.getDate();
    const startingDayOfWeek = firstDay.getDay();

    const days = [];

    // Días del mes anterior (para completar la primera semana)
    const prevMonth = new Date(year, month - 1, 0);
    const daysInPrevMonth = prevMonth.getDate();
    for (let i = startingDayOfWeek - 1; i >= 0; i--) {
      days.push({
        day: daysInPrevMonth - i,
        isCurrentMonth: false,
        date: new Date(year, month - 2, daysInPrevMonth - i),
      });
    }

    // Días del mes actual
    for (let day = 1; day <= daysInMonth; day++) {
      days.push({
        day,
        isCurrentMonth: true,
        date: new Date(year, month - 1, day),
      });
    }

    // Días del mes siguiente (para completar la última semana)
    const remainingDays = 42 - days.length;
    for (let day = 1; day <= remainingDays; day++) {
      days.push({
        day,
        isCurrentMonth: false,
        date: new Date(year, month, day),
      });
    }

    return days;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    // Validar URL de Teams
    if (!meetingUrl.includes('teams.microsoft.com') && !meetingUrl.includes('teams.live.com')) {
      setError('La URL debe ser de Microsoft Teams');
      setLoading(false);
      return;
    }

    // Si es formato largo corporativo, exigir ID de reunion y codigo de acceso
    if (showLongFormatFields) {
      const idTrim = meetingId.trim();
      const passTrim = passcode.trim();
      if (!idTrim) {
        setError('Para enlaces de Teams corporativos introduce el ID de reunion (solo numeros)');
        setLoading(false);
        return;
      }
      if (!/^\d{10,15}$/.test(idTrim)) {
        setError('El ID de reunion debe ser numerico (10-15 digitos)');
        setLoading(false);
        return;
      }
      if (!passTrim) {
        setError('Para enlaces de Teams corporativos introduce el codigo de acceso');
        setLoading(false);
        return;
      }
    }

    // Validar fecha y hora
    if (!selectedDate || !startHours || !startMinutes) {
      setError('Por favor, selecciona una fecha y hora de inicio');
      setLoading(false);
      return;
    }

    try {
      // Combinar fecha y hora
      const startDateTime = new Date(`${selectedDate}T${startHours}:${startMinutes}`);
      const endDateTime = (endHours && endMinutes) ? new Date(`${selectedDate}T${endHours}:${endMinutes}`) : undefined;

      // Validar que la hora de fin sea después de la de inicio
      if (endDateTime && endDateTime <= startDateTime) {
        setError('La hora de fin debe ser posterior a la hora de inicio');
        setLoading(false);
        return;
      }

      // Crear reunión (incluir ID y codigo si es formato largo)
      await meetingsApi.create(
        meetingUrl,
        startDateTime.toISOString(),
        endDateTime?.toISOString(),
        title || undefined,
        user?.email,
        showLongFormatFields ? meetingId.trim() : undefined,
        showLongFormatFields ? passcode.trim() : undefined
      );

      onSuccess();
      onClose();
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Error al crear la reunión');
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  const days = getDaysInMonth();
  const currentDate = selectedDate ? new Date(selectedDate) : new Date();
  const monthName = format(currentDate, 'MMMM yyyy', { locale: es });
  const weekDays = ['Dom', 'Lun', 'Mar', 'Mié', 'Jue', 'Vie', 'Sáb'];

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex items-center justify-center min-h-screen px-4 pt-4 pb-20 text-center sm:block sm:p-0">
        {/* Overlay: no cerrar al hacer clic para no perder datos por error */}
        <div
          className="fixed inset-0 transition-opacity bg-gray-500 bg-opacity-75"
          aria-hidden="true"
        />

        {/* Modal */}
        <div className="inline-block align-bottom bg-white dark:bg-slate-800 rounded-lg text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-2xl sm:w-full transition-colors">
          <div className="bg-white dark:bg-slate-800 px-4 pt-5 pb-4 sm:p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900 dark:text-slate-50">
                Añadir Cosmos Notetaker a una reunión
              </h3>
              <button
                onClick={onClose}
                className="text-gray-400 dark:text-slate-500 hover:text-gray-500 dark:hover:text-slate-300"
              >
                <XMarkIcon className="h-6 w-6" />
              </button>
            </div>

            <form onSubmit={handleSubmit} className="space-y-4">
              {/* URL de Teams */}
              <div>
                <label htmlFor="meetingUrl" className="block text-sm font-medium text-gray-700 dark:text-slate-300">
                  URL de la reunión de Teams
                </label>
                <input
                  type="url"
                  id="meetingUrl"
                  value={meetingUrl}
                  onChange={(e) => setMeetingUrl(e.target.value)}
                  placeholder="https://teams.microsoft.com/l/meetup-join/..."
                  className="mt-1 block w-full px-3 py-2 rounded-md border-gray-300 dark:border-slate-600 shadow-sm bg-white dark:bg-slate-700 text-gray-900 dark:text-slate-50 focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                  required
                />
              </div>

              {/* ID de reunion y Codigo de acceso (solo para formato largo corporativo) */}
              {showLongFormatFields && (
                <>
                  <div>
                    <label htmlFor="meetingId" className="block text-sm font-medium text-gray-700 dark:text-slate-300">
                      ID de reunion
                    </label>
                    <input
                      type="text"
                      id="meetingId"
                      value={meetingId}
                      onChange={(e) => setMeetingId(e.target.value.replace(/\D/g, '').slice(0, 15))}
                      placeholder="Solo numeros (10-15 digitos)"
                      className="mt-1 block w-full px-3 py-2 rounded-md border-gray-300 dark:border-slate-600 shadow-sm bg-white dark:bg-slate-700 text-gray-900 dark:text-slate-50 focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                      inputMode="numeric"
                      maxLength={15}
                    />
                  </div>
                  <div>
                    <label htmlFor="passcode" className="block text-sm font-medium text-gray-700 dark:text-slate-300">
                      Codigo de acceso
                    </label>
                    <input
                      type="text"
                      id="passcode"
                      value={passcode}
                      onChange={(e) => setPasscode(e.target.value)}
                      placeholder="PIN o codigo de la reunion"
                      className="mt-1 block w-full px-3 py-2 rounded-md border-gray-300 dark:border-slate-600 shadow-sm bg-white dark:bg-slate-700 text-gray-900 dark:text-slate-50 focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                    />
                  </div>
                </>
              )}

              {/* Calendario */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
                  Fecha de la reunión
                </label>
                <div className="border border-gray-300 dark:border-slate-600 rounded-lg p-4 bg-white dark:bg-slate-700">
                  <div className="flex items-center justify-between mb-4">
                    <h4 className="text-lg font-semibold capitalize text-gray-900 dark:text-slate-50">{monthName}</h4>
                    <div className="flex gap-2">
                      <button
                        type="button"
                        onClick={() => {
                          const [year, month] = selectedDate.split('-').map(Number);
                          const newDate = new Date(year, month - 2, 1);
                          setSelectedDate(format(newDate, 'yyyy-MM-dd'));
                        }}
                        className="px-2 py-1 text-sm text-gray-600 dark:text-slate-400 hover:bg-gray-100 dark:hover:bg-slate-600 rounded"
                      >
                        ←
                      </button>
                      <button
                        type="button"
                        onClick={() => {
                          const [year, month] = selectedDate.split('-').map(Number);
                          const newDate = new Date(year, month, 1);
                          setSelectedDate(format(newDate, 'yyyy-MM-dd'));
                        }}
                        className="px-2 py-1 text-sm text-gray-600 dark:text-slate-400 hover:bg-gray-100 dark:hover:bg-slate-600 rounded"
                      >
                        →
                      </button>
                    </div>
                  </div>

                  {/* Días de la semana */}
                  <div className="grid grid-cols-7 gap-1 mb-2">
                    {weekDays.map((day) => (
                      <div key={day} className="text-center text-xs font-medium text-gray-500 dark:text-slate-400 py-2">
                        {day}
                      </div>
                    ))}
                  </div>

                  {/* Días del mes */}
                  <div className="grid grid-cols-7 gap-1">
                    {days.map((dayInfo, idx) => {
                      const isSelected =
                        dayInfo.isCurrentMonth &&
                        selectedDate &&
                        format(dayInfo.date, 'yyyy-MM-dd') === selectedDate;

                      return (
                        <button
                          key={idx}
                          type="button"
                          onClick={() => {
                            if (dayInfo.isCurrentMonth) {
                              setSelectedDate(format(dayInfo.date, 'yyyy-MM-dd'));
                            }
                          }}
                          className={`py-2 text-sm rounded ${
                            !dayInfo.isCurrentMonth
                              ? 'text-gray-300 dark:text-slate-600'
                              : isSelected
                              ? 'bg-primary-600 dark:bg-primary-500 text-white'
                              : 'text-gray-700 dark:text-slate-300 hover:bg-gray-100 dark:hover:bg-slate-600'
                          }`}
                        >
                          {dayInfo.day}
                        </button>
                      );
                    })}
                  </div>
                </div>
              </div>

              {/* Horas */}
              <div className="grid grid-cols-2 gap-4">
                {/* HORA DE INICIO */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-slate-300">
                    Hora de inicio
                  </label>
                  <div className="mt-1 flex items-center gap-1">
                    {/* Input Horas */}
                    <div className="flex items-center">
                      <input
                        type="text"
                        inputMode="numeric"
                        maxLength={2}
                        value={startHours}
                        onFocus={(e) => e.target.select()}
                        onChange={(e) => {
                          const val = e.target.value.replace(/\D/g, '').slice(0, 2);
                          const num = parseInt(val, 10);
                          if (val === '' || (num >= 0 && num <= 23)) {
                            setStartHours(val);
                          }
                        }}
                        onBlur={(e) => {
                          const val = e.target.value;
                          if (val) {
                            setStartHours(val.padStart(2, '0'));
                          }
                        }}
                        className="w-10 text-center rounded-md border-gray-300 dark:border-slate-600 shadow-sm bg-white dark:bg-slate-700 text-gray-900 dark:text-slate-50 focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                        placeholder="HH"
                      />
                      <div className="flex flex-col ml-0.5">
                        <button
                          type="button"
                          onClick={() => {
                            const h = parseInt(startHours || '0', 10);
                            setStartHours(String((h + 1) % 24).padStart(2, '0'));
                          }}
                          className="p-0.5 hover:bg-gray-100 dark:hover:bg-slate-600 rounded text-gray-600 dark:text-slate-400"
                        >
                          <ChevronUpIcon className="h-3 w-3" />
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            const h = parseInt(startHours || '0', 10);
                            setStartHours(String(h === 0 ? 23 : h - 1).padStart(2, '0'));
                          }}
                          className="p-0.5 hover:bg-gray-100 dark:hover:bg-slate-600 rounded text-gray-600 dark:text-slate-400"
                        >
                          <ChevronDownIcon className="h-3 w-3" />
                        </button>
                      </div>
                    </div>

                    <span className="text-gray-900 dark:text-slate-50 font-medium">:</span>

                    {/* Input Minutos */}
                    <div className="flex items-center">
                      <input
                        type="text"
                        inputMode="numeric"
                        maxLength={2}
                        value={startMinutes}
                        onFocus={(e) => e.target.select()}
                        onChange={(e) => {
                          const val = e.target.value.replace(/\D/g, '').slice(0, 2);
                          const num = parseInt(val, 10);
                          if (val === '' || (num >= 0 && num <= 59)) {
                            setStartMinutes(val);
                          }
                        }}
                        onBlur={(e) => {
                          const val = e.target.value;
                          if (val) {
                            setStartMinutes(val.padStart(2, '0'));
                          }
                        }}
                        className="w-10 text-center rounded-md border-gray-300 dark:border-slate-600 shadow-sm bg-white dark:bg-slate-700 text-gray-900 dark:text-slate-50 focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                        placeholder="MM"
                      />
                      <div className="flex flex-col ml-0.5">
                        <button
                          type="button"
                          onClick={() => {
                            const m = parseInt(startMinutes || '0', 10);
                            setStartMinutes(String((m + 1) % 60).padStart(2, '0'));
                          }}
                          className="p-0.5 hover:bg-gray-100 dark:hover:bg-slate-600 rounded text-gray-600 dark:text-slate-400"
                        >
                          <ChevronUpIcon className="h-3 w-3" />
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            const m = parseInt(startMinutes || '0', 10);
                            setStartMinutes(String(m === 0 ? 59 : m - 1).padStart(2, '0'));
                          }}
                          className="p-0.5 hover:bg-gray-100 dark:hover:bg-slate-600 rounded text-gray-600 dark:text-slate-400"
                        >
                          <ChevronDownIcon className="h-3 w-3" />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>

                {/* HORA DE FIN */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-slate-300">
                    Hora de fin
                  </label>
                  <div className="mt-1 flex items-center gap-1">
                    {/* Input Horas */}
                    <div className="flex items-center">
                      <input
                        type="text"
                        inputMode="numeric"
                        maxLength={2}
                        value={endHours}
                        onFocus={(e) => {
                          // Marcar como editado manualmente y seleccionar todo el texto
                          endTimeManuallyModified.current = true;
                          e.target.select();
                        }}
                        onChange={(e) => {
                          endTimeManuallyModified.current = true;
                          const val = e.target.value.replace(/\D/g, '').slice(0, 2);
                          const num = parseInt(val, 10);
                          if (val === '' || isNaN(num) || (num >= 0 && num <= 23)) {
                            setEndHours(val);
                          }
                        }}
                        onBlur={(e) => {
                          const val = e.target.value;
                          if (val) {
                            const num = parseInt(val, 10);
                            if (!isNaN(num) && num >= 0 && num <= 23) {
                              setEndHours(String(num).padStart(2, '0'));
                            }
                          }
                        }}
                        className="w-10 text-center rounded-md border-gray-300 dark:border-slate-600 shadow-sm bg-white dark:bg-slate-700 text-gray-900 dark:text-slate-50 focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                        placeholder="HH"
                      />
                      <div className="flex flex-col ml-0.5">
                        <button
                          type="button"
                          onClick={() => {
                            endTimeManuallyModified.current = true;
                            const h = parseInt(endHours || '0', 10);
                            setEndHours(String((h + 1) % 24).padStart(2, '0'));
                          }}
                          className="p-0.5 hover:bg-gray-100 dark:hover:bg-slate-600 rounded text-gray-600 dark:text-slate-400"
                        >
                          <ChevronUpIcon className="h-3 w-3" />
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            endTimeManuallyModified.current = true;
                            const h = parseInt(endHours || '0', 10);
                            setEndHours(String(h === 0 ? 23 : h - 1).padStart(2, '0'));
                          }}
                          className="p-0.5 hover:bg-gray-100 dark:hover:bg-slate-600 rounded text-gray-600 dark:text-slate-400"
                        >
                          <ChevronDownIcon className="h-3 w-3" />
                        </button>
                      </div>
                    </div>

                    <span className="text-gray-900 dark:text-slate-50 font-medium">:</span>

                    {/* Input Minutos */}
                    <div className="flex items-center">
                      <input
                        type="text"
                        inputMode="numeric"
                        maxLength={2}
                        value={endMinutes}
                        onFocus={(e) => {
                          // Marcar como editado manualmente y seleccionar todo el texto
                          endTimeManuallyModified.current = true;
                          e.target.select();
                        }}
                        onChange={(e) => {
                          endTimeManuallyModified.current = true;
                          const val = e.target.value.replace(/\D/g, '').slice(0, 2);
                          const num = parseInt(val, 10);
                          if (val === '' || isNaN(num) || (num >= 0 && num <= 59)) {
                            setEndMinutes(val);
                          }
                        }}
                        onBlur={(e) => {
                          const val = e.target.value;
                          if (val) {
                            const num = parseInt(val, 10);
                            if (!isNaN(num) && num >= 0 && num <= 59) {
                              setEndMinutes(String(num).padStart(2, '0'));
                            }
                          }
                        }}
                        className="w-10 text-center rounded-md border-gray-300 dark:border-slate-600 shadow-sm bg-white dark:bg-slate-700 text-gray-900 dark:text-slate-50 focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                        placeholder="MM"
                      />
                      <div className="flex flex-col ml-0.5">
                        <button
                          type="button"
                          onClick={() => {
                            endTimeManuallyModified.current = true;
                            const m = parseInt(endMinutes || '0', 10);
                            setEndMinutes(String((m + 1) % 60).padStart(2, '0'));
                          }}
                          className="p-0.5 hover:bg-gray-100 dark:hover:bg-slate-600 rounded text-gray-600 dark:text-slate-400"
                        >
                          <ChevronUpIcon className="h-3 w-3" />
                        </button>
                        <button
                          type="button"
                          onClick={() => {
                            endTimeManuallyModified.current = true;
                            const m = parseInt(endMinutes || '0', 10);
                            setEndMinutes(String(m === 0 ? 59 : m - 1).padStart(2, '0'));
                          }}
                          className="p-0.5 hover:bg-gray-100 dark:hover:bg-slate-600 rounded text-gray-600 dark:text-slate-400"
                        >
                          <ChevronDownIcon className="h-3 w-3" />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Título (opcional) */}
              <div>
                <label htmlFor="title" className="block text-sm font-medium text-gray-700 dark:text-slate-300">
                  Título de la reunión (opcional)
                </label>
                <input
                  type="text"
                  id="title"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="Ej: Reunión de equipo"
                  className="mt-1 block w-full px-3 py-2 rounded-md border-gray-300 dark:border-slate-600 shadow-sm bg-white dark:bg-slate-700 text-gray-900 dark:text-slate-50 focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
                />
              </div>

              {/* Error */}
              {error && (
                <div className="bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-md p-3">
                  <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
                </div>
              )}

              {/* Botones */}
              <div className="flex justify-end gap-3 pt-4">
                <button
                  type="button"
                  onClick={onClose}
                  className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-slate-300 bg-white dark:bg-slate-700 border border-gray-300 dark:border-slate-600 rounded-md hover:bg-gray-50 dark:hover:bg-slate-600"
                >
                  Cancelar
                </button>
                <button
                  type="submit"
                  disabled={loading}
                  className="px-4 py-2 text-sm font-medium text-white bg-primary-600 border border-transparent rounded-md hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? 'Creando...' : 'Aceptar'}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};
