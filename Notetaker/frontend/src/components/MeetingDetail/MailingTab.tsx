import { useState, useEffect } from 'react';
import { meetingsApi } from '../../services/api';
import { XMarkIcon, CheckCircleIcon, InformationCircleIcon } from '@heroicons/react/24/outline';
import type { Meeting, MeetingAccess } from '../../types';

interface MailingTabProps {
  meeting: Meeting;
  meetingAccess: MeetingAccess[];
  transcription: any;
}

export const MailingTab: React.FC<MailingTabProps> = ({ meeting, meetingAccess, transcription }) => {
  const [participantEmails, setParticipantEmails] = useState<string[]>([]);
  const [selectedRecipients, setSelectedRecipients] = useState<string[]>([]);
  const [additionalRecipients, setAdditionalRecipients] = useState<string[]>([]);
  const [ccRecipients, setCcRecipients] = useState<string[]>([]);
  const [sendToAll, setSendToAll] = useState(false);
  const [subject, setSubject] = useState('');
  const [newEmail, setNewEmail] = useState('');
  const [newCcEmail, setNewCcEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);

  // Cargar emails de participantes al montar
  useEffect(() => {
    const emails: string[] = [];
    
    // Obtener emails de MeetingAccess
    meetingAccess.forEach((access) => {
      if (access.user_email && !emails.includes(access.user_email)) {
        emails.push(access.user_email);
      }
    });
    
    // Obtener speakers únicos de la transcripción
    if (transcription?.conversation) {
      const speakers = new Set<string>();
      transcription.conversation.forEach((segment: any) => {
        if (segment.speaker) {
          speakers.add(segment.speaker);
        }
      });
      // Nota: Los speakers son nombres, no emails. Se mostrarán pero no se usarán como emails
      // a menos que coincidan con los emails de MeetingAccess
    }
    
    setParticipantEmails(emails);
    setSelectedRecipients(emails);
    setSendToAll(true);
  }, [meetingAccess, transcription]);

  // Actualizar asunto cuando cambia el título de la reunión
  useEffect(() => {
    if (meeting.title) {
      setSubject(`Resumen de reunión: ${meeting.title}`);
    } else {
      setSubject('Resumen de reunión');
    }
  }, [meeting.title]);

  // Actualizar destinatarios seleccionados cuando cambia "enviar a todos"
  useEffect(() => {
    if (sendToAll) {
      setSelectedRecipients([...participantEmails]);
    }
  }, [sendToAll, participantEmails]);

  // Validar formato de email
  const isValidEmail = (email: string): boolean => {
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    return emailRegex.test(email);
  };

  // Añadir email adicional
  const handleAddEmail = () => {
    const trimmedEmail = newEmail.trim();
    if (!trimmedEmail) return;
    
    if (!isValidEmail(trimmedEmail)) {
      setError('Por favor, introduce un email válido');
      return;
    }
    
    if (additionalRecipients.includes(trimmedEmail) || selectedRecipients.includes(trimmedEmail)) {
      setError('Este email ya está en la lista');
      return;
    }
    
    setAdditionalRecipients([...additionalRecipients, trimmedEmail]);
    setNewEmail('');
    setError(null);
  };

  // Añadir email CC
  const handleAddCcEmail = () => {
    const trimmedEmail = newCcEmail.trim();
    if (!trimmedEmail) return;
    
    if (!isValidEmail(trimmedEmail)) {
      setError('Por favor, introduce un email válido');
      return;
    }
    
    if (ccRecipients.includes(trimmedEmail)) {
      setError('Este email ya está en la lista de CC');
      return;
    }
    
    setCcRecipients([...ccRecipients, trimmedEmail]);
    setNewCcEmail('');
    setError(null);
  };

  // Remover email de participantes seleccionados
  const handleRemoveParticipant = (email: string) => {
    setSelectedRecipients(selectedRecipients.filter(e => e !== email));
    setSendToAll(false);
  };

  // Remover email adicional
  const handleRemoveAdditional = (email: string) => {
    setAdditionalRecipients(additionalRecipients.filter(e => e !== email));
  };

  // Remover email CC
  const handleRemoveCc = (email: string) => {
    setCcRecipients(ccRecipients.filter(e => e !== email));
  };

  // Toggle selección de participante
  const handleToggleParticipant = (email: string) => {
    if (selectedRecipients.includes(email)) {
      handleRemoveParticipant(email);
    } else {
      setSelectedRecipients([...selectedRecipients, email]);
      setSendToAll(false);
    }
  };

  // Enviar email
  const handleSendEmail = async () => {
    setError(null);
    setSuccess(null);
    
    // Validar que hay al menos un destinatario
    const allRecipients = [...selectedRecipients, ...additionalRecipients];
    if (allRecipients.length === 0) {
      setError('Debes seleccionar al menos un destinatario');
      return;
    }
    
    // Validar asunto
    if (!subject.trim()) {
      setError('El asunto es obligatorio');
      return;
    }
    
    setLoading(true);
    
    try {
      const result = await meetingsApi.sendSummaryEmail(
        meeting.id,
        allRecipients,
        subject.trim(),
        ccRecipients.length > 0 ? ccRecipients : undefined,
        additionalRecipients.length > 0 ? additionalRecipients : undefined
      );
      
      setSuccess(result.message || 'Email enviado exitosamente');
      // Limpiar formulario después de 3 segundos
      setTimeout(() => {
        setSuccess(null);
      }, 3000);
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Error al enviar el email');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h2 className="text-xl font-semibold text-gray-900 dark:text-slate-50 mb-2">
          Enviar resumen por email
        </h2>
        <p className="text-sm text-gray-600 dark:text-slate-400">
          Envía un resumen de la reunión por correo electrónico a los participantes y otros destinatarios.
        </p>
      </div>

      {/* Checkbox para enviar a todos */}
      <div className="flex items-start space-x-3">
        <input
          type="checkbox"
          id="sendToAll"
          checked={sendToAll}
          onChange={(e) => {
            setSendToAll(e.target.checked);
            if (e.target.checked) {
              setSelectedRecipients([...participantEmails]);
            }
          }}
          className="mt-1 h-4 w-4 text-primary-600 focus:ring-primary-500 border-gray-300 rounded"
        />
        <label htmlFor="sendToAll" className="text-sm font-medium text-gray-700 dark:text-slate-300">
          Enviar a todos los participantes de la reunión
        </label>
      </div>

      {/* Lista de participantes */}
      {participantEmails.length > 0 && (
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
            Para <span className="text-red-500">*</span>
          </label>
          <div className="flex flex-wrap gap-2 p-3 border border-gray-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700 min-h-[50px]">
            {participantEmails.map((email) => (
              <div
                key={email}
                className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-sm ${
                  selectedRecipients.includes(email)
                    ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-400 border border-green-300 dark:border-green-700'
                    : 'bg-gray-100 dark:bg-slate-600 text-gray-700 dark:text-slate-300 border border-gray-300 dark:border-slate-500'
                }`}
              >
                {selectedRecipients.includes(email) && (
                  <CheckCircleIcon className="h-4 w-4" />
                )}
                <span>{email}</span>
                {selectedRecipients.includes(email) && (
                  <button
                    type="button"
                    onClick={() => handleRemoveParticipant(email)}
                    className="ml-1 text-gray-500 hover:text-gray-700 dark:text-slate-400 dark:hover:text-slate-200"
                  >
                    <XMarkIcon className="h-4 w-4" />
                  </button>
                )}
                {!selectedRecipients.includes(email) && (
                  <button
                    type="button"
                    onClick={() => handleToggleParticipant(email)}
                    className="ml-1 text-gray-500 hover:text-gray-700 dark:text-slate-400 dark:hover:text-slate-200"
                    title="Añadir a destinatarios"
                  >
                    <span className="text-xs">+</span>
                  </button>
                )}
              </div>
            ))}
            {participantEmails.length === 0 && (
              <span className="text-gray-500 dark:text-slate-400 text-sm">No hay participantes disponibles</span>
            )}
          </div>
        </div>
      )}

      {/* Emails adicionales */}
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
          Añadir destinatarios adicionales
        </label>
        <div className="flex gap-2">
          <input
            type="email"
            value={newEmail}
            onChange={(e) => {
              setNewEmail(e.target.value);
              setError(null);
            }}
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                handleAddEmail();
              }
            }}
            placeholder="email@ejemplo.com"
            className="flex-1 px-3 py-2 border border-gray-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700 text-gray-900 dark:text-slate-50 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
          <button
            type="button"
            onClick={handleAddEmail}
            className="px-4 py-2 bg-primary-600 text-white rounded-lg hover:bg-primary-700 focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
          >
            Añadir
          </button>
        </div>
        {additionalRecipients.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-2">
            {additionalRecipients.map((email) => (
              <div
                key={email}
                className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-sm bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-400 border border-blue-300 dark:border-blue-700"
              >
                <span>{email}</span>
                <button
                  type="button"
                  onClick={() => handleRemoveAdditional(email)}
                  className="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-200"
                >
                  <XMarkIcon className="h-4 w-4" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* CC */}
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
          CC
        </label>
        <div className="flex gap-2">
          <input
            type="email"
            value={newCcEmail}
            onChange={(e) => {
              setNewCcEmail(e.target.value);
              setError(null);
            }}
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                handleAddCcEmail();
              }
            }}
            placeholder="--Seleccionar correo electrónico--"
            className="flex-1 px-3 py-2 border border-gray-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700 text-gray-900 dark:text-slate-50 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          />
          <button
            type="button"
            onClick={handleAddCcEmail}
            className="px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 focus:ring-2 focus:ring-gray-500 focus:ring-offset-2"
          >
            Agregar
          </button>
        </div>
        {ccRecipients.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-2">
            {ccRecipients.map((email) => (
              <div
                key={email}
                className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-sm bg-gray-100 dark:bg-slate-600 text-gray-700 dark:text-slate-300 border border-gray-300 dark:border-slate-500"
              >
                <span>{email}</span>
                <button
                  type="button"
                  onClick={() => handleRemoveCc(email)}
                  className="text-gray-500 hover:text-gray-700 dark:text-slate-400 dark:hover:text-slate-200"
                >
                  <XMarkIcon className="h-4 w-4" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Asunto */}
      <div>
        <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
          Asunto <span className="text-red-500">*</span>
        </label>
        <input
          type="text"
          value={subject}
          onChange={(e) => setSubject(e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700 text-gray-900 dark:text-slate-50 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          placeholder="Asunto del email"
        />
      </div>

      {/* Mensajes de error y éxito */}
      {error && (
        <div className="bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg p-4 flex items-start gap-3">
          <InformationCircleIcon className="h-5 w-5 text-red-600 dark:text-red-400 flex-shrink-0 mt-0.5" />
          <p className="text-sm text-red-700 dark:text-red-400">{error}</p>
        </div>
      )}

      {success && (
        <div className="bg-green-50 dark:bg-green-900/30 border border-green-200 dark:border-green-800 rounded-lg p-4 flex items-start gap-3">
          <CheckCircleIcon className="h-5 w-5 text-green-600 dark:text-green-400 flex-shrink-0 mt-0.5" />
          <p className="text-sm text-green-700 dark:text-green-400">{success}</p>
        </div>
      )}

      {/* Botón de envío */}
      <div>
        <button
          type="button"
          onClick={handleSendEmail}
          disabled={loading || (selectedRecipients.length === 0 && additionalRecipients.length === 0) || !subject.trim()}
          className="w-full px-4 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-primary-600"
        >
          {loading ? 'Enviando...' : 'Enviar resumen por email'}
        </button>
      </div>
    </div>
  );
};
