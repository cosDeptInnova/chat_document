import { useState, useEffect } from 'react';
import type { User } from '../types';
import { usersApi } from '../services/api';
import { useAuth } from '../context/AuthContext';
import { XMarkIcon, PlusIcon } from '@heroicons/react/24/outline';
import { ConfirmDialog } from '../components/ConfirmDialog';

interface EditUserModalProps {
  user: User;
  onClose: () => void;
  onSave: (updates: Partial<User>) => Promise<void>;
}

const EditUserModal: React.FC<EditUserModalProps> = ({ user, onClose, onSave }) => {
  const [displayName, setDisplayName] = useState(user.display_name || '');
  const [email, setEmail] = useState(user.email);
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    try {
      await onSave({
        display_name: displayName,
        email: email,
      });
      onClose();
    } catch (error) {
      console.error('Error guardando usuario:', error);
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-slate-800 rounded-lg shadow-xl max-w-md w-full mx-4 transition-colors">
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-slate-700">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-slate-50">Editar Usuario</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-500 dark:hover:text-slate-300"
          >
            <XMarkIcon className="h-6 w-6" />
          </button>
        </div>
        
        <div className="p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1">
              Nombre a mostrar
            </label>
            <input
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 text-gray-900 dark:text-slate-50 focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1">
              Email
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 text-gray-900 dark:text-slate-50 focus:outline-none focus:ring-2 focus:ring-primary-500"
            />
          </div>
          
        </div>
        
        <div className="flex justify-end gap-3 p-6 border-t border-gray-200 dark:border-slate-700">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-slate-300 bg-gray-100 dark:bg-slate-700 rounded-md hover:bg-gray-200 dark:hover:bg-slate-600"
          >
            Cancelar
          </button>
          <button
            onClick={handleSave}
            disabled={saving}
            className="px-4 py-2 text-sm font-medium text-white bg-primary-600 rounded-md hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {saving ? 'Guardando...' : 'Guardar'}
          </button>
        </div>
      </div>
    </div>
  );
};

interface CreateUserModalProps {
  onClose: () => void;
  onCreate: (email: string, password: string, displayName?: string) => Promise<void>;
}

const CreateUserModal: React.FC<CreateUserModalProps> = ({ onClose, onCreate }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [creating, setCreating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleCreate = async () => {
    if (!email || !password) {
      setError('Email y contraseña son obligatorios');
      return;
    }

    if (password.length < 8) {
      setError('La contraseña debe tener al menos 8 caracteres');
      return;
    }

    setCreating(true);
    setError(null);
    try {
      await onCreate(email, password, displayName || undefined);
      onClose();
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Error al crear usuario');
    } finally {
      setCreating(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white dark:bg-slate-800 rounded-lg shadow-xl max-w-md w-full mx-4 transition-colors">
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-slate-700">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-slate-50">Crear Usuario</h2>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-500 dark:hover:text-slate-300"
          >
            <XMarkIcon className="h-6 w-6" />
          </button>
        </div>
        
        <div className="p-6 space-y-4">
          {error && (
            <div className="p-3 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-400 text-sm">
              {error}
            </div>
          )}

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1">
              Email <span className="text-red-500">*</span>
            </label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 text-gray-900 dark:text-slate-50 focus:outline-none focus:ring-2 focus:ring-primary-500"
              placeholder="usuario@ejemplo.com"
              required
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1">
              Contraseña (de 1 uso) <span className="text-red-500">*</span>
            </label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 text-gray-900 dark:text-slate-50 focus:outline-none focus:ring-2 focus:ring-primary-500"
              placeholder="Mínimo 8 caracteres"
              required
            />
            <p className="mt-1 text-xs text-gray-500 dark:text-slate-400">
              El usuario deberá cambiar esta contraseña en el primer login
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-1">
              Nombre a mostrar (opcional)
            </label>
            <input
              type="text"
              value={displayName}
              onChange={(e) => setDisplayName(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-slate-600 rounded-md bg-white dark:bg-slate-700 text-gray-900 dark:text-slate-50 focus:outline-none focus:ring-2 focus:ring-primary-500"
              placeholder="Nombre del usuario"
            />
          </div>
        </div>
        
        <div className="flex justify-end gap-3 p-6 border-t border-gray-200 dark:border-slate-700">
          <button
            onClick={onClose}
            className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-slate-300 bg-gray-100 dark:bg-slate-700 rounded-md hover:bg-gray-200 dark:hover:bg-slate-600"
          >
            Cancelar
          </button>
          <button
            onClick={handleCreate}
            disabled={creating || !email || !password}
            className="px-4 py-2 text-sm font-medium text-white bg-primary-600 rounded-md hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {creating ? 'Creando...' : 'Crear Usuario'}
          </button>
        </div>
      </div>
    </div>
  );
};

export const AdminUsers: React.FC = () => {
  const [users, setUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editingUser, setEditingUser] = useState<User | null>(null);
  const [creatingUser, setCreatingUser] = useState(false);
  const [licenseChangeConfirm, setLicenseChangeConfirm] = useState<{ userId: string; newLicense: string; currentLicense: string; selectElement: HTMLSelectElement | null } | null>(null);
  const [syncingCalendarUserId, setSyncingCalendarUserId] = useState<string | null>(null);
  const { user } = useAuth();

  useEffect(() => {
    const loadUsers = async () => {
      if (!user?.email) {
        setError('No se pudo obtener el email del usuario');
        setLoading(false);
        return;
      }

      try {
        const usersList = await usersApi.list(user.email);
        // Ordenar usuarios: online primero, luego offline
        const sortedUsers = [...usersList].sort((a, b) => {
          // Si ambos tienen el mismo estado, mantener orden original
          if (a.is_online === b.is_online) return 0;
          // Online primero: si a está online, va primero (-1), si no, va después (1)
          return a.is_online ? -1 : 1;
        });
        setUsers(sortedUsers);
        setError(null);
      } catch (err: any) {
        console.error('Error cargando usuarios:', err);
        setError(err.response?.data?.detail || 'Error al cargar usuarios');
      } finally {
        setLoading(false);
      }
    };

    loadUsers();

    // Actualizar lista de usuarios cada 10 segundos para ver cambios de estado online/offline
    const interval = setInterval(loadUsers, 10000);

    return () => clearInterval(interval);
  }, [user]);

  const handleLicenseChange = async (userId: string, newLicense: string) => {
    if (!user?.email) return;
    
    try {
      console.log('🔄 Cambiando licencia de usuario', userId, 'a', newLicense);
      const updatedUser = await usersApi.update(userId, user.email, { license: newLicense });
      console.log('✅ Usuario actualizado:', updatedUser);
      // Recargar lista de usuarios
      const usersList = await usersApi.list(user.email);
      // Ordenar usuarios: online primero, luego offline
      const sortedUsers = [...usersList].sort((a, b) => {
        if (a.is_online === b.is_online) return 0;
        return b.is_online ? 1 : -1;
      });
      setUsers(sortedUsers);
      setError(null);
    } catch (err: any) {
      console.error('❌ Error actualizando licencia:', err);
      setError(err.response?.data?.detail || 'Error al actualizar licencia');
    }
  };

  const handleEditUser = async (updates: Partial<User>) => {
    if (!user?.email || !editingUser) return;
    
    try {
      await usersApi.update(editingUser.id, user.email, {
        display_name: updates.display_name ?? undefined,
        email: updates.email,
      });
      // Recargar lista de usuarios
      const usersList = await usersApi.list(user.email);
      setUsers(usersList);
      setEditingUser(null);
      setError(null);
    } catch (err: any) {
      console.error('Error actualizando usuario:', err);
      setError(err.response?.data?.detail || 'Error al actualizar usuario');
      throw err;
    }
  };

  const handleCreateUser = async (email: string, password: string, displayName?: string) => {
    if (!user?.email) return;
    
    try {
      await usersApi.create(user.email, email, password, displayName);
      // Recargar lista de usuarios
      const usersList = await usersApi.list(user.email);
      setUsers(usersList);
      setCreatingUser(false);
      setError(null);
    } catch (err: any) {
      console.error('Error creando usuario:', err);
      setError(err.response?.data?.detail || 'Error al crear usuario');
      throw err;
    }
  };


  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-slate-50">Gestión de usuarios</h1>
          <p className="mt-2 text-gray-600 dark:text-slate-400">Administra usuarios y licencias</p>
        </div>
        <button
          onClick={() => setCreatingUser(true)}
          className="flex items-center gap-2 px-4 py-2 bg-primary-600 text-white rounded-lg font-semibold hover:bg-primary-700 transition-colors"
        >
          <PlusIcon className="h-5 w-5" />
          Crear Usuario
        </button>
      </div>

      {error && (
        <div className="bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <p className="text-sm text-red-800 dark:text-red-400">
            <strong>Error:</strong> {error}
          </p>
        </div>
      )}

      <div className="bg-white dark:bg-slate-800 rounded-lg shadow overflow-hidden transition-colors">
        <table className="min-w-full divide-y divide-gray-200 dark:divide-slate-700">
          <thead className="bg-gray-50 dark:bg-slate-700">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-slate-300 uppercase tracking-wider">
                Usuario
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-slate-300 uppercase tracking-wider">
                Estado
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-slate-300 uppercase tracking-wider">
                Rol
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-slate-300 uppercase tracking-wider">
                Licencia
              </th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 dark:text-slate-300 uppercase tracking-wider">
                Acciones
              </th>
            </tr>
          </thead>
          <tbody className="bg-white dark:bg-slate-800 divide-y divide-gray-200 dark:divide-slate-700">
            {users.map((userItem) => (
              <tr key={userItem.id}>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div>
                    <div className="text-sm font-medium text-gray-900 dark:text-slate-50">
                      {userItem.display_name || userItem.email}
                    </div>
                    <div className="text-sm text-gray-500 dark:text-slate-400">{userItem.email}</div>
                    {userItem.outlook_webhook_expired && (
                      <div className="mt-1 text-xs font-semibold text-red-600 dark:text-red-400" title={userItem.outlook_webhook_status ?? undefined}>
                        {userItem.outlook_webhook_status === 'no_configurado'
                          ? 'Webhook de Outlook no configurado - Resincronizar calendario'
                          : 'Webhook de Outlook expirado - Resincronizar calendario'}
                      </div>
                    )}
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <div className="flex items-center gap-2">
                    <div
                      className={`h-3 w-3 rounded-full ${
                        userItem.is_online
                          ? 'bg-green-500 animate-pulse'
                          : 'bg-red-500'
                      }`}
                      title={userItem.is_online ? 'Online' : 'Offline'}
                    />
                    <span
                      className={`text-sm font-medium ${
                        userItem.is_online
                          ? 'text-green-600 dark:text-green-400'
                          : 'text-red-600 dark:text-red-400'
                      }`}
                    >
                      {userItem.is_online ? 'Online' : 'Offline'}
                    </span>
                  </div>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <span
                    className={`px-2 py-1 text-xs font-semibold rounded ${
                      userItem.is_admin
                        ? 'bg-purple-100 dark:bg-purple-900/30 text-purple-800 dark:text-purple-400'
                        : 'bg-gray-100 dark:bg-slate-700 text-gray-800 dark:text-slate-300'
                    }`}
                  >
                    {userItem.is_admin ? 'Admin' : 'Usuario'}
                  </span>
                </td>
                <td className="px-6 py-4 whitespace-nowrap">
                  <select
                    key={`license-${userItem.id}-${userItem.license}`}
                    className="text-sm border border-gray-300 dark:border-slate-600 rounded px-2 py-1 bg-white dark:bg-slate-700 text-gray-900 dark:text-slate-50 focus:outline-none focus:ring-2 focus:ring-primary-500"
                    value={userItem.license || 'basic'}
                    onChange={async (e) => {
                      const newLicense = e.target.value;
                      const currentLicense = userItem.license || 'basic';
                      
                      if (newLicense === currentLicense) {
                        return;
                      }
                      
                      setLicenseChangeConfirm({
                        userId: userItem.id,
                        newLicense: newLicense,
                        currentLicense: currentLicense,
                        selectElement: e.target,
                      });
                      // Resetear el select mientras se muestra el diálogo
                      e.target.value = currentLicense;
                    }}
                  >
                    <option value="basic">Basic</option>
                    <option value="advanced">Advanced</option>
                    <option value="pro">Pro</option>
                  </select>
                </td>
                <td className="px-6 py-4 whitespace-nowrap text-sm">
                  <div className="flex items-center gap-3">
                    <button
                      onClick={() => setEditingUser(userItem)}
                      className="text-primary-600 dark:text-primary-400 hover:text-primary-900 dark:hover:text-primary-300 font-medium"
                    >
                      Editar
                    </button>
                    {userItem.outlook_webhook_expired && (
                      <button
                        type="button"
                        onClick={async () => {
                          if (!user?.email || syncingCalendarUserId) return;
                          setSyncingCalendarUserId(userItem.id);
                          try {
                            await usersApi.syncUserCalendar(user.email, userItem.email);
                            const list = await usersApi.list(user.email, { skipCache: true });
                            setUsers(list);
                          } catch (err: any) {
                            setError(err?.response?.data?.detail || err?.message || 'Error sincronizando calendario');
                          } finally {
                            setSyncingCalendarUserId(null);
                          }
                        }}
                        disabled={!!syncingCalendarUserId}
                        className="flex items-center gap-1 text-blue-600 dark:text-blue-400 hover:text-blue-900 dark:hover:text-blue-300 font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                        title="Sincronizar calendario de este usuario (como si lo hiciera el en Ajustes)"
                      >
                        {syncingCalendarUserId === userItem.id ? 'Sincronizando...' : 'Sincronizar calendario'}
                      </button>
                    )}
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      {users.length === 0 && !loading && !error && (
        <div className="bg-gray-50 dark:bg-slate-700 rounded-lg p-4 text-center">
          <p className="text-sm text-gray-600 dark:text-slate-400">
            No hay usuarios registrados en el sistema.
          </p>
        </div>
      )}

      {editingUser && (
        <EditUserModal
          user={editingUser}
          onClose={() => setEditingUser(null)}
          onSave={handleEditUser}
        />
      )}

      {creatingUser && (
        <CreateUserModal
          onClose={() => setCreatingUser(false)}
          onCreate={handleCreateUser}
        />
      )}

      {licenseChangeConfirm && (
        <ConfirmDialog
          isOpen={!!licenseChangeConfirm}
          onClose={() => {
            // Restaurar el valor del select si se cancela
            if (licenseChangeConfirm.selectElement) {
              licenseChangeConfirm.selectElement.value = licenseChangeConfirm.currentLicense;
            }
            setLicenseChangeConfirm(null);
          }}
          onConfirm={async () => {
            if (!licenseChangeConfirm) return;
            await handleLicenseChange(licenseChangeConfirm.userId, licenseChangeConfirm.newLicense);
            setLicenseChangeConfirm(null);
          }}
          title="Cambiar licencia"
          message={`¿Cambiar licencia de este usuario de ${licenseChangeConfirm.currentLicense} a ${licenseChangeConfirm.newLicense}?`}
          confirmText="Sí, cambiar"
          cancelText="Cancelar"
          confirmButtonColor="blue"
        />
      )}
    </div>
  );
};
