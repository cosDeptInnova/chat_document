import { useState, useEffect } from 'react';
import { useAuth } from '../../context/AuthContext';
import { useNavigate } from 'react-router-dom';
import { ArrowRightOnRectangleIcon } from '@heroicons/react/24/outline';
import { ThemeToggle } from '../ThemeToggle';
import { securityApi } from '../../services/api';

const DAYS_WARNING = 30;

export const Topbar: React.FC = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [sslDaysLeft, setSslDaysLeft] = useState<number | null>(null);

  useEffect(() => {
    if (!user?.is_admin) {
      setSslDaysLeft(null);
      return;
    }
    let cancelled = false;
    securityApi.getSSLCertificate()
      .then((res) => {
        if (cancelled || !res?.certificate?.valid_to) return;
        const expiry = new Date(res.certificate.valid_to);
        const now = new Date();
        const days = Math.ceil((expiry.getTime() - now.getTime()) / (1000 * 60 * 60 * 24));
        if (days <= DAYS_WARNING) setSslDaysLeft(days);
        else setSslDaysLeft(null);
      })
      .catch(() => setSslDaysLeft(null));
    return () => { cancelled = true; };
  }, [user?.is_admin, user?.email]);

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <div className="h-16 bg-white dark:bg-slate-800 border-b border-gray-200 dark:border-slate-700 flex items-center justify-between px-6 transition-colors">
      {/* Lado izquierdo - Toggle de tema */}
      <div className="flex items-center">
        <ThemeToggle />
      </div>

      {/* Centro - Aviso SSL próximo a caducar (solo administradores) */}
      <div className="flex-1 flex items-center justify-center min-w-0 px-4">
        {user?.is_admin && sslDaysLeft !== null && sslDaysLeft <= DAYS_WARNING && (
          <p className="text-sm font-medium text-amber-700 dark:text-amber-300 bg-amber-100 dark:bg-amber-900/30 px-3 py-1.5 rounded truncate max-w-full">
            El certificado SSL está próximo a caducar. Días restantes: {Math.max(0, sslDaysLeft)}
          </p>
        )}
      </div>

      {/* Lado derecho - Usuario, etiquetas y cerrar sesión */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-3">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-slate-50">
            {user?.display_name || user?.email}
          </h2>
          {user?.is_admin && (
            <span className="px-2 py-1 text-xs font-semibold text-primary-700 dark:text-primary-400 bg-primary-100 dark:bg-primary-900/30 rounded">
              Admin
            </span>
          )}
          {user?.license && (
            <span className="px-2 py-1 text-xs font-semibold text-gray-700 dark:text-slate-300 bg-gray-100 dark:bg-slate-700 rounded capitalize">
              {user.license}
            </span>
          )}
        </div>
        <button
          onClick={handleLogout}
          className="flex items-center px-4 py-2 text-sm font-medium text-gray-700 dark:text-slate-300 hover:bg-gray-50 dark:hover:bg-slate-700 rounded-lg transition-colors"
        >
          <ArrowRightOnRectangleIcon className="h-5 w-5 mr-2" />
          Cerrar sesión
        </button>
      </div>
    </div>
  );
};

