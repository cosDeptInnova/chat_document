import type { ReactNode } from 'react';
import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { Topbar } from './Topbar';
import { useAuth } from '../../context/AuthContext';
import { useHeartbeat } from '../../hooks/useHeartbeat';

interface MainLayoutProps {
  children: ReactNode;
}

export const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
  const location = useLocation();
  const { refreshUser, user } = useAuth();

  // Enviar heartbeats periódicos para indicar que el usuario está online
  useHeartbeat(user?.email || null, !!user?.email, 15000); // Cada 15 segundos

  // Refrescar información del usuario cada vez que cambia la ruta
  useEffect(() => {
    if (user?.email) {
      refreshUser();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.pathname]);

  return (
    <div className="flex h-screen bg-white dark:bg-slate-900 transition-colors">
      <Sidebar />
      <div className="flex-1 flex flex-col">
        <Topbar />
        <main className="flex-1 overflow-y-auto p-6 bg-white dark:bg-slate-900">
          {children}
        </main>
      </div>
    </div>
  );
};

