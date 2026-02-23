import { useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import {
  HomeIcon,
  CalendarIcon,
  ClockIcon,
  ChartBarIcon,
  Cog6ToothIcon,
  UserGroupIcon,
  FolderIcon,
  ShieldCheckIcon,
  Bars3Icon,
} from '@heroicons/react/24/outline';
import { useAuth } from '../../context/AuthContext';
import { useTheme } from '../../context/ThemeContext';

const SIDEBAR_COLLAPSED_KEY = 'notetaker-sidebar-collapsed';

const navigation = [
  { name: 'Dashboard', href: '/dashboard', icon: HomeIcon },
  { name: 'Próximas', href: '/meetings/upcoming', icon: CalendarIcon },
  { name: 'Pasadas', href: '/meetings/past', icon: ClockIcon },
  { name: 'Analytics', href: '/analytics', icon: ChartBarIcon },
  { name: 'Configuración', href: '/settings', icon: Cog6ToothIcon },
];

const adminNavigation = [
  { name: 'Usuarios', href: '/admin/users', icon: UserGroupIcon },
  { name: 'Reuniones', href: '/admin/meetings', icon: FolderIcon },
  { name: 'Ajustes de Seguridad', href: '/admin/security', icon: ShieldCheckIcon },
];

export const Sidebar: React.FC = () => {
  const location = useLocation();
  const { user } = useAuth();
  const { theme } = useTheme();
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem(SIDEBAR_COLLAPSED_KEY) ?? 'false');
    } catch {
      return false;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(SIDEBAR_COLLAPSED_KEY, JSON.stringify(collapsed));
    } catch {
      // ignore
    }
  }, [collapsed]);

  const toggleCollapsed = () => setCollapsed((prev: boolean) => !prev);

  return (
    <div
      className={`flex flex-col bg-gray-200 dark:bg-slate-800 border-r border-gray-200 dark:border-slate-700 h-full transition-all duration-200 overflow-x-hidden ${
        collapsed ? 'w-16' : 'w-64'
      }`}
    >
      <div
        className={`flex flex-col border-b border-gray-200 dark:border-slate-700 transition-all duration-200 ${
          collapsed ? 'px-2 pt-6 pb-4 gap-3' : 'pl-3 pt-6 pb-4 gap-3'
        }`}
      >
        {!collapsed ? (
          <>
            <div className="flex items-center gap-2 w-full flex-nowrap min-w-0 pr-6">
              <img
                src="/cosmos-logo.png"
                alt="COS Logo"
                className="h-10 w-auto min-w-0 shrink"
              />
              <button
                type="button"
                onClick={toggleCollapsed}
                className="shrink-0 w-10 h-10 rounded-full flex items-center justify-center bg-gray-300 dark:bg-slate-500 text-gray-700 dark:text-slate-200 hover:bg-gray-500 hover:dark:bg-slate-400 focus:outline-none focus:ring-2 focus:ring-primary-500 transition-colors"
                title="Compactar menú"
                aria-label="Compactar menú"
              >
                <Bars3Icon className="h-5 w-5" />
              </button>
            </div>
            {theme === 'light' ? (
              <img
                src="/notetaker-light.png"
                alt="Notetaker Logo"
                className="h-16 w-auto"
              />
            ) : (
              <img
                src="/notetaker-dark.png"
                alt="Notetaker Logo"
                className="h-16 w-auto"
              />
            )}
          </>
        ) : (
          <>
            <div className="flex items-center justify-center w-full">
              <button
                type="button"
                onClick={toggleCollapsed}
                className="shrink-0 w-10 h-10 rounded-full flex items-center justify-center bg-gray-300 dark:bg-slate-500 text-gray-700 dark:text-slate-200 hover:bg-gray-500 hover:dark:bg-slate-400 focus:outline-none focus:ring-2 focus:ring-primary-500 transition-colors"
                title="Expandir menú"
                aria-label="Expandir menú"
              >
                <Bars3Icon className="h-5 w-5" />
              </button>
            </div>
            {/* Espacio invisible para mantener la misma altura que el logo de Notetaker */}
            <div className="h-16 w-full" aria-hidden="true" />
          </>
        )}
      </div>

      <nav className={`flex-1 py-6 space-y-1 ${collapsed ? 'px-2' : 'px-4'}`}>
        {navigation.map((item) => {
          const isActive = location.pathname === item.href;
          const linkContent = (
            <Link
              key={item.name}
              to={item.href}
              title={collapsed ? item.name : undefined}
              className={`flex items-center text-sm font-medium rounded-lg transition-colors ${
                collapsed
                  ? 'justify-center p-3'
                  : 'px-4 py-3'
              } ${
                isActive
                  ? 'bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-400'
                  : 'text-gray-700 dark:text-slate-300 hover:bg-gray-50 dark:hover:bg-slate-700'
              }`}
            >
              <item.icon className={collapsed ? 'h-5 w-5' : 'mr-3 h-5 w-5'} />
              {!collapsed && item.name}
            </Link>
          );
          return linkContent;
        })}

        {user?.is_admin && (
          <>
            {!collapsed && (
              <div className="pt-4 mt-4 border-t border-gray-200 dark:border-slate-700">
                <p className="px-4 text-xs font-semibold text-gray-500 dark:text-slate-400 uppercase tracking-wider">
                  Administración
                </p>
              </div>
            )}
            {collapsed && <div className="pt-4 mt-4 border-t border-gray-200 dark:border-slate-700" />}
            {adminNavigation.map((item) => {
              const isActive = location.pathname === item.href;
              return (
                <Link
                  key={item.name}
                  to={item.href}
                  title={collapsed ? item.name : undefined}
                  className={`flex items-center text-sm font-medium rounded-lg transition-colors ${
                    collapsed ? 'justify-center p-3' : 'px-4 py-3'
                  } ${
                    isActive
                      ? 'bg-primary-50 dark:bg-primary-900/30 text-primary-700 dark:text-primary-400'
                      : 'text-gray-700 dark:text-slate-300 hover:bg-gray-50 dark:hover:bg-slate-700'
                  }`}
                >
                  <item.icon className={collapsed ? 'h-5 w-5' : 'mr-3 h-5 w-5'} />
                  {!collapsed && item.name}
                </Link>
              );
            })}
          </>
        )}
      </nav>
    </div>
  );
};
