import React from 'react';

export interface IntegrationCardProps {
  name: string;
  logo: string | React.ReactNode;
  description: string;
  isConnected: boolean;
  connectedAt?: string | null;
  onConnect: () => void;
  onDisconnect: () => void;
  loading?: boolean;
}

export const IntegrationCard: React.FC<IntegrationCardProps> = ({
  name,
  logo,
  description,
  isConnected,
  connectedAt,
  onConnect,
  onDisconnect,
  loading = false,
}) => {
  return (
    <div className="bg-white dark:bg-slate-800 rounded-lg border border-gray-200 dark:border-slate-700 p-6 transition-all hover:shadow-md">
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center gap-4">
          <div className="flex-shrink-0">
            {typeof logo === 'string' ? (
              <img src={logo} alt={name} className="w-12 h-12 object-contain" />
            ) : (
              <div className="w-12 h-12 flex items-center justify-center text-2xl">
                {logo}
              </div>
            )}
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-slate-50">
              {name}
            </h3>
            {isConnected && connectedAt && (
              <p className="text-xs text-gray-500 dark:text-slate-400 mt-1">
                Conectado el {new Date(connectedAt).toLocaleDateString('es-ES', {
                  year: 'numeric',
                  month: 'long',
                  day: 'numeric',
                })}
              </p>
            )}
          </div>
        </div>
        {isConnected && (
          <div className="flex-shrink-0">
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400">
              <svg
                className="w-4 h-4 mr-1"
                fill="currentColor"
                viewBox="0 0 20 20"
              >
                <path
                  fillRule="evenodd"
                  d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                  clipRule="evenodd"
                />
              </svg>
              Conectado
            </span>
          </div>
        )}
      </div>

      <p className="text-sm text-gray-600 dark:text-slate-400 mb-4">
        {description}
      </p>

      <div className="flex gap-2">
        {isConnected ? (
          <button
            onClick={onDisconnect}
            disabled={loading}
            className="flex-1 px-4 py-2 text-sm font-medium text-red-700 dark:text-red-400 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg hover:bg-red-100 dark:hover:bg-red-900/50 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Desconectando...' : 'Desconectar'}
          </button>
        ) : (
          <button
            onClick={onConnect}
            disabled={loading}
            className="flex-1 px-4 py-2 text-sm font-medium text-white bg-primary-600 dark:bg-primary-500 rounded-lg hover:bg-primary-700 dark:hover:bg-primary-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? 'Conectando...' : 'Conectar'}
          </button>
        )}
      </div>
    </div>
  );
};

