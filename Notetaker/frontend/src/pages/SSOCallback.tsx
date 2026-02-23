import { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { authApi } from '../services/api';

/**
 * Pagina que recibe los datos de Cosmos (email, display_name) por query,
 * llama a POST /api/auth/sso-login (get-or-create), establece sesion y redirige al dashboard.
 */
export const SSOCallback: React.FC = () => {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { setSessionFromSSO } = useAuth();
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const email = searchParams.get('email');
    // Soporta display_name (underscore) y display-name (guion) por compatibilidad con Cosmos
    const displayNameRaw = searchParams.get('display_name') || searchParams.get('display-name') || '';
    const displayName = displayNameRaw.trim() || undefined;
    const cosmosToken = searchParams.get('token') || searchParams.get('cosmos_token') || undefined;

    if (!email || !email.trim()) {
      setError('Falta el parametro email en la URL.');
      return;
    }

    let cancelled = false;

    const run = async () => {
      try {
        const user = await authApi.ssoLogin(email.trim(), displayName, cosmosToken);
        if (cancelled) return;
        setSessionFromSSO(user);
        const redirectPath = localStorage.getItem('redirect_after_login');
        if (redirectPath && redirectPath !== '/login' && redirectPath !== '/sso-callback') {
          localStorage.removeItem('redirect_after_login');
          navigate(redirectPath, { replace: true });
        } else {
          navigate('/dashboard', { replace: true });
        }
      } catch (err: unknown) {
        if (cancelled) return;
        const message = err instanceof Error ? err.message : 'Error al iniciar sesion con SSO.';
        setError(message);
      }
    };

    run();
    return () => {
      cancelled = true;
    };
  }, [searchParams, navigate, setSessionFromSSO]);

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-100 dark:bg-slate-900 px-4">
        <div className="max-w-md w-full bg-white dark:bg-slate-800 rounded-lg shadow p-6">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-slate-50 mb-2">Error en SSO</h2>
          <p className="text-gray-700 dark:text-slate-300 mb-4">{error}</p>
          <button
            type="button"
            onClick={() => navigate('/dashboard', { replace: true })}
            className="w-full py-2 px-4 bg-primary-600 text-white rounded-lg hover:bg-primary-700"
          >
            Ir al inicio
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100 dark:bg-slate-900">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600" />
    </div>
  );
};
