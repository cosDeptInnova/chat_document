import { useEffect } from 'react';

const COSMOS_LOGIN_URL = import.meta.env.VITE_COSMOS_LOGIN_URL || 'https://cosmos.cosgs.int';

/**
 * Redirige al login de Cosmos (SSO). Se usa en la ruta /login para que
 * quien intente entrar por URL al login de Notetaker vaya a Cosmos.
 */
export const RedirectToCosmosLogin: React.FC = () => {
  useEffect(() => {
    const returnTo = encodeURIComponent(window.location.origin + '/sso-callback');
    const url = COSMOS_LOGIN_URL.includes('?')
      ? `${COSMOS_LOGIN_URL}&return_to=${returnTo}`
      : `${COSMOS_LOGIN_URL}?return_to=${returnTo}`;
    window.location.href = url;
  }, []);

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100 dark:bg-slate-900">
      <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600" />
    </div>
  );
};
