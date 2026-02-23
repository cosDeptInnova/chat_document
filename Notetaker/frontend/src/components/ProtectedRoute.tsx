import { useEffect } from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const COSMOS_LOGIN_URL = import.meta.env.VITE_COSMOS_LOGIN_URL || 'https://cosmos.cosgs.int';

interface ProtectedRouteProps {
  children: React.ReactNode;
  requireAdmin?: boolean;
}

export const ProtectedRoute: React.FC<ProtectedRouteProps> = ({ children, requireAdmin = false }) => {
  const { isAuthenticated, user, loading } = useAuth();

  useEffect(() => {
    if (loading || isAuthenticated) return;
    const currentPath = window.location.pathname + window.location.search;
    if (currentPath !== '/login' && currentPath !== '/sso-callback' && currentPath !== '/') {
      localStorage.setItem('redirect_after_login', currentPath);
    }
    const returnTo = encodeURIComponent(window.location.origin + '/sso-callback');
    const redirectUrl = COSMOS_LOGIN_URL.includes('?')
      ? `${COSMOS_LOGIN_URL}&return_to=${returnTo}`
      : `${COSMOS_LOGIN_URL}?return_to=${returnTo}`;
    window.location.href = redirectUrl;
  }, [loading, isAuthenticated]);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600" />
      </div>
    );
  }

  if (requireAdmin && (!user || !user.is_admin)) {
    return <Navigate to="/dashboard" replace />;
  }

  return <>{children}</>;
};

