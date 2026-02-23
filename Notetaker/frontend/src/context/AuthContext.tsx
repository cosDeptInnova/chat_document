import React, { createContext, useContext, useState, useEffect, useRef, useCallback } from 'react';
import type { ReactNode } from 'react';
import { authApi } from '../services/api';
import type { User } from '../types';
import { isTokenExpired } from '../utils/tokenUtils';

interface AuthContextType {
  user: User | null;
  loading: boolean;
  login: (email: string, password: string, displayName?: string, rememberMe?: boolean) => Promise<void>;
  setSessionFromSSO: (user: User) => void;
  logout: () => void;
  refreshUser: () => Promise<void>;
  isAuthenticated: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const intervalRef = useRef<number | null>(null);

  // Función para limpiar sesión cuando el token expira
  const clearExpiredSession = () => {
    console.log('Token expirado, limpiando sesión');
    localStorage.removeItem('auth_token');
    localStorage.removeItem('user_email');
    localStorage.removeItem('user');
    setUser(null);
  };

  // Función para verificar si el token está expirado y limpiar si es necesario
  const checkTokenExpiration = useCallback((): boolean => {
    const token = localStorage.getItem('auth_token');
    if (!token) {
      // Si no hay token pero hay usuario en estado, limpiar
      if (user) {
        clearExpiredSession();
      }
      return false;
    }

    if (isTokenExpired(token)) {
      clearExpiredSession();
      return false;
    }

    return true;
  }, [user]);

  // Refrescar información del usuario desde el backend
  const refreshUserFromBackend = async (email: string) => {
    try {
      // Verificar token antes de hacer la petición
      if (!checkTokenExpiration()) {
        return;
      }

      const userData = await authApi.getCurrentUser(email);
      setUser(userData);
      localStorage.setItem('user', JSON.stringify(userData));
    } catch (error) {
      console.error('Error refrescando usuario:', error);
      // Si el error es 401 (token expirado), limpiar sesión
      if ((error as any)?.response?.status === 401) {
        clearExpiredSession();
      }
    }
  };

  // Función pública para refrescar el usuario
  const refreshUser = async () => {
    if (user?.email) {
      await refreshUserFromBackend(user.email);
    }
  };

  // Efecto inicial: verificar token y restaurar sesión
  useEffect(() => {
    // Verificar token antes de restaurar sesión
    const token = localStorage.getItem('auth_token');
    const storedUser = localStorage.getItem('user');
    
    if (token && storedUser) {
      // Verificar si el token está realmente expirado
      if (isTokenExpired(token)) {
        // Token realmente expirado, limpiar y requerir login
        clearExpiredSession();
        setLoading(false);
        return;
      }
      
      try {
        const parsedUser = JSON.parse(storedUser);
        setUser(parsedUser);
        // Refrescar información del usuario desde el backend
        if (parsedUser.email) {
          refreshUserFromBackend(parsedUser.email);
        }
      } catch (e) {
        console.error('Error parseando usuario del localStorage:', e);
        clearExpiredSession();
      }
    } else if (storedUser && !token) {
      // Hay usuario pero no token, limpiar (sesión antigua sin token)
      clearExpiredSession();
    }
    
    setLoading(false);
  }, []);

  // Verificación periódica de expiración del token (cada 5 minutos)
  useEffect(() => {
    if (!user) {
      // Si no hay usuario, no hay nada que verificar
      return;
    }

    // Verificar inmediatamente
    checkTokenExpiration();

    // Configurar verificación periódica cada 5 minutos
    intervalRef.current = setInterval(() => {
      const isValid = checkTokenExpiration();
      if (!isValid) {
        console.log('Token expirado durante verificación periódica');
      }
    }, 5 * 60 * 1000); // 5 minutos

    // Limpiar intervalo al desmontar
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [user, checkTokenExpiration]);

  // Verificar token cuando la pestaña vuelve a estar visible
  useEffect(() => {
    if (!user) {
      return;
    }

    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        // Cuando la pestaña vuelve a estar visible, verificar el token
        console.log('Pestaña visible, verificando expiración del token');
        checkTokenExpiration();
      }
    };

    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [user, checkTokenExpiration]);

  const login = async (email: string, password: string, displayName?: string, rememberMe: boolean = false) => {
    try {
      const userData = await authApi.login(email, password, displayName, rememberMe);
      setUser(userData);
      localStorage.setItem('user', JSON.stringify(userData));
    } catch (error) {
      console.error('Error en login:', error);
      throw error;
    }
  };

  const setSessionFromSSO = (userData: User) => {
    setUser(userData);
    // Persistir usuario (incluye display_name de Cosmos) para que Topbar y demas usen el nombre correcto
    localStorage.setItem('user', JSON.stringify(userData));
  };

  const logout = () => {
    setUser(null);
    authApi.logout();
    // Limpiar intervalo de verificación
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
  };

  return (
    <AuthContext.Provider
      value={{
        user,
        loading,
        login,
        setSessionFromSSO,
        logout,
        refreshUser,
        isAuthenticated: !!user,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

