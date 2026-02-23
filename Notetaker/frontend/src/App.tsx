import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { AuthProvider, useAuth } from './context/AuthContext';
import { ThemeProvider } from './context/ThemeContext';
import { ProtectedRoute } from './components/ProtectedRoute';
import { MainLayout } from './components/Layout/MainLayout';
import { RedirectToCosmosLogin } from './pages/RedirectToCosmosLogin';
import { SSOCallback } from './pages/SSOCallback';
import { ForgotPassword } from './pages/ForgotPassword';
import { ResetPassword } from './pages/ResetPassword';
import { ChangePassword } from './pages/ChangePassword';
import { ForcePasswordChange } from './pages/ForcePasswordChange';
import { Dashboard } from './pages/Dashboard';
import { MeetingsList } from './pages/MeetingsList';
import { MeetingDetail } from './pages/MeetingDetail';
import { AdminUsers } from './pages/AdminUsers';
import { AdminMeetings } from './pages/AdminMeetings';
import { AdminSecurity } from './pages/AdminSecurity';
import { Analytics } from './pages/Analytics';
import { Settings } from './pages/Settings';

const AppRoutes = () => {
  const { isAuthenticated, loading, user } = useAuth();

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  // Si el usuario está autenticado y debe cambiar su contraseña, mostrar página de cambio forzado
  if (isAuthenticated && user?.must_change_password) {
    return (
      <Routes>
        <Route path="/force-password-change" element={<ForcePasswordChange />} />
        <Route path="*" element={<Navigate to="/force-password-change" replace />} />
      </Routes>
    );
  }

  return (
    <Routes>
      <Route path="/login" element={<RedirectToCosmosLogin />} />
      <Route path="/sso-callback" element={<SSOCallback />} />
      <Route path="/forgot-password" element={<ForgotPassword />} />
      <Route path="/reset-password" element={<ResetPassword />} />
      <Route path="/force-password-change" element={<ForcePasswordChange />} />

      <Route
        path="/"
        element={
          <ProtectedRoute>
            <MainLayout>
              <Navigate to="/dashboard" replace />
            </MainLayout>
          </ProtectedRoute>
        }
      />

      <Route
        path="/dashboard"
        element={
          <ProtectedRoute>
            <MainLayout>
              <Dashboard />
            </MainLayout>
          </ProtectedRoute>
        }
      />

      <Route
        path="/meetings/upcoming"
        element={
          <ProtectedRoute>
            <MainLayout>
              <MeetingsList type="upcoming" />
            </MainLayout>
          </ProtectedRoute>
        }
      />

      <Route
        path="/meetings/past"
        element={
          <ProtectedRoute>
            <MainLayout>
              <MeetingsList type="past" />
            </MainLayout>
          </ProtectedRoute>
        }
      />

      <Route
        path="/meetings/:meetingId"
        element={
          <ProtectedRoute>
            <MainLayout>
              <MeetingDetail />
            </MainLayout>
          </ProtectedRoute>
        }
      />

      <Route
        path="/analytics"
        element={
          <ProtectedRoute>
            <MainLayout>
              <Analytics />
            </MainLayout>
          </ProtectedRoute>
        }
      />

      <Route
        path="/settings"
        element={
          <ProtectedRoute>
            <MainLayout>
              <Settings />
            </MainLayout>
          </ProtectedRoute>
        }
      />

      <Route
        path="/settings/change-password"
        element={
          <ProtectedRoute>
            <MainLayout>
              <ChangePassword />
            </MainLayout>
          </ProtectedRoute>
        }
      />

      <Route
        path="/admin/users"
        element={
          <ProtectedRoute requireAdmin>
            <MainLayout>
              <AdminUsers />
            </MainLayout>
          </ProtectedRoute>
        }
      />

      <Route
        path="/admin/meetings"
        element={
          <ProtectedRoute requireAdmin>
            <MainLayout>
              <AdminMeetings />
            </MainLayout>
          </ProtectedRoute>
        }
      />

      <Route
        path="/admin/security"
        element={
          <ProtectedRoute requireAdmin>
            <MainLayout>
              <AdminSecurity />
            </MainLayout>
          </ProtectedRoute>
        }
      />

      <Route path="*" element={<Navigate to="/dashboard" replace />} />
    </Routes>
  );
};

function App() {
  return (
    <BrowserRouter>
      <ThemeProvider>
        <AuthProvider>
          <AppRoutes />
        </AuthProvider>
      </ThemeProvider>
    </BrowserRouter>
  );
}

export default App;
