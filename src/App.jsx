// src/App.jsx
import "@fortawesome/fontawesome-free/css/all.min.css";
import React, { useState, useEffect } from "react";
import { Routes, Route, Navigate } from "react-router-dom";

import MainLayout from "./layouts/MainLayout";
import ChatPage from "./pages/NewChatMainPanel";
import KnowledgePage from "./pages/KnowledgeBaseMainPanel";
import UseCasesMainPanel from "./pages/UseCasesMainPanel";
import ConversationHistory from "./pages/ConversationHistoryMainPanel";

import VoiceAgent from "./pages/VoicePage/VoiceAgentPage";
import VoiceLayout from "./layouts/VoiceLayout";
import TextCompareMainPanel from "./pages/TextCompareMainPanel";

/**
 * Autenticación basada en el backend:
 * - Llama a /api/modelo/me con cookies (access_token).
 * - Si responde 200 → tenemos usuario.
 * - Si responde 401/403 → no hay sesión.
 */
function useServerAuth() {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const res = await fetch("/api/modelo/me", {
          credentials: "include", // importante para enviar la cookie access_token
        });

        if (!res.ok) {
          if (!cancelled) {
            // 401 / 403 → sin sesión
            setUser(null);
            setLoading(false);
          }
          return;
        }

        const data = await res.json();
        if (!cancelled) {
          setUser(data);
          setLoading(false);
        }
      } catch (err) {
        console.error("Error cargando sesión desde /api/modelo/me:", err);
        if (!cancelled) {
          setUser(null);
          setLoading(false);
        }
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  return { loading, user, setUser };
}

export default function App() {
  const { loading, user, setUser } = useServerAuth();

  // Si no hay sesión, mandamos al flujo SSO de auth_sso
  useEffect(() => {
    if (!loading && !user) {
      // Después del SSO volverá a http://.../main/chat
      window.location.href = "/api/auth/login?next=/main/chat";
    }
  }, [loading, user]);

  // Pantalla de carga mientras comprobamos sesión
  if (loading) {
    return (
      <div className="w-full h-screen flex items-center justify-center">
        <span>Cargando sesión de COSMOS…</span>
      </div>
    );
  }

  // Estado intermedio mientras redirige al SSO
  if (!user) {
    return (
      <div className="w-full h-screen flex items-center justify-center">
        <span>Redirigiendo al inicio de sesión corporativo…</span>
      </div>
    );
  }

  // Con usuario autenticado → renderizamos toda la SPA
  return (
    <Routes>
      {/* Raíz: llévame al chat principal */}
      <Route path="/" element={<Navigate to="/main/chat" replace />} />

      {/* Alias directos al comparador */}
      <Route
        path="/text-compare"
        element={<Navigate to="/main/text-compare" replace />}
      />
      <Route
        path="/comparador"
        element={<Navigate to="/main/text-compare" replace />}
      />

      {/* Main con todas las páginas hijas */}
      <Route
        path="/main/*"
        element={<MainLayout user={user} setUser={setUser} />}
      >
        {/* Hijos de MainLayout */}
        <Route index element={<UseCasesMainPanel />} />
        <Route path="chat" element={<ChatPage />} />
        <Route path="knowledge" element={<KnowledgePage />} />
        <Route path="history" element={<ConversationHistory />} />

        {/* Comparador: ruta nueva + alias legacy */}
        <Route path="text-compare" element={<TextCompareMainPanel />} />
        <Route
          path="text_compare"
          element={<Navigate to="text-compare" replace />}
        />
      </Route>

      {/* Página independiente del Agente de Voz */}
      <Route
        path="/voice-agent"
        element={
          <VoiceLayout user={user} setUser={setUser}>
            <VoiceAgent />
          </VoiceLayout>
        }
      />

      {/* Cualquier otra ruta rara → al chat principal */}
      <Route path="*" element={<Navigate to="/main/chat" replace />} />
    </Routes>
  );
}
