// src/layouts/MainLayout.jsx
// Layout general para la app (sidebar + header + panel central)

import React, { useState, useEffect } from "react";
import { useSearchParams, useLocation, useNavigate } from "react-router-dom";

import Sidebar from "../components/Sidebar";
import ConfigurationAndHelp from "../components/ConfigurationAndHelp";
import Header from "../components/Header";

import { useSettings } from "../hooks/useSettings";
import { bootstrapModelo, bootstrapChatdoc } from "../lib/api";

import CasosUsoMainPanel from "../pages/UseCasesMainPanel";
import NuevoChatMainPanel from "../pages/NewChatMainPanel";
import BaseConocimientosMainPanel from "../pages/KnowledgeBaseMainPanel";
import HistorialMainPanel from "../pages/ConversationHistoryMainPanel";
import TextCompareMainPanel from "../pages/TextCompareMainPanel";

export default function MainLayout({ user, setUser }) {
  // Si el ancho es >= 768 (Desktop) empieza en true (abierto).
  // Si es menor (Móvil), empieza en false (cerrado/mini).
  const [isSidebarOpen, setIsSidebarOpen] = useState(window.innerWidth >= 768);

  const { darkMode } = useSettings();
  const [isDarkMode, setIsDarkMode] = useState(darkMode);
  const [isConfigHelpOpen, setIsConfigHelpOpen] = useState(false);

  // Detectar si es móvil para la lógica del overlay
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

  const [searchParams, setSearchParams] = useSearchParams();
  const location = useLocation();
  const navigate = useNavigate();

  const selectedOption = searchParams.get("selectedOption") || "casos";
  const initialMessage = searchParams.get("initialMessage") || "";
  const chatMode = searchParams.get("chatMode") || "modelo";

  // ¿Estamos en la ruta del comparador? (p.ej. /main/text-compare)
  const isTextCompareRoute = location.pathname.includes("/text-compare");

  // Bootstrap CSRF / sesiones de los micros
  useEffect(() => {
    bootstrapModelo().catch((err) => {
      console.warn("Error en bootstrapModelo:", err);
    });

    bootstrapChatdoc().catch((err) => {
      console.warn("Error en bootstrapChatdoc:", err);
    });
  }, []);

  // Efecto para detectar cambio de tamaño de pantalla y ajustar comportamiento
  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      
      // Si redimensionan la ventana del navegador en PC de pequeña a grande,
      // forzamos que se abra para que no se quede la barra mini en una pantalla gigante.
      if (!mobile) {
        setIsSidebarOpen(true);
      }
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Cambiar de panel y mantener URL sincronizada
  const changeOption = (opt) => {
    setSearchParams((prev) => {
      const p = new URLSearchParams(prev);
      p.set("selectedOption", opt);
      // si cambiamos de opción, limpiamos chatId salvo que sigas en "nuevo"
      if (opt !== "nuevo") {
        p.delete("chatId");
      }
      return p;
    });
    // En móvil, al seleccionar una opción, cerramos (volvemos a mini) automáticamente
    if (isMobile) {
      setIsSidebarOpen(false);
    }

    // Navegación de rutas según la opción elegida
    if (opt === "comparador") {
      // Ir a la ruta específica del comparador
      if (!location.pathname.includes("/text-compare")) {
        navigate("/main/text-compare");
      }
    } else {
      // Para el resto de opciones, garantizamos que estamos en /main
      if (location.pathname.includes("/text-compare")) {
        // salimos de la ruta del comparador
        navigate("/main");
      }
      // si ya estás en /main, no hace falta tocar nada más:
      // el cambio de searchParams controla el panel visible
    }
  };

  // Sincronizar cambios de Settings con el estado local
  useEffect(() => {
    setIsDarkMode(darkMode);
  }, [darkMode]);

  // ---- Selección de contenido principal ----
  let mainContent = null;

  if (isTextCompareRoute || selectedOption === "comparador") {
    // Comparador de archivos
    mainContent = <TextCompareMainPanel isDarkMode={isDarkMode} />;
  } else if (selectedOption === "nuevo") {
    mainContent = (
      <NuevoChatMainPanel
        isDarkMode={isDarkMode}
        initialMessage={initialMessage}
        chatId={searchParams.get("chatId")}
        chatMode={chatMode}
      />
    );
  } else if (selectedOption === "casos") {
    mainContent = <CasosUsoMainPanel isDarkMode={isDarkMode} />;
  } else if (selectedOption === "historial") {
    mainContent = (
      <HistorialMainPanel
        isDarkMode={isDarkMode}
        onOpenConversation={(chatId) => {
          // Cambiar al panel de nuevo chat y guardar el chatId
          setSearchParams((prev) => {
            const p = new URLSearchParams(prev);
            p.set("selectedOption", "nuevo");
            p.set("chatId", chatId);
            return p;
          });
          if (location.pathname.includes("/text-compare")) {
            navigate("/main");
          }
        }}
      />
    );
  } else if (selectedOption === "conocimientos") {
    mainContent = <BaseConocimientosMainPanel isDarkMode={isDarkMode} />;
  } else {
    // fallback por si acaso
    mainContent = <CasosUsoMainPanel isDarkMode={isDarkMode} />;
  }

  return (
    <div
      className={`flex font-sans h-screen w-full overflow-hidden ${
        isDarkMode ? "bg-gray-900" : "bg-gray-50"
      }`}
    >
      {/* Sidebar */}
      {isMobile && isSidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 transition-opacity duration-300 backdrop-blur-sm"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      <div
        className={`
          fixed inset-y-0 left-0 z-50 h-full transition-all duration-300
          md:relative md:z-auto
        `}
      >
        <Sidebar
          onClose={() => setIsSidebarOpen(!isSidebarOpen)}
          isDarkMode={isDarkMode}
          isOpen={isSidebarOpen}
          openConfigHelp={() => setIsConfigHelpOpen(true)}
          selectedOption={selectedOption}
          setSelectedOption={changeOption}
        />
      </div>

      {/* Main Content */}
      <div className="flex flex-1 flex-col h-screen overflow-hidden relative transition-all duration-300 pl-[50px] md:pl-0 w-full">
        <Header
          isDarkMode={isDarkMode}
          setIsDarkMode={setIsDarkMode}
          user={user}
          setUser={setUser}
        />

        {/* Igual que antes: el panel principal va directamente aquí */}
        <div className="flex-1 w-full overflow-hidden relative flex flex-col">
          {mainContent}
        </div>
      </div>

      {/* Configuración y ayuda */}
      <ConfigurationAndHelp
        isOpen={isConfigHelpOpen}
        onClose={() => setIsConfigHelpOpen(false)}
        isDarkMode={isDarkMode}
      />
    </div>
  );
}