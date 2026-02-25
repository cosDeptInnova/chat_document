import React, { useState, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import Header from "../components/Header";
import VoiceSidebar from "../components/voice/VoiceSidebar";
import { useSettings } from "../hooks/useSettings";
import VoiceHelp from "../components/voice/VoiceHelp";

import cosmosIconBlue from "../images/cosmos_1_1758106496233.png";
import VoiceAgentPage from "../pages/VoicePage/VoiceAgentPage";
import VoiceConversationHistoryPage from "../pages/VoicePage/VoiceConversationHistoryPage";

export default function VoiceLayout({ user, setUser }) {
  // >= 768 (Desktop) -> Empieza abierto. < 768 (Móvil) -> Empieza cerrado (modo mini)
  const [isSidebarOpen, setIsSidebarOpen] = useState(window.innerWidth >= 768);

  const { darkMode } = useSettings();
  const [isDarkMode, setIsDarkMode] = useState(() => darkMode);

  // Estado del sidebar
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  // Estado para que alterne el video de hablando o escuchando segun el chat de mensajes
  const [isAgentSpeaking,  setIsAgentSpeking] = useState(false);

  // Para saber si estams o no en movil y ajustar el sidebar
  const [isMobile, setIsMobile] = useState(window.innerWidth < 768);

  useEffect(() => {
    const handleResize = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      if (!mobile) setIsSidebarOpen(true);
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Mantiene sincronizado el darkMode global y local
  useEffect(() => {
    setIsDarkMode(darkMode);
  }, [darkMode]);

  // Search params
  const [searchParams, setSearchParams] = useSearchParams();
  const view = searchParams.get("view") || "conversacion";
  const chatIdParam = searchParams.get("id") || null; // Variable para guardar el Id del chat para que se abra en la página de conversación actual

  const openConversation = (chatId) => {
    setSearchParams({ view: "conversacion", id: chatId });
  };

  const handleOptionChange = (option) => {
    if (option === "conversacion") {
      setSearchParams({ view: "conversacion" });
    } else {
      setSearchParams({ view: "historial" });
    }
    // En móvil, cerrar sidebar al navegar
    if (isMobile) setIsSidebarOpen(false);
  };

  useEffect(() => {
    if (isDarkMode) document.body.classList.add("dark-mode");
    else document.body.classList.remove("dark-mode");
  }, [isDarkMode]);

  return (
    <div className={`voice-layout flex h-screen h-[100dvh] w-screen overflow-hidden transition-colors duration-300`}>
      
      {/* OVERLAY (FONDO OSCURO) - Solo en móvil y si está abierto */}
      {isMobile && isSidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 transition-opacity duration-300 backdrop-blur-sm"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}

      {/* CONTENEDOR DEL SIDEBAR */}
      <div
        className={`
          fixed inset-y-0 left-0 z-50 h-full transition-all duration-300
          md:relative md:z-auto
        `}
      >
        <VoiceSidebar
          isDarkMode={isDarkMode}
          isOpen={isSidebarOpen}
          setIsOpen={setIsSidebarOpen}
          selectedOption={view}
          setSelectedOption={handleOptionChange}
          openHelp={() => setIsHelpOpen(true)}
          isAgentSpeaking={isAgentSpeaking}
        />
      </div>

      {/* CONTENIDO PRINCIPAL */}
      {/* padding-left solo en móvil para respetar la barra mini */}
      <div className="flex flex-col flex-1 min-w-0 pl-[50px] md:pl-0 transition-all duration-300">
        <Header
          isDarkMode={isDarkMode}
          setIsDarkMode={setIsDarkMode}
          user={user}
          setUser={setUser}
          extraContent={ // Contenido extra para el Header solo en VoiceLayout
            <>
              <div
                className={`w-8 h-8 rounded-lg flex items-center justify-center overflow-hidden flex-shrink-0 ${
                  isDarkMode ? "bg-gray-700" : "bg-gray-200"
                }`}
              >
                <img
                  src={cosmosIconBlue}
                  alt="COSMOS"
                  className="w-8 h-8 object-cover"
                />
              </div>
              <div className="min-w-0 overflow-hidden">
                <h1
                  className={`text-base md:text-lg font-semibold truncate ${
                    isDarkMode ? "text-blue-400" : "text-blue-600"
                  }`}
                >
                  Agente de Voz COSMOS
                </h1>
                <p
                  className={`text-[10px] md:text-xs truncate ${
                    isDarkMode ? "text-gray-400" : "text-gray-500"
                  }`}
                >
                  Asistente de Soporte IT
                </p>
              </div>
            </>
          }
        />

        <main className="voice-layout-main flex-1 relative overflow-hidden transition-all duration-300">
          <div className="absolute inset-0 w-full h-full flex flex-col">
            {view === "conversacion" && (
              <VoiceAgentPage 
                isDarkMode={isDarkMode}
                //onEndCall={handleEndCall}
                chatId={chatIdParam} // Y aquí le pasamos el Id del chat a abrir seleccionado desde el historialç
                onAgentSpekingChange={setIsAgentSpeking} // Callback para actualizar quien esta hablando y se muestre un video u otro
              />
            )}

            {view === "historial" && (
              <VoiceConversationHistoryPage
                isDarkMode={isDarkMode}
                onOpenConversation={openConversation}
                // Aquí luego cargamos la conversación específica
              />
            )}
          </div>
        </main>
      </div>

      {/* Modal de Ayuda */}
      <VoiceHelp
        isOpen={isHelpOpen}
        onClose={() => setIsHelpOpen(false)}
        isDarkMode={isDarkMode}
      />
    </div>
  );
}