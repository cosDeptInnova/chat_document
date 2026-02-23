import React, { useState, useEffect, useRef } from "react";
import { FiMenu, FiMessageSquare, FiClock, FiHelpCircle } from "react-icons/fi";
import cosmosLogo from "../../images/cosmosLogo.png";
import VoiceAvatar from "./VoiceAvatar";

export default function VoiceSidebar({
  isDarkMode,
  isOpen,
  setIsOpen,
  selectedOption,
  setSelectedOption,
  openHelp,
  isAgentSpeaking,
}) {
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);
  // Ancho inicial por defecto
  const [sidebarWidth, setSidebarWidth] = useState(320);
  const isResizing = useRef(false);

  const navOptions = [
    { id: "conversacion", icon: <FiMessageSquare className="w-4 h-4 md:w-5 md:h-5 mr-2 md:mr-3" />, label: "Conversación actual" },
    { id: "historial", icon: <FiClock className="w-4 h-4 md:w-5 md:h-5 mr-2 md:mr-3" />, label: "Historial de conversaciones" },
  ];

  // Para detectar cambios de tamaño de ventana
  useEffect(() => {
    const handleResize = () => setWindowWidth(window.innerWidth);
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Ajustar ancho dinámico (Solo afecta en Desktop si no se ha redimensionado manualmente)
  // En móvil, el ancho lo controla más bien el contenedor del layout.
  useEffect(() => {
    if (windowWidth < 768) {
      setSidebarWidth(isOpen ? 260 : 50); 
    } else if (windowWidth < 1024) {
      setSidebarWidth(isOpen ? 220 : 60);
    } else {
      setSidebarWidth(isOpen ? 300 : 70);
    }
  }, [windowWidth, isOpen]);

  // Control para redimensionar manualmente
  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isResizing.current) return;
      // Limitamos el ancho máximo y mínimo
      const newWidth = Math.max(60, Math.min(e.clientX, 450));
      setSidebarWidth(newWidth);
    };

    const handleMouseUp = () => (isResizing.current = false);

    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);
    return () => {
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
    };
  }, []);

  // Mostrar avatar solo si hay espacio suficiente y estamos en conversación
  const showAvatar = selectedOption === "conversacion" && isOpen && sidebarWidth > 180;

  return (
    <aside
      className={`relative h-full flex flex-col justify-between rounded-r-2xl md:rounded-r-3xl shadow-xl transition-all duration-300 ease-in-out border-r ${
        isDarkMode
          ? "bg-gray-800 text-white border-gray-700" 
          : "bg-gray-200 text-gray-800 border-gray-300"
      }`}
      style={{ width: sidebarWidth }}
    >
      {/* Sección superior con logo y botón de menú */}
      <div className="flex-1 flex flex-col overflow-y-auto hide-scrollbar">
        <div className={`flex items-center ${isOpen ? "justify-between" : "justify-center"} px-3 py-4 md:py-6`}>
          {isOpen && (
            <img src={cosmosLogo} alt="COSMOS Logo" className="h-8 md:h-10 object-contain transition-opacity duration-300" />
          )}

          <button
            onClick={() => setIsOpen(!isOpen)}
            className={`p-2 rounded-full transition-colors flex-shrink-0 ${
              isDarkMode
                ? "bg-gray-700 text-white hover:bg-gray-600"
                : "bg-gray-300 text-gray-800 hover:bg-gray-400"
            }`}
            aria-label={isOpen ? "Cerrar menú" : "Abrir menú"}
          >
            <FiMenu className="w-5 h-5 md:w-6 md:h-6" />
          </button>
        </div>

        {/* Opciones de navegación */}
        {isOpen && (
          <nav className="flex flex-col gap-2 mt-2 px-3">
            {navOptions.map(({ id, icon, label }) => {
              const isActive = selectedOption === id;
              return (
                <button
                  key={id}
                  onClick={() => setSelectedOption(id)}
                  className={`flex items-center px-3 py-2.5 rounded-xl transition-all duration-200 text-left text-xs md:text-sm font-medium truncate
                    ${
                      isActive
                        ? isDarkMode
                        ? "bg-blue-600/20 text-blue-300 shadow-sm border border-blue-500/30"
                        : "bg-white text-blue-700 shadow-sm border border-blue-100"
                      : isDarkMode
                        ? "text-gray-400 hover:bg-gray-700 hover:text-white"
                        : "text-gray-600 hover:bg-gray-300 hover:text-gray-900"
                    }
                  `}
                  title={label}
                >
                  {icon}
                  <span className="truncate">{label}</span>
                </button>
              );
            })}
          </nav>
        )}

        {/* Avatar dinámico que se oculta si el sidebar se hace muy pequeño */}
        {showAvatar && (
          <div className="p-4 flex flex-col items-center">
            <h3 className="text-base font-semibold mb-4 text-center">
              Agente COSMOS
            </h3>
            <div className="w-full flex justify-center">
              <VoiceAvatar
                isAgentSpeaking={isAgentSpeaking}
                videosPreloaded={true}
                className="w-full max-w-[240px] sm:max-w-[260px] md:max-w-[280px] lg:max-w-[300px] rounded-2xl overflow-hidden transition-all duration-300 max-h-[220px] object-contain"
              />
            </div>
          </div>
        )}
      </div>

      {/* Pie del sidebar con información */}
      {isOpen && (
        <div className={`text-xs space-y-2 px-3 pb-3 ${isDarkMode ? "text-gray-300" : "text-gray-500"}`}>
          <button
            onClick={openHelp}
            className={`flex items-center space-x-2 p-2 rounded transition w-full text-left ${
              isDarkMode ? "hover:bg-gray-700" : "hover:bg-gray-300"
            }`}
          >
            <FiHelpCircle className="w-4 h-4 mr-2" />
            <span>Ayuda</span>
          </button>

          <hr className={`${isDarkMode ? "border-gray-600" : "border-gray-300"}`} />
          <div className="truncate">
            <p className="italic">COSMOS Voice Agent</p>
            <p className="italic">Powered by COS Global Services</p>
            <p className="text-[10px] font-thin">Versión 1.0</p>
          </div>
        </div>
      )}

      {/* Barra de resize lateral */}
      <div
        onMouseDown={() => (isResizing.current = true)}
        className="absolute top-0 right-0 h-full w-4 cursor-ew-resize bg-transparent hover:bg-gray-400/20"
        style={{ zIndex: 50 }}
      />
    </aside>
  );
}