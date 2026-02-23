import React, { useState, useEffect, useRef } from 'react';
import cosmosLogo from "../images/cosmosLogo.png";
import { FiMenu, FiPenTool, FiClock, FiCpu, FiDatabase, FiSettings } from "react-icons/fi";

export default function Sidebar({ onClose, isDarkMode, isOpen, openConfigHelp, selectedOption, setSelectedOption }) {
  const [windowWidth, setWindowWidth] = useState(window.innerWidth);
  const [sidebarWidth, setSidebarWidth] = useState(320);
  const isResizing = useRef(false);

  const navOptions = [
    { id: "nuevo", icon: <FiPenTool className="w-4 h-4 md:w-5 md:h-5 mr-2 md:mr-3" />, label: "Nuevo Chat" },
    { id: "historial", icon: <FiClock className="w-4 h-4 md:w-5 md:h-5 mr-2 md:mr-3" />, label: "Accede a tu historial" },
    { id: "casos", icon: <FiCpu className="w-4 h-4 md:w-5 md:h-5 mr-2 md:mr-3" />, label: "Casos de uso" },
    { id: "conocimientos", icon: <FiDatabase className="w-4 h-4 md:w-5 md:h-5 mr-2 md:mr-3" />, label: "Base de conocimientos" }
  ];

  useEffect(() => {
    const handleResize = () => setWindowWidth(window.innerWidth);
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Ajuste de ancho inicial según el dispositivo
  useEffect(() => {
    if (windowWidth < 768) {
      // Móvil: 260px abierto, 50px cerrado
      setSidebarWidth(isOpen ? 260 : 50);
    } else if (windowWidth < 1024) {
      // Tablet/Portátil pequeño: 220px abierto, 60px cerrado
      setSidebarWidth(isOpen ? 220 : 60);
    } else {
      // Desktop: 280px abierto, 70px cerrado
      setSidebarWidth(isOpen ? 280 : 70);
    }
  }, [windowWidth, isOpen]);

  // Redimensionamiento manual
  useEffect(() => {
    const handleMouseMove = (e) => {
      if (!isResizing.current) return;
      // Limitamos el ancho entre 60px y 450px
      const newWidth = Math.max(60, Math.min(e.clientX, 450));
      setSidebarWidth(newWidth);
    };

    const handleMouseUp = () => {
      isResizing.current = false;
    };

    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, []);

  return (
    <aside
      className={`relative h-full flex flex-col justify-between rounded-r-2xl md:rounded-r-3xl shadow-lg transition-all duration-300 ease-in-out ${
        isDarkMode 
          ? 'bg-gray-800 text-white' 
          : 'bg-gray-200 text-gray-800'
      }`}
      style={{ width: sidebarWidth }}>
      {/* Parte Superior ddel SideBar */}
      <div className="flex-1 flex flex-col overflow-y-auto hide-scrollbar">
        {/* Header del Sidebar (Logo + Toggle) */}
        <div className={`flex items-center ${isOpen ? 'justify-between' : 'justify-center'} px-3 py-4 md:py-6`}>
          {isOpen && (
            <img 
              src={cosmosLogo} 
              alt="COSMOS Logo" 
              className="h-8 md:h-10 transition-opacity duration-300"
            />
          )}
          <button
            onClick={onClose}
            className={`p-2 rounded-full flex items-center justify-center transition-colors flex-shrink-0 ${
              isDarkMode
                ? 'bg-gray-700 text-white hover:bg-gray-600'
                : 'bg-gray-300 text-gray-800 hover:bg-gray-400'
            }`}
            aria-label={isOpen ? "Cerrar menú" : "Abrir menú"}
          >
            <FiMenu className="w-5 h-5 md:w-6 md:h-6" />
          </button>
        </div>

        {/* Menú de Navegación */}
        {isOpen && (
          <nav className="flex flex-col gap-2 mt-4 px-3">
            {navOptions.map(({ id, icon, label }) => {
              const isActive = selectedOption === id;
              return (
                <button
                  key={id}
                  onClick={() => setSelectedOption(id)}
                  className={`
                    flex items-center px-3 py-2.5 rounded-xl transition-all duration-200 text-left truncate
                    text-xs md:text-sm font-medium
                    ${isActive
                      ? isDarkMode
                        ? 'bg-blue-600/20 text-blue-300 shadow-sm border border-blue-500/30 translate-y-[-1px]'
                        : 'bg-white text-blue-700 shadow-sm border border-blue-100 translate-y-[-1px]'
                      : isDarkMode
                      ? 'text-gray-400 hover:bg-gray-700 hover:text-white'
                      : 'text-gray-600 hover:bg-gray-300 hover:text-gray-900'
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
      </div>

      {/* Footer de la página */}
      {isOpen && (
        <div className={`p-3 md:p-4 ${isDarkMode ? 'border-gray-700' : 'border-gray-300'}`}>
          <button
            onClick={openConfigHelp}
            className={`flex items-center w-full gap-2 p-2.5 rounded-lg transition-colors text-xs md:text-sm font-medium mb-3 text-left ${
              isDarkMode ? 'hover:bg-gray-700 text-gray-300' : 'hover:bg-gray-300 text-gray-600'
            }`}
          >
            <FiSettings className="w-3 h-3 md:w-4 md:h-4 flex-shrink-0" />
            <span>Ajustes y ayuda</span>
          </button>

          <hr className={`${isDarkMode ? 'border-gray-600' : 'border-gray-300'}`} />
          <div className="text-[10px] md:text-xs opacity-50 space-y-0.5 truncate">
            <p className="italic">Powered by COS Global Services</p>
            <p className="font-thin">Versión 1.0</p>
          </div>
        </div>
      )}

      {/* Barra de resize visible */}
      <div
        onMouseDown={() => (isResizing.current = true)}
        className="absolute top-0 right-0 h-full w-1.5 cursor-ew-resize hover:bg-blue-500/50 transition-colors z-50 hidden md:block"
      />
    </aside>
  );
}