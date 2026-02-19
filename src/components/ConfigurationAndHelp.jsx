import React, { useState, useEffect, useRef } from 'react';
import { FiSettings, FiHelpCircle, FiX } from "react-icons/fi";
import HelpContent from "./HelpContent";
import SettingsContent from './SettingsContent';

export default function ConfigurationAndHelp({ isOpen, onClose, isDarkMode }) {
  const [visible, setVisible] = useState(false);
  const modalRef = useRef(null);
  const [activeTab, setActiveTab] = useState("ajustes");
  const [indicatorStyle, setIndicatorStyle] = useState({});
  const tabsRef = useRef({});

  useEffect(() => {
    if (isOpen) {
      setVisible(true);
    }
  }, [isOpen]);

  const handleClose = () => {
    setVisible(false);
    setTimeout(() => onClose(), 300);
  };

  useEffect(() => {
    const handleOutsideClick = (e) => {
      if (e.target === modalRef.current) {
        handleClose();
      }
    };
    window.addEventListener('click', handleOutsideClick);
    return () => window.removeEventListener('click', handleOutsideClick);
  }, []);

  // Actualizar la posición del indicador cuando cambia la pestaña activa
  useEffect(() => {
    if (tabsRef.current[activeTab]) {
      const tabElement = tabsRef.current[activeTab];
      const { offsetLeft, offsetWidth } = tabElement;
      
      setIndicatorStyle({
        left: offsetLeft,
        width: offsetWidth,
        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)'
      });
    }
  }, [activeTab]);

  if (!isOpen) return null;

  return (
    <div 
      ref={modalRef} 
      className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 transition-opacity duration-300 p-4"
    >
      <div className={`relative w-full max-w-lg md:max-w-3xl h-auto max-h-[85vh] flex flex-col rounded-2xl md:rounded-3xl shadow-2xl transform transition-all duration-500 
        ${visible ? 'opacity-100 translate-y-0 scale-100' : 'opacity-0 translate-y-4 scale-95'} 
        ${isDarkMode ? 'bg-gray-900 text-white' : 'bg-white text-blue-900'}`}
      >
        {/* Header con TabBar */}
        <div className="flex justify-between items-center p-4 md:p-6 pb-2 md:pb-4">
          <h2 className={`text-2xl md:text-3xl font-extrabold ${isDarkMode ? 'text-blue-400' : 'text-blue-700'}`}>
            Ajustes y Ayuda
          </h2>
          <button
            onClick={handleClose}
            className="bg-red-500 hover:bg-red-600 text-white rounded-full p-2 shadow transition-colors"
          >
            <FiX className="w-5 h-5" />
          </button>
        </div>

        {/* Tabs con indicador animado */}
        <div className="relative mb-4 md:mb-8 px-5 md:px-8">
          <div className="flex border-b border-gray-300 dark:border-gray-700">
            <button 
              ref={el => tabsRef.current.ajustes = el}
              onClick={() => setActiveTab("ajustes")} 
              className={`flex items-center justify-center flex-1 py-3 md:py-4 text-center text-sm md:text-base font-medium transition-colors duration-300 ${
                activeTab === "ajustes" 
                  ? (isDarkMode ? "text-white" : "text-blue-600") 
                  : (isDarkMode ? "text-gray-400 hover:text-white" : "text-gray-500 hover:text-blue-600")
              }`}
            >
              <FiSettings className="w-4 h-4 md:w-5 md:h-5 mr-2" />
              Ajustes
            </button>
            
            <button 
              ref={el => tabsRef.current.ayuda = el}
              onClick={() => setActiveTab("ayuda")} 
              className={`flex items-center justify-center flex-1 py-3 md:py-4 text-center text-sm md:text-base font-medium transition-colors duration-300 ${
                activeTab === "ayuda" 
                  ? (isDarkMode ? "text-white" : "text-blue-600") 
                  : (isDarkMode ? "text-gray-400 hover:text-white" : "text-gray-500 hover:text-blue-600")
              }`}
            >
              <FiHelpCircle className="w-4 h-4 md:w-5 md:h-5 mr-2" />
              Ayuda
            </button>
          </div>
          
          {/* Indicador animado */}
          <div 
            className="absolute bottom-0 h-[2px] md:h-1 bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-300"
            style={indicatorStyle}
          />
        </div>

        {/* Contenido dinámico */}
        <div className="flex-1 overflow-y-auto px-5 md:px-8 pb-8 hide-scrollbar">
          {activeTab === "ajustes" && <SettingsContent isDarkMode={isDarkMode}/>}
          
          {activeTab === "ayuda" && <HelpContent isDarkMode={isDarkMode}/>}
        </div>
      </div>
    </div>
  );
}
