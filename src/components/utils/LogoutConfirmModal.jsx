import React, { useState, useEffect, useRef } from 'react';

export default function LogoutConfirmModal({ onClose, isOpen, isDarkMode, onConfirm }) {
  const [isVisible, setIsVisible] = useState(false);
  const modalRef = useRef();

  // Animación de apertura/cierre
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => setIsVisible(true), 10);
    } else {
      setIsVisible(false);
    }
  }, [isOpen]);

  // Cierre con animación
  const handleClose = () => {
    setIsVisible(false);
    setTimeout(() => onClose(), 300);
  };

  // Llama a la función que te pasa el Header
  const handleLogout = () => {
    if (onConfirm) onConfirm();
    handleClose();
  };

  // Cierre al hacer click fuera del modal
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (modalRef.current && !modalRef.current.contains(e.target)) {
        handleClose();
      }
    };
    if (isOpen) document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, [isOpen]);

  if (!isOpen && !isVisible) return null;

  return (
    <div
      className={`fixed inset-0 z-50 flex items-center justify-center bg-opacity-60 backdrop-blur-sm transition-opacity duration-300 p-4 ${
        isVisible ? 'opacity-100' : 'opacity-0 pointer-events-none'
      }`}
    >
      <div
        ref={modalRef}
        className={`rounded-2xl p-6 md:p-8 w-full max-w-sm shadow-2xl text-center relative transform transition-all duration-300 ${
          isVisible ? 'scale-100 opacity-100' : 'scale-95 opacity-0'
        } ${isDarkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-900'}`}
      >
        <div className={`w-12 h-12 md:w-16 md:h-16 rounded-full flex items-center justify-center mx-auto mb-4 ${isDarkMode ? 'bg-red-900/30' : 'bg-red-100'}`}>
          <svg className={`w-6 h-6 md:w-8 md:h-8 ${isDarkMode ? 'text-red-500' : 'text-red-600'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1" />
          </svg>
        </div>

        <h3 className={`text-lg md:text-xl font-semibold mb-2 ${isDarkMode ? 'text-white' : 'text-gray-800'}`}>Cerrar sesión</h3>
        <p className={`mb-6 text-sm md:text-base ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>¿Estás seguro de que quieres cerrar tu sesión?</p>

        <div className="flex justify-center gap-3">
          <button
            className={`px-4 py-3 md:py-2.5 rounded-xl transition-colors duration-200 font-medium flex-1 text-sm md:text-base ${
              isDarkMode
                ? 'bg-gray-700 hover:bg-gray-600 text-white'
                : 'bg-gray-200 hover:bg-gray-300 text-gray-800'
            }`}
            onClick={handleClose}
          >
            Cancelar
          </button>
          <button
            className="px-4 py-3 md:py-2.5 rounded-xl bg-red-600 hover:bg-red-700 text-white transition-colors duration-200 font-medium flex-1 shadow-md hover:shadow-lg text-sm md:text-base"
            onClick={handleLogout}
          >
            Cerrar sesión
          </button>
        </div>

        <button
          onClick={handleClose}
          className="absolute top-3 right-3 bg-red-600 hover:bg-red-700 text-white rounded-full p-1.5 md:p-2 shadow"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12"/>
          </svg>
        </button>
      </div>
    </div>
  );
}