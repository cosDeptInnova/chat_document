import React, { useState, useEffect, useRef } from 'react';

export default function ImagePreviewModal({ isOpen, onClose, imageUrl, isDarkMode }) {
  const [visible, setVisible] = useState(false);
  const modalOverlayRef = useRef(null);

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
      if (e.target === modalOverlayRef.current) {
        handleClose();
      }
    };
    window.addEventListener('click', handleOutsideClick);
    return () => window.removeEventListener('click', handleOutsideClick);
  }, []);

  if (!isOpen || !imageUrl) return null;

  return (
    <div
      ref={modalOverlayRef}
      className="fixed inset-0 bg-black bg-opacity-70 flex items-center justify-center z-50 transition-opacity duration-300 p-4"
      onClick={(e) => {
        if (e.target === modalOverlayRef.current) {
          handleClose(); // o closeImageModal()
        }
      }}
    >
      <div
        className={`relative max-w-4xl max-h-[90vh] p-4 rounded-lg shadow-lg transform transition-all duration-300
          ${visible ? 'opacity-100 translate-y-0 scale-100' : 'opacity-0 -translate-y-6 scale-95'}
          ${isDarkMode ? 'bg-gray-900' : 'bg-white'}
        `}
        // Evitamos que clicks dentro de la imagen cierren el modal
        onClick={(e) => e.stopPropagation()}
      >
        {/* Imagen ajustada */}
        <img
          src={imageUrl}
          alt="Vista previa"
          className="w-auto h-auto max-w-full max-h-[85vh] object-contain rounded-lg select-none"
        />
        <button
          onClick={handleClose}
          className="absolute top-2 right-2 bg-red-600 hover:bg-red-700 text-white rounded-full p-2 shadow"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
    </div>
  );
}