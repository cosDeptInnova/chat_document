import React from "react";
import { FiAlertTriangle, FiX } from "react-icons/fi";

export default function DeleteChatModal({
  isDarkMode,
  isOpen,
  isClosing,
  onClose,
  onConfirm,
}) {
  if (!isOpen) return null;

  return (
    <div
      className={`fixed inset-0 flex items-center justify-center bg-black/50 backdrop-blur-sm z-50 p-4 transition-opacity duration-300
                  ${isClosing ? "animate-fadeOut opacity-0" : "animate-fadeIn opacity-100"}`}
      onClick={onClose}
    >
      <div
        className={`${isDarkMode ? "bg-gray-900 border-gray-700 text-white" : "bg-white border-gray-200 text-gray-900"} 
                    backdrop-blur-md p-6 rounded-2xl max-w-xs w-full relative shadow-2xl border transform transition-all duration-300
                    ${isClosing ? "scale-95 opacity-0" : "scale-100 opacity-100"}`}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Botón de cierre */}
        <button
          onClick={onClose}
          className={`absolute top-3 right-3 p-1.5 rounded-full shadow transition-colors duration-200
            ${isDarkMode ? "bg-red-900/50 hover:bg-red-800 text-red-200" : "bg-red-100 hover:bg-red-200 text-red-600"}`}
        >
          <FiX className="h-4 w-4" />
        </button>

        {/* Icono de advertencia */}
        <div className="flex justify-center mb-4">
          <div
            className={`w-14 h-14 rounded-full flex items-center justify-center shadow-inner
              ${isDarkMode ? "bg-red-900/30 text-red-400" : "bg-red-100 text-red-500"}`}
          >
            <FiAlertTriangle className="h-7 w-7" />
          </div>
        </div>

        {/* Título */}
        <h3 className="text-xl font-semibold mb-2 text-center">
          Confirmar eliminación
        </h3>

        {/* Mensaje */}
        <div className={`mb-6 text-center text-sm md:text-base leading-relaxed ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
          <p>¿Estás seguro de que quieres eliminar esta conversación?</p>
          <p className="mt-1 font-medium text-red-500">Se borrará para siempre.</p>
        </div>

        {/* Botones - SIEMPRE EN COLUMNA (Vertical) */}
        {/* Usamos 'flex-col' sin breakpoints para que sea igual en móvil y desktop */}
        <div className="flex flex-col gap-3">
          
          {/* Botón Eliminar (Lo pongo primero si quieres que sea el más accesible, o segundo. Aquí he puesto Eliminar primero para darle importancia) */}
          <button
            onClick={onConfirm}
            className="w-full px-4 py-3 rounded-xl bg-red-600 hover:bg-red-700 text-white font-medium shadow-md hover:shadow-lg transition-all duration-200 text-sm md:text-base"
          >
            Eliminar
          </button>

          {/* Botón Cancelar */}
          <button
            onClick={onClose}
            className={`w-full px-4 py-3 rounded-xl transition-colors duration-200 font-medium text-sm md:text-base
              ${isDarkMode 
                ? "bg-gray-800 text-gray-300 hover:bg-gray-700 hover:text-white" 
                : "bg-gray-100 text-gray-700 hover:bg-gray-200 hover:text-gray-900"}`}
          >
            Cancelar
          </button>

        </div>
      </div>
    </div>
  );
}