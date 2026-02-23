import React, { useState, useRef, useEffect } from "react";

export default function UserInfoModal({ user, onClose, isOpen, isDarkMode }) {
  const [isVisible, setIsVisible] = useState(false);
  const modalRef = useRef();

  // Si no hay user, usamos datos "mock"
  const mockUser = {
    name: "Carlos Martínez",
    email: "cmartinez@cosgs.com",
    department: "Operaciones e Innovación",
    role: "Usuario",
    createdAt: "2025-09-11T11:05:33.000Z",
  };

  // Normalizamos el objeto que viene de /me para que encaje con el modal
  const backendUser = user || {};

  const finalUser = {
    // Nombre: intentamos full_name → name → username → mock
    name:
      backendUser.full_name ||
      backendUser.name ||
      backendUser.username ||
      mockUser.name,

    // Email: si no existe en backend, usamos el mock
    email: backendUser.email || mockUser.email,

    // Departamentos: array de objetos → name / directory / department_directory
    department:
      backendUser.departments && backendUser.departments.length > 0
        ? backendUser.departments
            .map(
              (d) =>
                d.name ||
                d.directory ||
                d.department_directory ||
                String(d)
            )
            .join(", ")
        : mockUser.department,

    // Rol: role_name o "rol #id" (mismo criterio que el JS legacy)
    role:
      backendUser.role_name ||
      (backendUser.role_id ? `rol #${backendUser.role_id}` : mockUser.role),

    // Fecha de alta: created_at o date_joined, con fallback al mock
    createdAt:
      backendUser.created_at ||
      backendUser.date_joined ||
      mockUser.createdAt,
  };

  // Función para formatear la fecha y hora en español
  const formatDateTime = (isoDate) => {
    if (!isoDate) return "—";
    const date = new Date(isoDate);

    const fecha = date.toLocaleDateString("es-ES", {
      day: "2-digit",
      month: "long",
      year: "numeric",
    });

    const hora = date.toLocaleTimeString("es-ES", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });

    // Si es la 01:xx:xx o 13:xx:xx usamos "a la" en singular
    const preposicion = hora.startsWith("01") || hora.startsWith("13") ? "a la" : "a las";

    return `${fecha} ${preposicion} ${hora}`;
  };

  // Función para obtener iniciales del nombre
  const getInitials = (name) => {
    if (!name) return "U";
    return name
      .split(" ")
      .map((n) => n.charAt(0).toUpperCase())
      .join("")
      .slice(0, 2); // máximo dos letras
  };

  // Animación de apertura/cierre
  useEffect(() => {
    if (isOpen) {
      setTimeout(() => setIsVisible(true), 10);
    } else {
      setIsVisible(false);
    }
  }, [isOpen]);

  // Manejo del cierre con animación
  const handleClose = () => {
    setIsVisible(false); // inicia animación de cierre
    setTimeout(() => onClose(), 300); // espera animación antes de cerrar
  };

  // Click fuera del modal
  useEffect(() => {
    const handleClickOutside = (e) => {
      if (modalRef.current && !modalRef.current.contains(e.target)) {
        handleClose();
      }
    };
    if (isOpen) document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isOpen]);

  if (!isOpen && !isVisible) return null;

  return (
    <div
      className={`fixed inset-0 z-50 flex items-center justify-center bg-opacity-60 backdrop-blur-sm transition-opacity duration-300 p-4 ${
        isVisible ? "opacity-100" : "opacity-0 pointer-events-none"
      }`}
    >
      <div
        ref={modalRef}
        className={`rounded-2xl p-6 md:p-8 w-full max-w-md md:max-w-lg shadow-2xl relative transform transition-all duration-300 ${
          isVisible ? "scale-100 opacity-100" : "scale-95 opacity-0"
        } ${isDarkMode ? "bg-gray-800 text-gray-100" : "bg-white text-gray-900"}`}
      >
        {/* Encabezado con avatar */}
        <div className="flex flex-col md:flex-row items-center mb-6 md:mb-8 text-center md:text-left">
          {/* Contenedor relativo para el avatar y su borde animado */}
          <div className="relative w-14 h-14 md:w-16 md:h-16 mb-3 md:mb-0 md:mr-4 flex-shrink-0">
            {/* Anillo animado alrededor del avatar */}
            <div className="absolute inset-0 rounded-full p-[2px] bg-gradient-to-tr from-purple-500 via-pink-500 to-yellow-400 animate-spin">
              <div className={`w-full h-full rounded-full ${isDarkMode ? "bg-gray-900" : "bg-white"}`}></div>
            </div>

            {/* Avatar interior con iniciales */}
            <div
              className={`absolute inset-[3px] md:inset-[4px] rounded-full flex items-center justify-center text-lg md:text-xl font-bold shadow-lg
                ${isDarkMode ? "bg-gray-800 text-blue-300 shadow-blue-400/40" : "bg-white text-blue-700 shadow-blue-300/40"}`}
            >
              {getInitials(finalUser.name)}
            </div>
          </div>

          {/* Título */}
          <h2
            className={`text-2xl md:text-3xl font-bold relative pb-0.5 md:pb-1 before:absolute before:bottom-[-4px] before:left-0 before:h-1 before:w-full before:rounded-full
              ${isDarkMode 
                ? "text-blue-200 before:bg-gradient-to-r from-purple-400 via-pink-400 to-yellow-300" 
                : "text-blue-800 before:bg-gradient-to-r from-purple-500 via-pink-500 to-yellow-400"}`}
          >
            Información personal
          </h2>
        </div>

        <div className="space-y-4 md:space-y-6 mb-6 md:mb-8">
          {/* Nombre */}
          <div className="flex items-center">
            <div
              className={`w-12 h-12 rounded-full flex items-center justify-center mr-4 ${
                isDarkMode ? "bg-blue-900/30" : "bg-blue-100"
              }`}
            >
              <svg
                className={`w-6 h-6 ${isDarkMode ? "text-blue-400" : "text-blue-600"}`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                />
              </svg>
            </div>
            <div className="min-w-0">
              <p className={`text-xs md:text-sm ${isDarkMode ? "text-gray-400" : "text-gray-500"}`}>
                Nombre completo
              </p>
              <p className="font-medium text-sm md:text-base break-words">{finalUser.name}</p>
            </div>
          </div>

          {/* Correo */}
          <div className="flex items-center">
            <div
              className={`w-12 h-12 rounded-full flex items-center justify-center mr-4 ${
                isDarkMode ? "bg-blue-900/30" : "bg-blue-100"
              }`}
            >
              <svg
                className={`w-6 h-6 ${isDarkMode ? "text-blue-400" : "text-blue-600"}`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                />
              </svg>
            </div>
            <div className="min-w-0">
              <p className={`text-xs md:text-sm ${isDarkMode ? "text-gray-400" : "text-gray-500"}`}>
                Correo electrónico
              </p>
              <p className="font-medium text-sm md:text-base break-words">{finalUser.email}</p>
            </div>
          </div>

          {/* Departamento */}
          <div className="flex items-center">
            <div
              className={`w-12 h-12 rounded-full flex items-center justify-center mr-4 ${
                isDarkMode ? "bg-blue-900/30" : "bg-blue-100"
              }`}
            >
              <svg
                className={`w-6 h-6 ${isDarkMode ? "text-blue-400" : "text-blue-600"}`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M3 21h18M4 21V5a2 2 0 012-2h4a2 2 0 012 2v16M14 21V9a2 2 0 012-2h2a2 2 0 012 2v12M8 9h.01M8 13h.01M8 17h.01"
                />
              </svg>
            </div>
            <div className="min-w-0">
              <p className={`text-xs md:text-sm ${isDarkMode ? "text-gray-400" : "text-gray-500"}`}>
                Departamento
              </p>
              <p className="font-medium text-sm md:text-base break-words">{finalUser.department}</p>
            </div>
          </div>

          {/* Rol de Usuario */}
          <div className="flex items-center">
            <div
              className={`w-12 h-12 rounded-full flex items-center justify-center mr-4 ${
                isDarkMode ? "bg-blue-900/30" : "bg-blue-100"
              }`}
            >
              <svg
                className={`w-6 h-6 ${isDarkMode ? "text-blue-400" : "text-blue-600"}`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4z"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M6 20v-1a6 6 0 0112 0v1"
                />
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 14l3 2-3 6-3-6 3-2z"
                />
              </svg>
            </div>
            <div className="min-w-0">
              <p className={`text-xs md:text-sm ${isDarkMode ? "text-gray-400" : "text-gray-500"}`}>
                Rol de usuario
              </p>
              <p className="font-medium text-sm md:text-base break-words">{finalUser.role}</p>
            </div>
          </div>

          {/* Fecha de alta */}
          <div className="flex items-center">
            <div
              className={`w-12 h-12 rounded-full flex items-center justify-center mr-4 ${
                isDarkMode ? "bg-blue-900/30" : "bg-blue-100"
              }`}
            >
              <svg
                className={`w-6 h-6 ${isDarkMode ? "text-blue-400" : "text-blue-600"}`}
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z"
                />
              </svg>
            </div>
            <div className="min-w-0">
              <p className={`text-xs md:text-sm ${isDarkMode ? "text-gray-400" : "text-gray-500"}`}>
                Fecha de alta
              </p>
              <p className="font-medium text-sm md:text-base break-words">{formatDateTime(finalUser.createdAt)}</p>
            </div>
          </div>
        </div>

        {/* Botón cerrar (X) */}
        <button
          onClick={handleClose}
          className="absolute top-3 right-3 bg-red-600 hover:bg-red-700 text-white rounded-full p-1.5 md:p-2 shadow z-10"
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-4 w-4 md:h-5 md:w-5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        {/* Botón Aceptar */}
        <button
          onClick={handleClose}
          className={`w-full py-2.5 md:py-3 rounded-xl font-medium transition-colors duration-200 shadow-md hover:shadow-lg text-sm md:text-base ${
            isDarkMode ? "bg-blue-700 hover:bg-blue-600 text-white" : "bg-blue-600 hover:bg-blue-700 text-white"
          }`}
        >
          Aceptar
        </button>
      </div>
    </div>
  );
}
