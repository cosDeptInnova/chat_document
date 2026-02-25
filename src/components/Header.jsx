import React, { useState, useRef, useEffect } from 'react';
import UserInfoModal from './utils/UserInfoModal';
import ChangePasswordModal from './utils/ChangePasswordModal';
import LogoutConfirmModal from './utils/LogoutConfirmModal';
import imagenPerfil from '../images/imagenPerfil.jfif';
import { useNavigate } from 'react-router-dom';
import { logout as apiLogout } from '../lib/api';

export default function Header({ isDarkMode, setIsDarkMode, user, setUser, extraContent }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [showUserInfo, setShowUserInfo] = useState(false);
  const [showChangePassword, setShowChangePassword] = useState(false);
  const [showLogoutConfirm, setShowLogoutConfirm] = useState(false);
  const menuRef = useRef(null);
  const navigate = useNavigate();

  // Mostramos el nombre de usuario en el Header igual que en el UserInfoModal
  const displayName = 
    user?.full_name || 
    user?.name || 
    user?.username || 
    user?.email || 
    "Invitado";

  // Cerrar menú al clicar fuera
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (menuRef.current && !menuRef.current.contains(event.target)) {
        setMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const handleLogout = async () => {
    try {
      // Notificar al backend / auth_sso
      await apiLogout();
    } catch (err) {
      console.error("Error al hacer logout en backend:", err);
    } finally {
      // Limpieza local
      setUser(null);
      localStorage.removeItem("user");
      navigate("/login");
    }
  };

  return (
    <div className={`flex justify-between items-center px-4 md:px-10 py-4 ${isDarkMode ? "bg-gray-900" : "bg-white"}`}>
      {/* Parte izquierda */}
      <div className="flex items-center gap-3 md:gap-6 flex-1 min-w-0 mr-2">
        {/* La variable extraContent sirve para pintar información extra en el Header (en VoiceAgent) */}
        {extraContent && (
          <div className="flex items-center space-x-2 md:space-x-3 min-w-0 overflow-hidden">
            {extraContent}
          </div>
        )}
      
        {/* Toggle modo oscuro */}
        <div className={`relative flex items-center justify-between rounded-full px-1 py-1 w-20 md:w-24 h-10 md:h-12 flex-shrink-0 transition-colors duration-300 ${isDarkMode ? "bg-gray-800" : "bg-gray-200"}`}>
          <div className={`absolute top-1 left-1 w-8 h-8 md:w-10 md:h-10 rounded-full shadow-md bg-white transform transition-transform duration-300 ease-in-out
            ${isDarkMode ? "translate-x-0" : "translate-x-10 md:translate-x-12"}`} />
          <button onClick={() => setIsDarkMode(true)} className={`z-10 p-1 md:p-2 transition-transform duration-300 ${isDarkMode ? "scale-125 text-yellow-300" : "text-gray-600 hover:scale-110"}`}>
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill={isDarkMode ? "currentColor" : "none"} viewBox="0 0 24 24" stroke="currentColor">
              <path d="M21 12.79A9 9 0 0111.21 3a7 7 0 100 14 9 9 0 009.79-4.21z" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
          <button onClick={() => setIsDarkMode(false)} className={`z-10 p-1 md:p-2 transition-transform duration-300 ${!isDarkMode ? "scale-125 text-yellow-500" : "text-gray-400 hover:scale-110"}`}>
            <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill={!isDarkMode ? "currentColor" : "none"} viewBox="0 0 24 24" stroke="currentColor">
              <path d="M12 4V2m0 20v-2m8-8h2M2 12h2m15.07-7.07l1.42-1.42M4.93 19.07l1.42-1.42M19.07 19.07l1.42 1.42M4.93 4.93l1.42 1.42" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <circle cx="12" cy="12" r="5" stroke="currentColor" strokeWidth="2"/>
            </svg>
          </button>
        </div>
      </div>

      {/* Info usuario (Ocultamos el nombre en pantallas pequeñas como la de móvil) */}
      <div className="relative flex-shrink-0" ref={menuRef}>
        <div className="flex items-center gap-2 cursor-pointer" onClick={() => setMenuOpen(!menuOpen)}>
          <span className={`hidden sm:block max-w-[150px] md:max-w-[300px] whitespace-nowrap truncate ${isDarkMode ? "text-white" : "text-gray-800"}`}>
            Hola,{" "}
            <span className="text-blue-600 font-semibold">
              {displayName}
            </span>
          </span>
          <img
            src={imagenPerfil}
            alt="avatar"
            className="w-10 h-10 rounded-full object-cover flex-shrink-0 border-2 border-transparent hover:border-blue-500 transition-colors"
          />
        </div>

        {/* Menú desplegable */}
        {menuOpen && (
          <div
            className={`absolute right-0 mt-2 w-56 rounded-xl shadow-2xl py-2 z-50 border overflow-hidden transition-all duration-300 transform origin-top-right
              ${isDarkMode ? "bg-gray-800 border-gray-700" : "bg-white border-blue-100"}`}
          >
            <button
              className={`w-full text-left px-4 py-3 flex items-center transition-colors duration-200 group
                ${isDarkMode ? "hover:bg-blue-900/20 text-gray-200" : "hover:bg-blue-50 text-gray-800"}`}
              onClick={() => {
                setShowUserInfo(true);
                setMenuOpen(false);
              }}
            >
              <svg
                className={`w-5 h-5 mr-3 group-hover:scale-110 transition-transform ${
                  isDarkMode ? "text-blue-400" : "text-blue-600"
                }`}
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
              Información personal
            </button>

            <div className={`${isDarkMode ? "border-gray-700" : "border-gray-200"} border-t my-1`}></div>

            <button
              className={`w-full text-left px-4 py-3 flex items-center transition-colors duration-200 group
                ${isDarkMode ? "hover:bg-red-900/20 text-red-400" : "hover:bg-red-50 text-red-600"}`}
              onClick={() => {
                setShowLogoutConfirm(true);
                setMenuOpen(false);
              }}
            >
              <svg
                className="w-5 h-5 mr-3 group-hover:scale-110 transition-transform"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M17 16l4-4m0 0l-4-4m4 4H7m6 4v1a3 3 0 01-3 3H6a3 3 0 01-3-3V7a3 3 0 013-3h4a3 3 0 013 3v1"
                />
              </svg>
              Cerrar sesión
            </button>
          </div>
        )}
      </div>

      {/* Modales */}
      <UserInfoModal 
        user={user} 
        onClose={() => setShowUserInfo(false)} 
        isOpen={showUserInfo}
        isDarkMode={isDarkMode}
      />
      <ChangePasswordModal
        onClose={() => setShowChangePassword(false)}
        isOpen={showChangePassword}
        isDarkMode={isDarkMode}
      />
      <LogoutConfirmModal
        onConfirm={handleLogout}
        onClose={() => setShowLogoutConfirm(false)}
        isOpen={showLogoutConfirm}
        isDarkMode={isDarkMode}
      />
    </div>
  );
}
