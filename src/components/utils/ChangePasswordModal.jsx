import React, { useState, useEffect, useRef } from "react";
import PasswordField from "./PasswordField";

export default function ChangePasswordModal({ onClose, isOpen, isDarkMode }) {
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [isVisible, setIsVisible] = useState(false);

  const [showCurrentPassword, setShowCurrentPassword] = useState(false);
  const [showNewPassword, setShowNewPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  const [message, setMessage] = useState({ text: "", type: "" });

  const modalRef = useRef(null);

  useEffect(() => {
    if (isOpen) {
      setTimeout(() => setIsVisible(true), 10);
    } else {
      setIsVisible(false);
    }
  }, [isOpen]);

  useEffect(() => {
    function handleOutsideClick(e) {
      if (modalRef.current && !modalRef.current.contains(e.target)) {
        handleClose();
      }
    }
    if (isOpen) document.addEventListener("mousedown", handleOutsideClick);
    return () => document.removeEventListener("mousedown", handleOutsideClick);
  }, [isOpen]);

  const handleClose = () => {
    setIsVisible(false);
    setTimeout(() => {
      setCurrentPassword("");
      setNewPassword("");
      setConfirmPassword("");
      setShowCurrentPassword(false);
      setShowNewPassword(false);
      setShowConfirmPassword(false);
      setMessage({ text: "", type: "" });
      onClose();
    }, 300);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (newPassword !== confirmPassword) {
      setMessage({ text: "Las contraseñas no coinciden.", type: "error" });
      return;
    }
    setMessage({ text: "Contraseña actualizada correctamente.", type: "success" });
    setTimeout(handleClose, 1500);
  };

  if (!isOpen && !isVisible) return null;

  return (
    <div
      className={`fixed inset-0 z-50 flex items-center justify-center bg-opacity-60 backdrop-blur-sm transition-opacity duration-300 p-4 ${
        isVisible ? "opacity-100" : "opacity-0 pointer-events-none"
      }`}
    >
      <div
        ref={modalRef}
        className={`${
          isDarkMode ? "bg-gray-800 text-white" : "bg-white text-gray-900"
        } rounded-2xl p-6 w-96 shadow-2xl relative transform transition-all duration-300 ${
          isVisible ? "scale-100 opacity-100" : "scale-95 opacity-0"
        }`}
      >
        <h2 className={`text-2xl font-bold mb-6 flex items-center ${isDarkMode ? "text-blue-200" : "text-blue-800"}`}>
          <svg className="w-6 h-6 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 7a2 2 0 012 2m4 0a6 6 0 01-7.743 5.743L11 17H9v2H7v2H4a1 1 0 01-1-1v-2.586a1 1 0 01.293-.707l5.964-5.964A6 6 0 1121 9z"/>
          </svg>
          Cambiar contraseña
        </h2>

        {message.text && (
          <p className={`text-sm mb-3 ${message.type === "error" ? "text-red-500" : "text-green-500"}`}>
            {message.text}
          </p>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
         <PasswordField
            label="Contraseña actual"
            value={currentPassword}
            onChange={setCurrentPassword}
            showPassword={showCurrentPassword}
            toggleShowPassword={() => setShowCurrentPassword(prev => !prev)}
            isDarkMode={isDarkMode}
            autoComplete="current-password"
          />

          <PasswordField
            label="Nueva contraseña"
            value={newPassword}
            onChange={setNewPassword}
            showPassword={showNewPassword}
            toggleShowPassword={() => setShowNewPassword(prev => !prev)}
            isDarkMode={isDarkMode}
            autoComplete="new-password"
          />

          <PasswordField
            label="Confirmar nueva contraseña"
            value={confirmPassword}
            onChange={setConfirmPassword}
            showPassword={showConfirmPassword}
            toggleShowPassword={() => setShowConfirmPassword(prev => !prev)}
            isDarkMode={isDarkMode}
            autoComplete="new-password"
          />

          <div className="flex justify-end gap-3 pt-2">
            <button
              type="button"
              className={`px-5 py-2.5 rounded-xl transition-colors duration-200 font-medium ${
                isDarkMode ? "bg-gray-700 hover:bg-gray-600 text-white" : "bg-gray-200 hover:bg-gray-300 text-gray-800"
              }`}
              onClick={handleClose}
            >
              Cancelar
            </button>
            <button
              type="submit"
              className="px-5 py-2.5 rounded-xl bg-blue-600 hover:bg-blue-700 text-white transition-colors duration-200 font-medium shadow-md hover:shadow-lg"
            >
              Actualizar
            </button>
          </div>
        </form>

        <button
          onClick={handleClose}
          className="absolute top-3 right-3 bg-red-600 hover:bg-red-700 text-white rounded-full p-2 shadow"
        >
          <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12"/>
          </svg>
        </button>
      </div>
    </div>
  );
}