import React from "react";
import { useSettings } from "../hooks/useSettings";

export default function SettingsContent({ isDarkMode }) {
  const {
    darkMode, setDarkMode,
    volume, setVolume,
    speed, setSpeed,
    tone, setTone,
    language, setLanguage
  } = useSettings();

  return (
    <div className="space-y-2 md:space-y-4 animate-fadeIn pb-4">
      {/* Preferencia de modo oscuro */}
      <div
        className={`p-1.5 md:p-2 rounded-xl shadow transition-colors duration-300 ${
          isDarkMode ? "bg-gray-700 text-white" : "bg-gray-100 text-gray-900"
        }`}
      >
        <div className="flex justify-between items-center">
          <span className="text-sm md:text-base font-medium">Preferencia de modo oscuro</span>
          <button
            onClick={() => setDarkMode(!darkMode)}
            className={`w-12 h-6 md:w-14 md:h-7 flex items-center rounded-full p-1 transition-colors duration-300 ${
              darkMode ? "bg-blue-500" : "bg-gray-400"
            }`}
          >
            <div
              className={`w-4 h-4 md:w-5 md:h-5 bg-white rounded-full shadow-md transform transition-transform duration-300 ${
                darkMode ? "translate-x-6 md:translate-x-7" : "translate-x-0"
              }`}
            />
          </button>
        </div>
      </div>

      {/* Ajustes de voz */}
      <div
        className={`p-2 md:p-4 rounded-xl shadow space-y-3 md:space-y-4 ${
          isDarkMode ? "bg-gray-700 text-white" : "bg-gray-100 text-gray-900"
        }`}
      >
        <h4 className="font-semibold text-base md:text-lg mb-2 md:mb-4 border-b border-gray-500/20 pb-2">Ajustes de voz</h4>

        <div>
          <label className="block text-sm mb-1">Volumen: {volume}%</label>
          <input
            type="range"
            min="0"
            max="100"
            value={volume}
            onChange={(e) => setVolume(parseInt(e.target.value))}
            className="w-full accent-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm mb-1">
            Velocidad: {speed.toFixed(1)}x
          </label>
          <input
            type="range"
            min="0.5"
            max="2"
            step="0.1"
            value={speed}
            onChange={(e) => setSpeed(parseFloat(e.target.value))}
            className="w-full accent-blue-500"
          />
        </div>

        <div>
          <label className="block text-sm mb-1">Tono: {tone}</label>
          <input
            type="range"
            min="-10"
            max="10"
            value={tone}
            onChange={(e) => setTone(parseInt(e.target.value))}
            className="w-full accent-blue-500"
          />
        </div>

        {/*
        <div>
          <label className="block text-sm mb-1">Idioma de voz</label>
          <select
            value={language}
            onChange={(e) => setLanguage(e.target.value)}
            className={`w-full px-3 py-2 rounded-lg border shadow-sm focus:ring-2 focus:ring-blue-500 outline-none ${
              isDarkMode
                ? "bg-gray-600 text-white border-gray-500"
                : "bg-white text-gray-900 border-gray-300"
            }`}
          >
            <option value="es">Español</option>
            <option value="en">Inglés</option>
            <option value="fr">Francés</option>
            <option value="de">Alemán</option>
            <option value="it">Italiano</option>
            <option value="pt">Portugués</option>
            <option value="ja">Japonés</option>
            <option value="zh">Chino</option>
          </select>
          <p className={`text-xs mt-1 ${isDarkMode ? "text-gray-300" : "text-gray-500"}`}>
            El idioma seleccionado se aplicará únicamente a la reproducción de voz de los mensajes.
          </p>
        </div>*/}
      </div>
    </div>
  );
}