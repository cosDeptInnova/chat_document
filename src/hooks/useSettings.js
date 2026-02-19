import { useState, useEffect, createContext, useContext } from "react";

// Creamos el contexto
const SettingsContext = createContext();

export function SettingsProvider({ children }) {
  // Cargamos los sjustes en los estados al iniciar (para que se guarden al rrefrescar, cambiar de pestaña, abrir la app en el navegador.....)
  const saved = JSON.parse(localStorage.getItem("app_settings") || "{}");

  const [darkMode, setDarkMode] = useState(saved.darkMode ?? false);
  const [volume, setVolume] = useState(saved.volume ?? 50);
  const [speed, setSpeed] = useState(saved.speed ?? 1);
  const [tone, setTone] = useState(saved.tone ?? 0);
  const [language, setLanguage] = useState(saved.language ?? "es");

  // Guardar todos los ajustes juntos
  useEffect(() => {
    const settings = { darkMode, volume, speed, tone, language };
    localStorage.setItem("app_settings", JSON.stringify(settings));
  }, [darkMode, volume, speed, tone, language]);

  // Valor que estará disponible en toda la interfaz
  const value = {
    darkMode,
    setDarkMode,
    volume,
    setVolume,
    speed,
    setSpeed,
    tone,
    setTone,
    language,
    setLanguage,
  };

  return (
    <SettingsContext.Provider value={value}>
      {children}
    </SettingsContext.Provider>
  );
}

// Hook personalizado para acceder a los ajustes
export function useSettings() {
  return useContext(SettingsContext);
}
