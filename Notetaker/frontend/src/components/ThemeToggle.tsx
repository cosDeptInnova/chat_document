import { SunIcon, MoonIcon } from '@heroicons/react/24/outline';
import { useTheme } from '../context/ThemeContext';

export const ThemeToggle = () => {
  const { theme, toggleTheme } = useTheme();

  return (
    <button
      onClick={toggleTheme}
      type="button"
      className="relative inline-flex h-8 w-14 items-center rounded-full transition-all duration-300 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2"
      style={{
        backgroundColor: theme === 'dark' ? '#0066FF' : '#E5E7EB',
      }}
      aria-label={theme === 'light' ? 'Cambiar a modo oscuro' : 'Cambiar a modo claro'}
    >
      {/* Icono de sol - siempre visible a la izquierda */}
      <div className="absolute left-1.5 flex items-center justify-center z-10">
        <SunIcon 
          className={`h-4 w-4 transition-colors duration-300 ${
            theme === 'light' 
              ? 'text-white' // Sol en blanco cuando es día (fondo gris)
              : 'text-gray-300 opacity-60' // Sol tenue cuando es noche (fondo azul)
          }`} 
        />
      </div>
      
      {/* Icono de luna - siempre visible a la derecha */}
      <div className="absolute right-1.5 flex items-center justify-center z-10">
        <MoonIcon 
          className={`h-4 w-4 transition-colors duration-300 ${
            theme === 'dark' 
              ? 'text-white' // Luna en blanco cuando es noche (fondo azul)
              : 'text-gray-600 opacity-70' // Luna oscura cuando es día (fondo gris)
          }`} 
        />
      </div>
      
      {/* Círculo deslizante */}
      <span
        className={`inline-flex items-center justify-center h-6 w-6 transform rounded-full bg-white shadow-md transition-transform duration-300 z-20 ${
          theme === 'dark' ? 'translate-x-7' : 'translate-x-1'
        }`}
      />
    </button>
  );
};
