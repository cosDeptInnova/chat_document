import React from 'react';
import { FiMic, FiMicOff, FiVolumeX } from "react-icons/fi";
import { PhoneOff } from "lucide-react";
import { Button } from "../utils/Button";
import { Slider } from "../utils/Slider";

export default function VoiceControls({
  micMuted,
  ttsVolume,
  toggleMicrophone,
  updateTtsVolume,
  handleEndCall,
  isDarkMode,
  className = ""
}) {

  const handleVolumePreset = (volume) => {
    updateTtsVolume(volume / 100);
  };

  return (
    <div
      className={`voice-controls-container w-full py-2 sm:py-4 px-2 sm:px-6 border-t z-50 ${
        isDarkMode
          ? "bg-gray-900 border-gray-700 shadow-[0_-4px_6px_-1px_rgba(0,0,0,0.2)]"
          : "bg-gray-50 border-gray-200 shadow-[0_-4px_15px_-3px_rgba(0,0,0,0.1)]"
      } ${className}`}
    >
      <div className="max-w-4xl mx-auto flex items-center justify-between gap-2 sm:gap-6">
        
        {/* BOTÓN MICROFONO */}
        <div className="flex flex-col items-center justify-center min-w-[3.5rem] sm:w-20">
          <Button
            onClick={toggleMicrophone}
            className={`w-12 h-12 sm:w-14 sm:h-14 rounded-full flex items-center justify-center border-2 transition-all duration-300 shadow-lg active:scale-95
              ${micMuted
                ? isDarkMode
                  ? "border-red-500/70 bg-red-500/10 text-red-400 hover:bg-red-500/20 hover:border-red-400 hover:text-red-300"
                  : "border-red-500/60 bg-red-100 text-red-600 hover:bg-red-200 hover:border-red-400 hover:text-red-700"
                : isDarkMode
                  ? "border-green-500/70 bg-green-500/10 text-green-400 hover:bg-green-500/20 hover:border-green-400 hover:text-green-300"
                  : "border-green-500/60 bg-green-100 text-green-600 hover:bg-green-200 hover:border-green-400 hover:text-green-700"
              }`}
            variant="ghost"
          >
            {micMuted ? <FiMicOff className="w-5 h-5 sm:w-6 sm:h-6" /> : <FiMic className="w-5 h-5 sm:w-6 sm:h-6" />}
          </Button>
          <span className={`text-[10px] mt-1 font-medium hidden sm:block ${micMuted ? "text-red-500" : "text-green-500"}`}>
             {micMuted ? "Muteado" : "Escuchando"}
          </span>
        </div>

        {/* CONTROLES DE VOLUMEN (Expandibles) */}
        <div className="flex-1 flex flex-col items-center justify-center px-2 sm:px-4 max-w-md">
           <div className="w-full flex items-center gap-3 bg-gray-500/5 rounded-full px-3 py-2 sm:py-3">
              <FiVolumeX className="w-4 h-4 text-gray-400 flex-shrink-0" onClick={() => updateTtsVolume(0)} cursor="pointer"/>
              
              <Slider
                value={[ttsVolume * 100]}
                onValueChange={([value]) => updateTtsVolume(value / 100)}
                max={100}
                step={1}
                className="flex-1 cursor-pointer relative flex items-center select-none touch-none w-full h-5"
                trackClassName={`relative grow rounded-full h-1.5 ${isDarkMode ? "bg-gray-600" : "bg-gray-300"}`}
                rangeClassName={`absolute h-full rounded-full ${isDarkMode ? "bg-blue-500" : "bg-blue-500"}`}
                thumbClassName={`block w-4 h-4 sm:w-5 sm:h-5 rounded-full border-2 focus:outline-none focus:ring-2 transition-transform hover:scale-110 ${
                   isDarkMode ? "bg-white border-blue-400" : "bg-white border-blue-500 shadow"
                }`}
              />
              
              <div className="flex items-center gap-1 min-w-[2.5rem] justify-end">
                 <span className={`text-[10px] sm:text-xs font-mono font-bold ${isDarkMode ? "text-blue-300" : "text-blue-600"}`}>
                    {Math.round(ttsVolume * 100)}%
                 </span>
              </div>
           </div>

           {/* Presets (Ocultos en móvil muy pequeño) */}
           <div className="w-full justify-between gap-2 mt-2 px-1 hidden sm:flex">
              {[25, 50, 75, 100].map(vol => (
                 <button
                    key={vol}
                    onClick={() => handleVolumePreset(vol)}
                    className={`flex-1 py-1 rounded text-[10px] font-medium transition-colors border
                       ${Math.round(ttsVolume * 100) === vol
                          ? isDarkMode ? "bg-blue-900/40 border-blue-500 text-blue-200" : "bg-blue-100 border-blue-300 text-blue-700"
                          : isDarkMode ? "border-transparent text-white hover:bg-gray-700" : "border-transparent text-black hover:bg-gray-200"
                       }
                    `}
                 >
                    {vol}%
                 </button>
              ))}
           </div>
        </div>

        {/* BOTÓN COLGAR */}
        <div className="flex flex-col items-center justify-center min-w-[3.5rem] sm:w-20">
          <Button
            onClick={() => { handleEndCall?.(); }}
            className={`w-12 h-12 sm:w-14 sm:h-14 rounded-full flex items-center justify-center border transition-all shadow-lg active:scale-95
              ${isDarkMode
                ? "border-red-500/70 bg-red-500/10 text-red-400 hover:bg-red-500/20 hover:border-red-400 hover:text-red-300"
                : "border-red-500/60 bg-red-100 text-red-600 hover:bg-red-200 hover:border-red-400 hover:text-red-700"
              }`}
            variant="ghost"
          >
            <PhoneOff className="w-5 h-5 sm:w-6 sm:h-6" />
          </Button>
          <span className={`text-[10px] mt-1 font-medium hidden sm:block ${isDarkMode ? "text-gray-400" : "text-gray-500"}`}>
             Salir
          </span>
        </div>

      </div>

      {/* Shortcuts (Solo visibles en Desktop MD+) */}
      <div className={`mt-2 text-center text-[9px] opacity-40 hidden md:block ${isDarkMode ? "text-gray-400" : "text-gray-500"}`}>
         <span className="mx-2">Ctrl+M: Mute</span> • <span className="mx-2">Ctrl+↑/↓: Volumen</span>
      </div>
    </div>
  );  
}