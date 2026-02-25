import React, { useState } from 'react';
import { rateMessage } from '../../lib/api'; 

export default function MessageFeedback ({ messageId, initialLiked, isDarkMode }) {
  const [liked, setLiked] = useState(initialLiked);
  const [loading, setLoading] = useState(false);

  const handleRate = async (newStatus) => {
    if (loading) return;

    // Si pulsamos lo que ya estaba marcado, lo quitamos (toggle a null)
    const statusToSend = (liked === newStatus) ? null : newStatus;

    // Actualización visual inmediata
    const previousState = liked;
    setLiked(statusToSend);
    setLoading(true);

    try {
      await rateMessage(messageId, statusToSend);
    } catch (error) {
      console.error("Error enviando feedback:", error);
      // Si falla, volvemos al estado anterior
      setLiked(previousState);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex items-center gap-2 border-l border-gray-300 dark:border-gray-600 pl-2 ml-1">
      {/* Botón LIKE */}
      <button
        type="button"
        onClick={() => handleRate(true)}
        className={`text-[10px] sm:text-2xs transition-colors duration-200 ${
          liked === true
            ? "text-green-500" 
            : isDarkMode 
              ? "text-white hover:text-green-400" 
              : "text-gray-400 hover:text-green-600"
        }`}
        title="Me gusta"
      >
        <i className="fas fa-thumbs-up" />
      </button>

      {/* Botón DISLIKE */}
      <button
        type="button"
        onClick={() => handleRate(false)}
        className={`text-[10px] sm:text-2xs transition-colors duration-200 ${
          liked === false
            ? "text-red-500" 
            : isDarkMode 
              ? "text-white hover:text-red-400" 
              : "text-gray-400 hover:text-red-600"
        }`}
        title="No me gusta"
      >
        <i className="fas fa-thumbs-down" />
      </button>
    </div>
  );
};