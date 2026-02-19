import React, { useState, useEffect, useRef } from "react";
import { FiX, FiChevronDown } from "react-icons/fi";

export default function VoiceHelp({ isOpen, onClose, isDarkMode }) {
  const [visible, setVisible] = useState(false);
  const modalRef = useRef(null);
  const [expandedIndex, setExpandedIndex] = useState(null);

  useEffect(() => {
    if (isOpen) setVisible(true);
  }, [isOpen]);

  const handleClose = () => {
    setVisible(false);
    setTimeout(() => onClose(), 300);
  };

  useEffect(() => {
    const handleOutsideClick = (e) => {
      if (e.target === modalRef.current) handleClose();
    };
    window.addEventListener("click", handleOutsideClick);
    return () => window.removeEventListener("click", handleOutsideClick);
  }, []);

  const faqList = [
    {
      question: "¿Cómo inicio una conversación?",
      answer: "A través del micrófono hablaremos con el agente por voz, identifícate diciéndole tu ID de usuario.",
    },
    {
      question: "¿Cómo accedo a una conversación antigua?",
      answer: "A través de la opción de Historial de Conversaciones situada en la barra lateral, podremos seleccionar y cargar en el chat una conversación antigua con el agente de voz para continuarla.",
    },
    {
      question: "¿Cómo puedo guardar las conversaciones que más me interesen?",
      answer: "Usando el historial de conversaciones, podremos marcar las conversaciones como favoritas, a las que posteriormente podremos acceder a través de la opción de Favoritos situada también dentro del apartado Historial de conversaciones.",
    },
    {
      question: "¿Cómo puedo escribirle algo al agente?",
      answer: "Al tratarse de una conversación con un agente de voz, solamente se podrá hablar con el mismo por micrófono, para escribir un mensaje vuelva al chat de COSMOS.",
    },
    {
      question: "¿Puedo compartir mis conversaciones?",
      answer: "Sí, a través del botón de exportar en la página de la conversación, se descargará un archivo con la conversación completa con el agente de voz.",
    },
  ];

  const toggleIndex = (index) => {
    setExpandedIndex(expandedIndex === index ? null : index);
  };

  if (!isOpen) return null;

  return (
    <div
      ref={modalRef}
      className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50 transition-opacity duration-300 p-4"
    >
      <div
        className={`relative w-full max-w-lg md:max-w-3xl h-auto max-h-[85vh] flex flex-col rounded-3xl shadow-2xl transform transition-all duration-500 
        ${
          visible
            ? "opacity-100 translate-y-0 scale-100"
            : "opacity-0 -translate-y-6 scale-95"
        } 
        ${isDarkMode ? "bg-gray-900 text-white" : "bg-white text-blue-900"}`}
      >
        {/* Header */}
        <div className="flex justify-between items-center p-6 md:p-8 pb-4 md:pb-6 border-b border-transparent">
          <h2
            className={`text-2xl md:text-3xl font-extrabold ${
              isDarkMode ? "text-blue-400" : "text-blue-700"
            }`}
          >
            Centro de Ayuda
          </h2>
          <button
            onClick={handleClose}
            className="absolute top-2 right-2 bg-red-500 hover:bg-red-600 text-white rounded-full p-2 shadow transition-colors"
          >
            <FiX className="w-5 h-5" />
          </button>
        </div>

        {/* Preguntas y respuestas */}
        <div className="flex-1 overflow-y-auto px-6 md:px-8 pb-8 space-y-3 scrollbar-hide">
          {faqList.map((faq, index) => {
            const isExpanded = expandedIndex === index;
            return (
              <div
                key={index}
                className={`rounded-xl shadow-sm transition-colors duration-200 overflow-hidden ${
                  isDarkMode ? "bg-gray-800" : "bg-gray-100"
                }`}
              >
                <button
                  onClick={() => toggleIndex(index)}
                  className="w-full flex justify-between items-center text-left px-4 py-3 md:py-4 focus:outline-none hover:opacity-90 transition-opacity"
                >
                  <span
                    className={`text-sm md:text-base font-medium pr-4${
                      isDarkMode ? "text-white" : "text-gray-900"
                    }`}
                  >
                    {faq.question}
                  </span>
                  <FiChevronDown 
                    className={`flex-shrink-0 w-5 h-5 transition-transform duration-300 ease-in-out ${
                      isExpanded ? "rotate-180" : "rotate-0"
                    } ${isDarkMode ? "text-white" : "text-gray-800"}`} 
                  />
                </button>

                <div
                  className={`transition-all duration-300 ease-in-out overflow-hidden ${
                    isExpanded ? "max-h-96 opacity-100" : "max-h-0 opacity-0"
                  }`}
                >
                  <div className={`px-4 pb-4 pt-0 text-xs md:text-sm leading-relaxed ${isDarkMode ? "text-gray-300" : "text-gray-700"}`}>
                    {faq.answer}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}