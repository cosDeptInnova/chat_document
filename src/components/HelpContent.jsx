import React, { useState } from "react";

export default function HelpContent({ isDarkMode }) {
  const [expandedIndex, setExpandedIndex] = useState(null);

  const faqList = [
    { question: "¿Cómo inicio una nueva conversación?", answer: "Usaremos una de las opciones de la barra lateral, en concreto la opción de Nuevo Chat, que nos permitirá abrir una nueva conversación con COSMOS." },
    { question: "¿Cómo accedo a una convesacíon antigua?", answer: "A través de la opción de Accede a tu historial situada en la barra lateral, podremos seleccionar y cargar en el chat una conversación antigua para poder seguir con ella." },
    { question: "¿Cómo puedo guardar las conversaciones que más me interesen?", answer: "Usando el historial, podremos marcar las conversaciones como favoritas, a las que posteriormente podremos acceder a través de la opción de Favoritos situada también dentro del apartado Accede a tu historial, dentro de un deslizable." },
    { question: "¿Cómo uso las funcionalidades de COSMOS?", answer: "Mediante la opción de Casos de uso de la barra lateral, podremos acceder a las distintas funcionalidades que nos ofrece COSMOS en forma de tarjetas, que además se podran filtrar por departamentos." },
    { question: "¿Cómo cargo documentos a la base de conocimientos?", answer: "A través de la opción de Base de Conocimientos, situada en la barra lateral, podremos acceder a una pantalla que nos permitirá cargar y subir archivos a COSMOS tanto como para el directorio personal cómo para el departamental." },
  ];

  const toggleIndex = (index) => {
    setExpandedIndex(expandedIndex === index ? null : index);
  };

  return (
    <div className="space-y-2">
      {faqList.map((faq, index) => {
        const isExpanded = expandedIndex === index;
        return (
          <div
            key={index}
            className={`rounded-lg shadow-sm transition-colors duration-200 ${
              isDarkMode ? "bg-gray-700" : "bg-gray-100"
            }`}
          >
            <button
              onClick={() => toggleIndex(index)}
              className="w-full flex justify-between items-center text-left px-4 py-3 focus:outline-none"
            >
              <span
                className={`text-sm font-medium ${
                  isDarkMode ? "text-white" : "text-gray-900"
                }`}
              >
                {faq.question}
              </span>

              <i
                className={`fas ${
                  isExpanded ? "fa-chevron-up" : "fa-chevron-down"
                } text-sm transform transition-all duration-300 ease-in-out ${
                  isExpanded ? "translate-y-[-2px]" : "translate-y-[0px]"
                } ${isDarkMode ? "text-white" : "text-gray-800"}`}
              />
            </button>

            <div
              className={`px-4 text-sm overflow-hidden transition-all duration-300 ease-in-out ${
                isExpanded ? "max-h-40 opacity-100 py-2" : "max-h-0 opacity-0 py-0"
              } ${isDarkMode ? "text-gray-300" : "text-gray-700"}`}
            >
              {faq.answer}
            </div>
          </div>
        );
      })}
    </div>
  );
}