import React, { useState, useEffect } from "react";
import Tabs from "../components/use-Cases/Tabs";
import CardGrid from "../components/use-Cases/CardGrid";
import { useCases } from "../data/useCases";

export default function CasosUsoMainPanel({ isDarkMode }) {
  const [activeTab, setActiveTab] = useState("Más usados");
  const [selectedCard, setSelectedCard] = useState(null);

  // Resetear card seleccionada cuando cambias de tab
  useEffect(() => {
    const filtered = useCases
      .filter((u) => u.departments.includes(activeTab))
      .sort((a, b) => activeTab === "Más usados" ? b.popularity - a.popularity : 0);
      
    if (filtered.length > 0) {
      setSelectedCard(filtered[0].title);
    }
  }, [activeTab]);

  return (
    <main className={`flex-1 flex flex-col h-full w-full overflow-hidden ${isDarkMode ? "bg-gray-900" : "bg-white"} text-sm`}>
      {/* Contenedor con scroll interno */}
      <div className="flex-1 overflow-y-auto scrollbar-hide transition-all duration-300 ease-in-out
       px-4 md:px-10 2xl:px-20
       py-1 ">
        <div className="space-y-0 md:space-y-1 2xl:space-y-2 text-left">
            <h1 className={`font-extrabold text-xl md:text-2xl 2xl:text-4xl ${isDarkMode ? "text-blue-400" : "text-blue-700"} transition-all duration-300`}>
            Cosmos: más allá de la nube
            </h1>

            <p className={`${isDarkMode ? "text-gray-300" : "text-gray-600"} text-xs md:text-base 2xl:text-base max-w-3xl transition-all duration-300`}>
            Descubre nuestros casos de uso favoritos para aplicar nuestra IA
            </p>
        </div>

        <Tabs isDarkMode={isDarkMode} activeTab={activeTab} setActiveTab={setActiveTab} />

        <hr className={`my-1 ${isDarkMode ? "border-gray-600" : "border-gray-300"}`} />

        <div className="pb-10">
          <CardGrid
            isDarkMode={isDarkMode}
            activeTab={activeTab}
            selectedCard={selectedCard}
            setSelectedCard={setSelectedCard}
          />
        </div>
      </div>
    </main>
  );
}