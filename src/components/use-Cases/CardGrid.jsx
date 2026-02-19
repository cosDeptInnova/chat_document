import React from "react";
import Card from "./Card";
import { useCases } from "../../data/useCases";

export default function CardGrid({ isDarkMode, activeTab, selectedCard, setSelectedCard }) {
  const filtered = useCases
  .filter(useCase => useCase.departments.includes(activeTab))
  .sort((a, b) => activeTab === "Más usados" ? b.popularity - a.popularity : 0);

  return (
    <div className="p-4 grid gap-3 md:gap-4 grid-cols-1 md:grid-cols-3 2xl:grid-cols-5">
      {filtered.map((useCase, idx) => (
        <Card
          key={idx}
          useCase={useCase}
          isDarkMode={isDarkMode}
          active={selectedCard === useCase.title}
          onClick={() => setSelectedCard(useCase.title)}
        />
      ))}
    </div>
  );
}
