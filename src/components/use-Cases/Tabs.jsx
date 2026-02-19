import React from "react";

export default function Tabs({ isDarkMode, activeTab, setActiveTab }) {
  const tabs = ['Más usados', 'RRHH', 'Administración', 'Comunicación', 'Comercial', 'Agen-tic'];

  return (
    <div className="w-full p-2">
      <div className="grid grid-cols-2 gap-3 md:flex md:flex-wrap md:justify-left md:gap-4 2xl:gap-6">
        {tabs.map(tab => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            aria-selected={activeTab === tab}
            className={`
              w-full md:w-32
              px-2 py-2.5 md:px-6 md:py-2.5 
              rounded-md border transition-all duration-200 
              text-xs md:text-sm font-medium whitespace-nowrap
              flex items-center justify-center
              ${activeTab === tab
                ? 'bg-blue-600 text-white border-blue-600 shadow-md transform scale-105'
                : isDarkMode
                  ? 'bg-gray-800 text-gray-300 border-gray-700 hover:bg-gray-700 hover:text-white'
                  : 'bg-white text-gray-600 border-gray-200 hover:bg-gray-50 hover:text-blue-600 hover:border-blue-200'
              }
            `}
          >
            {tab}
          </button>
        ))}
      </div>
    </div>
  );
}