import React from 'react';

// Helper para conseguir colores e iconos según el archivo
const getIconForFile = (fileName) => {
  const ext = fileName.split('.').pop().toLowerCase();
  if (['pdf'].includes(ext)) return 'fa-file-pdf text-red-500';
  if (['doc', 'docx'].includes(ext)) return 'fa-file-word text-blue-500';
  if (['xls', 'xlsx', 'csv'].includes(ext)) return 'fa-file-excel text-green-500';
  if (['jpg', 'jpeg', 'png', 'gif'].includes(ext)) return 'fa-file-image text-purple-500';
  if (['txt'].includes(ext)) return 'fa-file-alt text-gray-500';
  return 'fa-file text-gray-400';
};

export const SourceChips = ({ sources, onSourceClick }) => {
  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-2 md:mt-2.5 2xl:mt-3 pt-2 md:pt-2.5 2xl:pt-3 border-t border-gray-100 dark:border-gray-700/50">
      
      {/* Contenedor del título y el icono */}
      <div className="flex items-center gap-1 md:gap-1.5 mb-1.5 md:mb-2 text-gray-500 dark:text-gray-400">
        {/* Icono de la lupa: Original era text-[10px] */}
        <i className="fas fa-search text-[8px] md:text-[9px] 2xl:text-[10px]"></i> 
        
        {/* Texto "Fuentes utilizadas:": Original era text-xs */}
        <span className="text-[10px] md:text-[11px] 2xl:text-xs font-semibold uppercase tracking-wide">
          Fuentes utilizadas:
        </span>
      </div>
      
      {/* Contenedor de los chips: Original gap era gap-2 */}
      <div className="flex flex-wrap gap-1.5 md:gap-2">
        {sources.map((source, idx) => {
          const iconClass = getIconForFile(source.file_name);

          return (
            <button
              key={`${source.file_id}-${idx}`}
              onClick={() => onSourceClick(source)}
              // Original: px-3 py-1.5 ... max-w-xs
              className="group flex items-center gap-1.5 md:gap-2 px-2 py-1 md:px-2.5 md:py-1 2xl:px-3 2xl:py-1.5 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 hover:border-blue-400 dark:hover:border-blue-500 hover:shadow-sm rounded-md transition-all text-left max-w-[160px] sm:max-w-[200px] md:max-w-[280px] 2xl:max-w-xs"
              title="Clic para ver el documento"
            >
              {/* Contenedor del icono: Original era w-8 h-8 */}
              <div className="w-5 h-5 md:w-6 md:h-6 2xl:w-8 2xl:h-8 flex items-center justify-center bg-gray-100 dark:bg-gray-900/50 rounded shrink-0 group-hover:scale-110 transition-transform">
                 {/* Icono del archivo: Original era text-xl */}
                 <i className={`fas ${iconClass} text-sm md:text-base 2xl:text-xl`}></i>
              </div>
              
              {/* Textos del chip */}
              <div className="flex flex-col overflow-hidden">
                {/* Nombre de archivo: Original NO tenía text- size (heredaba text-xs del padre) */}
                <span className="font-medium text-[10px] md:text-[11px] 2xl:text-xs text-gray-700 dark:text-gray-200 truncate block w-full">
                  {source.file_name}
                </span>
                {/* Subtítulo página: Original era text-[10px] */}
                <span className="text-[8px] md:text-[9px] 2xl:text-[10px] text-gray-500 dark:text-gray-400 mt-[1px]">
                  {source.page ? `Página ${source.page}` : 'Documento'}
                </span>
              </div>
            </button>
          )
        })}
      </div>
    </div>
  );
};