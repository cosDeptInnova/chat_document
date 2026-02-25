import React from 'react';
import { Trash2, Download, FileText, Image, File, Share2, Square, CheckSquare, ChevronUp, ChevronDown } from 'lucide-react';

const FileIcon = ({ type, className = "w-4 h-4 2xl:w-5 2xl:h-5" }) => {
    const t = type?.toLowerCase() || '';
    if (t.includes('pdf')) return <FileText className={`${className} text-red-500`} />;
    if (t.includes('word') || t.includes('doc')) return <FileText className={`${className} text-blue-500`} />;
    if (t.includes('image') || t.includes('png') || t.includes('jpg')) return <Image className={`${className} text-purple-500`} />;
    return <File className={`${className} text-gray-400`} />;
};

export const RagFileList = ({
    files, 
    onDelete,
    onDownload,
    onShare,
    context,
    selectedFiles, 
    onSelectFile,
    sortConfig,
    onSort,
    loading,
    isDarkMode
}) => {
    
    // 1. Estado de carga
    if (loading) {
        return (
            <div className="flex flex-col items-center justify-center py-12 md:py-16 2xl:py-20 text-slate-400">
                <div className="animate-spin rounded-full h-6 w-6 md:h-7 md:w-7 2xl:h-8 2xl:w-8 border-b-2 border-blue-600 mb-3 md:mb-4"></div>
                <p className="font-medium text-sm md:text-base">Consultando base de conocimientos...</p>
            </div>
        );
    }

    // 2. Estado vacío (sin archivos)
    if (!files || files.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center py-12 md:py-16 2xl:py-20 text-slate-400">
                <File className="w-8 h-8 md:w-10 md:h-10 2xl:w-12 2xl:h-12 mb-3 md:mb-4 opacity-20" />
                <p className="font-medium text-sm md:text-base">No se encontraron documentos en este destino</p>
            </div>
        );
    }

    const isMultiSelection = selectedFiles.size > 1;

    const SortIndicator = ({ columnKey }) => {
        if (sortConfig.key !== columnKey) return null;
        return sortConfig.direction === 'asc' ? 
            <ChevronUp className="w-3 h-3 md:w-3.5 md:h-3.5 2xl:w-4 2xl:h-4 ml-1 inline-block" /> : 
            <ChevronDown className="w-3 h-3 md:w-3.5 md:h-3.5 2xl:w-4 2xl:h-4 ml-1 inline-block" />;
    };

    return (
        <div className={`w-full ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
            <table className="min-w-full border-separate border-spacing-0">
                <thead>
                    <tr>
                        {/* Cabecera Checkbox */}
                        <th scope="col" className={`sticky top-0 z-20 px-3 py-3 md:px-4 md:py-3.5 2xl:px-6 2xl:py-4 border-b w-10 md:w-12 2xl:w-12 ${
                            isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-slate-100'
                        }`}>
                        </th>
                        
                        {/* Cabecera Nombre */}
                        <th scope="col" className={`sticky top-0 z-20 px-2 py-3 md:px-4 md:py-3.5 2xl:px-6 2xl:py-4 text-left text-[10px] md:text-[11px] 2xl:text-xs font-bold uppercase cursor-pointer whitespace-nowrap group ${
                            isDarkMode 
                            ? 'bg-gray-800 text-gray-400 hover:text-blue-400 border-gray-700' 
                            : 'bg-white text-slate-400 hover:text-blue-600 border-slate-100'
                        }`} onClick={() => onSort('name')}>
                            <span className="flex items-center">
                                Nombre del Archivo
                                <div className={`${sortConfig.key === 'name' ? 'opacity-100' : 'opacity-0 group-hover:opacity-40'}`}>
                                    <SortIndicator columnKey="name" />
                                    {sortConfig.key !== 'name' && <ChevronUp className="w-3 h-3 md:w-3.5 md:h-3.5 2xl:w-4 2xl:h-4 ml-1 inline-block opacity-0 group-hover:opacity-100" />}
                                </div>
                            </span>
                        </th>

                        {/* Cabeceras ocultas en móvil (hidden md:table-cell) */}
                        {context === 'department' && (
                            <th scope="col" className={`hidden md:table-cell sticky top-0 z-20 px-3 py-3 md:px-4 md:py-3.5 2xl:px-6 2xl:py-4 text-left text-[10px] md:text-[11px] 2xl:text-xs font-bold uppercase border-b ${
                                isDarkMode ? 'bg-gray-800 text-gray-400 border-gray-700' : 'bg-white text-slate-400 border-slate-100'
                            }`}>
                                Subido por
                            </th>
                        )}

                        <th scope="col" className={`hidden md:table-cell sticky top-0 z-20 px-3 py-3 md:px-4 md:py-3.5 2xl:px-6 2xl:py-4 text-left text-[10px] md:text-[11px] 2xl:text-xs font-bold uppercase cursor-pointer group ${
                            isDarkMode 
                            ? 'bg-gray-800 text-gray-400 hover:text-blue-400 border-gray-700' 
                            : 'bg-white text-slate-400 hover:text-blue-600 border-slate-100'
                        }`} onClick={() => onSort('date')}>
                            <span className="flex items-center">
                                Fecha y Hora
                                <div className={`${sortConfig.key === 'date' ? 'opacity-100' : 'opacity-0 group-hover:opacity-40'}`}>
                                    <SortIndicator columnKey="date" />
                                    {sortConfig.key !== 'date' && <ChevronUp className="w-3 h-3 md:w-3.5 md:h-3.5 2xl:w-4 2xl:h-4 ml-1 inline-block opacity-0 group-hover:opacity-100" />}
                                </div>
                            </span>
                        </th>

                        <th scope="col" className={`hidden md:table-cell sticky top-0 z-20 px-3 py-3 md:px-4 md:py-3.5 2xl:px-6 2xl:py-4 text-left text-[10px] md:text-[11px] 2xl:text-xs font-bold uppercase cursor-pointer group ${
                            isDarkMode 
                            ? 'bg-gray-800 text-gray-400 hover:text-blue-400 border-gray-700' 
                            : 'bg-white text-slate-400 hover:text-blue-600 border-slate-100'
                        }`} onClick={() => onSort('size')}>
                            <span className="flex items-center">
                                Tamaño
                                <div className={`${sortConfig.key === 'size' ? 'opacity-100' : 'opacity-0 group-hover:opacity-40'}`}>
                                    <SortIndicator columnKey="size" />
                                    {sortConfig.key !== 'size' && <ChevronUp className="w-3 h-3 md:w-3.5 md:h-3.5 2xl:w-4 2xl:h-4 ml-1 inline-block opacity-0 group-hover:opacity-100" />}
                                </div>
                            </span>
                        </th>

                        <th scope="col" className={`hidden md:table-cell sticky top-0 z-20 px-3 py-3 md:px-4 md:py-3.5 2xl:px-6 2xl:py-4 text-left text-[10px] md:text-[11px] 2xl:text-xs font-bold uppercase border-b ${
                            isDarkMode ? 'bg-gray-800 text-gray-400 border-gray-700' : 'bg-white text-slate-400 border-slate-100'
                        }`}>
                            Estado
                        </th>

                        {/* Cabecera Acciones (Visible en todos los tamaños para los botones flotantes) */}
                        <th scope="col" className={`sticky top-0 z-20 px-3 py-3 md:px-4 md:py-3.5 2xl:px-6 2xl:py-4 border-b ${
                            isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-slate-100'
                        }`}>
                            <span className="sr-only">Acciones</span>
                        </th>
                    </tr>
                </thead>
                <tbody className={isDarkMode ? 'bg-gray-800' : 'bg-white'}>
                    <tr className="h-2 md:h-4 2xl:h-4"></tr> {/* Fila de separación */}
                    {files.map((fileData, idx) => {
                        
                        const file = typeof fileData === 'string' 
                            ? { name: fileData, date: '--/--/--', size: '---', owner: 'Sistema', type: 'file' }
                            : fileData;
                        
                        const fileName = file.name || `Archivo-${idx}`;
                        const isSelected = selectedFiles.has(fileName);
                        const fileExtension = fileName.split('.').pop()?.toUpperCase() || 'FILE';
                        
                        return (
                            <tr
                                key={fileName + idx}
                                onClick={() => onSelectFile(fileName)}
                                className={`group border-b cursor-pointer ${
                                    isSelected 
                                    ? (isDarkMode ? 'bg-blue-900/30 border-blue-800' : 'bg-blue-50/60 border-slate-50') 
                                    : (isDarkMode ? 'border-gray-700 hover:bg-gray-700/50' : 'border-slate-50 hover:bg-slate-50/80')
                                }`}
                                title={selectedFiles.size === 0 ? "Haz clic para seleccionar y realizar acciones masivas" : ""}
                            >
                                {/* Checkbox */}
                                <td className="px-3 py-3 md:px-4 md:py-3 2xl:px-6 2xl:py-4 w-10 md:w-12 2xl:w-12 align-top md:align-middle pt-4 md:pt-3 2xl:pt-4">
                                    <div className={`${isSelected ? 'scale-100 opacity-100' : 'scale-90 opacity-0 group-hover:opacity-40'}`}>
                                        {isSelected 
                                            ? <CheckSquare className="w-4 h-4 2xl:w-5 2xl:h-5 text-blue-600" /> 
                                            : <Square className={`w-4 h-4 2xl:w-5 2xl:h-5 ${isDarkMode ? 'text-gray-500' : 'text-slate-400'}`} />
                                        }
                                    </div>
                                </td>
                                
                                {/* Celda Principal: Contiene nombre y (solo en móvil) info secundaria */}
                                <td className="px-2 py-3 md:px-4 md:py-3 2xl:px-6 2xl:py-4">
                                    <div className="flex items-start md:items-center">
                                        <div className={`mt-0.5 md:mt-0 shrink-0 h-8 w-8 md:h-9 md:w-9 2xl:h-10 2xl:w-10 flex items-center justify-center rounded-lg md:rounded-xl group-hover:scale-110 transition-transform ${
                                            isDarkMode ? 'bg-gray-700 text-blue-400' : 'bg-blue-50 text-blue-600'
                                        }`}>
                                            <FileIcon type={fileName.split('.').pop()} />
                                        </div>
                                        <div className="ml-3 md:ml-4 w-full">
                                            {/* Nombre de archivo (ajustado truncado para movil) */}
                                            <div className={`text-xs md:text-sm 2xl:text-sm font-semibold truncate max-w-[140px] xs:max-w-[200px] sm:max-w-[250px] md:max-w-[200px] 2xl:max-w-xs ${
                                                isDarkMode ? 'text-gray-200' : 'text-slate-700'
                                            }`} title={fileName}>
                                                {fileName}
                                            </div>
                                            
                                            {/* Subtítulo principal: Tipo de archivo */}
                                            <div className={`text-[10px] md:text-xs 2xl:text-xs font-medium mt-0.5 ${
                                                isDarkMode ? 'text-gray-500' : 'text-slate-400'
                                            }`}>
                                                {fileExtension}
                                            </div>

                                            {/* INFO SECUNDARIA (SOLO EN MÓVIL) - Oculto en md+ */}
                                            <div className="flex flex-col gap-0.5 mt-1 md:hidden">
                                                <span className={`text-[10px] italic ${isDarkMode ? 'text-gray-400' : 'text-slate-500'}`}>
                                                    {file.date || '--/--/----'} • {file.size || '---'}
                                                </span>
                                                <span className={`flex items-center text-[10px] ${isDarkMode ? 'text-green-400' : 'text-green-600'}`}>
                                                    <div className="w-1 h-1 rounded-full bg-green-500 mr-1.5" />
                                                    Indexado
                                                </span>
                                                {context === 'department' && (
                                                    <span className={`text-[10px] mt-0.5 ${isDarkMode ? 'text-gray-400' : 'text-slate-500'}`}>
                                                        Por: {file.owner || 'Sistema'}
                                                    </span>
                                                )}
                                            </div>
                                        </div>
                                    </div>
                                </td>

                                {/* Columnas regulares (Ocultas en móvil) */}
                                {context === 'department' && (
                                    <td className="hidden md:table-cell px-3 py-2.5 md:px-4 md:py-3 2xl:px-6 2xl:py-4 whitespace-nowrap">
                                        <span className={`inline-flex items-center px-2 py-0.5 md:px-2.5 md:py-0.5 rounded-full text-[10px] md:text-[11px] 2xl:text-xs font-medium ${
                                            isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-slate-100 text-slate-600'
                                        }`}>
                                            {file.owner || 'Sistema'}
                                        </span>
                                    </td>
                                )}

                                <td className={`hidden md:table-cell px-3 py-2.5 md:px-4 md:py-3 2xl:px-6 2xl:py-4 whitespace-nowrap text-xs md:text-sm 2xl:text-sm font-medium italic ${
                                    isDarkMode ? 'text-gray-400' : 'text-slate-500'
                                }`}>
                                    {file.date || '--/--/----'}
                                </td>

                                <td className={`hidden md:table-cell px-3 py-2.5 md:px-4 md:py-3 2xl:px-6 2xl:py-4 whitespace-nowrap text-xs md:text-sm 2xl:text-sm font-medium ${
                                    isDarkMode ? 'text-gray-400' : 'text-slate-500'
                                }`}>
                                    {file.size || '---'}
                                </td>

                                <td className="hidden md:table-cell px-3 py-2.5 md:px-4 md:py-3 2xl:px-6 2xl:py-4 whitespace-nowrap font-medium">
                                    <span className={`flex items-center text-[10px] md:text-xs 2xl:text-xs ${isDarkMode ? 'text-green-400' : 'text-green-600'}`}>
                                        <div className="w-1.5 h-1.5 rounded-full bg-green-500 mr-2" />
                                        Indexado
                                    </span>
                                </td>

                                {/* Columna de botones de acción (Visible en todos) */}
                                <td className="px-2 py-3 md:px-4 md:py-3 2xl:px-6 2xl:py-4 whitespace-nowrap text-right font-medium align-top md:align-middle pt-4 md:pt-3 2xl:pt-4">
                                    {!isMultiSelection && (
                                        <div className="flex flex-col md:flex-row items-end md:items-center justify-end gap-1 md:gap-1 opacity-100 md:opacity-0 group-hover:opacity-100 transition-opacity" onClick={(e) => e.stopPropagation()}>
                                            {/* FUNCIONALIDAD COMPARTIR PENDIENTE
                                            {context === 'personal' && (
                                                <button onClick={() => onShare(fileName)} className={`p-1.5 md:p-2 rounded-lg ${
                                                    isDarkMode 
                                                    ? 'text-gray-400 hover:text-indigo-400 hover:bg-indigo-900/30' 
                                                    : 'text-slate-400 hover:text-indigo-600 hover:bg-indigo-50'
                                                }`} title="Compartir con departamento">
                                                    <Share2 className="w-4 h-4 2xl:w-5 2xl:h-5" />
                                                </button>
                                            )} */}
                                            <button onClick={() => onDownload(fileName)} className={`p-1.5 md:p-2 rounded-lg transition-colors ${
                                                isDarkMode 
                                                ? 'bg-blue-900/30 text-blue-400 hover:text-blue-300 md:bg-transparent md:text-gray-400 md:hover:bg-blue-900/30 md:hover:text-blue-400' 
                                                : 'bg-blue-50 text-blue-600 hover:text-blue-700 md:bg-transparent md:text-slate-400 md:hover:bg-blue-50 md:hover:text-blue-600'
                                            }`} title="Descargar">
                                                <Download className="w-4 h-4 2xl:w-5 2xl:h-5" />
                                            </button>
                                            
                                            <button onClick={() => onDelete(fileName)} className={`p-1.5 md:p-2 rounded-lg transition-colors ${
                                                isDarkMode 
                                                ? 'bg-red-900/30 text-red-400 hover:text-red-300 md:bg-transparent md:text-gray-400 md:hover:bg-red-900/30 md:hover:text-red-400' 
                                                : 'bg-red-50 text-red-600 hover:text-red-700 md:bg-transparent md:text-slate-400 md:hover:bg-red-50 md:hover:text-red-600'
                                            }`} title="Eliminar">
                                                <Trash2 className="w-4 h-4 2xl:w-5 2xl:h-5" />
                                            </button>
                                        </div>
                                    )}
                                </td>
                            </tr>
                        );
                    })}
                </tbody>
            </table>
        </div>
    );
};