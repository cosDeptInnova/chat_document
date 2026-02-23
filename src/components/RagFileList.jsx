import React from 'react';
import { Trash2, Download, FileText, Image, File, Share2, Square, CheckSquare, ChevronUp, ChevronDown } from 'lucide-react';

const FileIcon = ({ type }) => {
    const t = type?.toLowerCase() || '';
    if (t.includes('pdf')) return <FileText className="w-5 h-5 text-red-500" />;
    if (t.includes('word') || t.includes('doc')) return <FileText className="w-5 h-5 text-blue-500" />;
    if (t.includes('image') || t.includes('png') || t.includes('jpg')) return <Image className="w-5 h-5 text-purple-500" />;
    return <File className="w-5 h-5 text-gray-400" />;
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
            <div className="flex flex-col items-center justify-center py-20 text-slate-400">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mb-4"></div>
                <p className="font-medium">Consultando base de conocimientos...</p>
            </div>
        );
    }

    // 2. Estado vacío (sin archivos)
    if (!files || files.length === 0) {
        return (
            <div className="flex flex-col items-center justify-center py-20 text-slate-400">
                <File className="w-12 h-12 mb-4 opacity-20" />
                <p className="font-medium">No se encontraron documentos en este destino</p>
            </div>
        );
    }

    const isMultiSelection = selectedFiles.size > 1;

    const SortIndicator = ({ columnKey }) => {
        if (sortConfig.key !== columnKey) return null;
        return sortConfig.direction === 'asc' ? 
            <ChevronUp className="w-4 h-4 ml-1 inline-block" /> : 
            <ChevronDown className="w-4 h-4 ml-1 inline-block" />;
    };

    return (
        <div className={`w-full ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
            <table className="min-w-full border-separate border-spacing-0">
                <thead>
                    <tr>
                        <th scope="col" className={`sticky top-0 z-20 px-6 py-4 border-b w-12 ${
                            isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-slate-100'
                        }`}>
                            {/* Checkbox column */}
                        </th>
                        
                        <th scope="col" className={`sticky top-0 z-20 px-6 py-4 text-left text-xs font-bold uppercase cursor-pointer transition-colors whitespace-nowrap group ${
                            isDarkMode 
                            ? 'bg-gray-800 text-gray-400 hover:text-blue-400 border-gray-700' 
                            : 'bg-white text-slate-400 hover:text-blue-600 border-slate-100'
                        }`} onClick={() => onSort('name')}>
                            <span className="flex items-center">
                                Nombre del Archivo
                                <div className={`transition-opacity duration-200 ${sortConfig.key === 'name' ? 'opacity-100' : 'opacity-0 group-hover:opacity-40'}`}>
                                    <SortIndicator columnKey="name" />
                                    {sortConfig.key !== 'name' && <ChevronUp className="w-4 h-4 ml-1 inline-block opacity-0 group-hover:opacity-100" />}
                                </div>
                            </span>
                        </th>

                        {context === 'department' && (
                            <th scope="col" className={`sticky top-0 z-20 px-6 py-4 text-left text-xs font-bold uppercase border-b ${
                                isDarkMode ? 'bg-gray-800 text-gray-400 border-gray-700' : 'bg-white text-slate-400 border-slate-100'
                            }`}>
                                Subido por
                            </th>
                        )}

                        <th scope="col" className={`sticky top-0 z-20 px-6 py-4 text-left text-xs font-bold uppercase cursor-pointer transition-colors group ${
                            isDarkMode 
                            ? 'bg-gray-800 text-gray-400 hover:text-blue-400 border-gray-700' 
                            : 'bg-white text-slate-400 hover:text-blue-600 border-slate-100'
                        }`} onClick={() => onSort('date')}>
                            <span className="flex items-center">
                                Fecha
                                <div className={`transition-opacity duration-200 ${sortConfig.key === 'date' ? 'opacity-100' : 'opacity-0 group-hover:opacity-40'}`}>
                                    <SortIndicator columnKey="date" />
                                    {sortConfig.key !== 'date' && <ChevronUp className="w-4 h-4 ml-1 inline-block opacity-0 group-hover:opacity-100" />}
                                </div>
                            </span>
                        </th>

                        <th scope="col" className={`sticky top-0 z-20 px-6 py-4 text-left text-xs font-bold uppercase cursor-pointer transition-colors group ${
                            isDarkMode 
                            ? 'bg-gray-800 text-gray-400 hover:text-blue-400 border-gray-700' 
                            : 'bg-white text-slate-400 hover:text-blue-600 border-slate-100'
                        }`} onClick={() => onSort('size')}>
                            <span className="flex items-center">
                                Tamaño
                                <div className={`transition-opacity duration-200 ${sortConfig.key === 'size' ? 'opacity-100' : 'opacity-0 group-hover:opacity-40'}`}>
                                    <SortIndicator columnKey="size" />
                                    {sortConfig.key !== 'size' && <ChevronUp className="w-4 h-4 ml-1 inline-block opacity-0 group-hover:opacity-100" />}
                                </div>
                            </span>
                        </th>

                        <th scope="col" className={`sticky top-0 z-20 px-6 py-4 text-left text-xs font-bold uppercase border-b ${
                            isDarkMode ? 'bg-gray-800 text-gray-400 border-gray-700' : 'bg-white text-slate-400 border-slate-100'
                        }`}>
                            Estado
                        </th>

                        <th scope="col" className={`sticky top-0 z-20 px-6 py-4 border-b ${
                            isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-slate-100'
                        }`}>
                            <span className="sr-only">Acciones</span>
                        </th>
                    </tr>
                </thead>
                <tbody className={isDarkMode ? 'bg-gray-800' : 'bg-white'}>
                    <tr className="h-4"></tr> {/* Fila de separación */}
                    {files.map((fileData, idx) => {
                        
                        // 1. Aseguramos que 'file' sea un objeto
                        const file = typeof fileData === 'string' 
                            ? { name: fileData, date: '--/--/--', size: '---', owner: 'Sistema', type: 'file' }
                            : fileData;
                        
                        // 2. DEFINIMOS 'fileName' explícitamente para evitar ReferenceError
                        const fileName = file.name || `Archivo-${idx}`;

                        // 3. Calculamos datos derivados
                        const isSelected = selectedFiles.has(fileName);
                        const fileExtension = fileName.split('.').pop()?.toUpperCase() || 'FILE';
                        
                        return (
                            <tr
                                key={fileName + idx} // Key única combinada por seguridad
                                onClick={() => onSelectFile(fileName)}
                                className={`group transition-all duration-200 border-b cursor-pointer relative ${
                                    isSelected 
                                    ? (isDarkMode ? 'bg-blue-900/30 border-blue-800' : 'bg-blue-50/60 border-slate-50') 
                                    : (isDarkMode ? 'border-gray-700 hover:bg-gray-700/50' : 'border-slate-50 hover:bg-slate-50/80')
                                }`}
                                title={selectedFiles.size === 0 ? "Haz clic para seleccionar y realizar acciones masivas" : ""}
                            >
                                <td className="px-6 py-4 w-12">
                                    <div className={`transition-all duration-200 ${isSelected ? 'scale-100 opacity-100' : 'scale-90 opacity-0 group-hover:opacity-40'}`}>
                                        {isSelected 
                                            ? <CheckSquare className="w-5 h-5 text-blue-600" /> 
                                            : <Square className={`w-5 h-5 ${isDarkMode ? 'text-gray-500' : 'text-slate-400'}`} />
                                        }
                                    </div>
                                </td>
                                
                                <td className="px-6 py-4">
                                    <div className="flex items-center">
                                        <div className={`shrink-0 h-10 w-10 flex items-center justify-center rounded-xl transition-transform group-hover:scale-110 ${
                                            isDarkMode ? 'bg-gray-700 text-blue-400' : 'bg-blue-50 text-blue-600'
                                        }`}>
                                            <FileIcon type={fileName.split('.').pop()} />
                                        </div>
                                        <div className="ml-4">
                                            <div className={`text-sm font-semibold truncate max-w-xs ${
                                                isDarkMode ? 'text-gray-200' : 'text-slate-700'
                                            }`} title={fileName}>
                                                {fileName}
                                            </div>
                                            <div className={`text-xs font-medium ${
                                                isDarkMode ? 'text-gray-500' : 'text-slate-400'
                                            }`}>{fileExtension}</div>
                                        </div>
                                    </div>
                                </td>

                                {context === 'department' && (
                                    <td className="px-6 py-4 whitespace-nowrap">
                                        <span className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                                            isDarkMode ? 'bg-gray-700 text-gray-300' : 'bg-slate-100 text-slate-600'
                                        }`}>
                                            {file.owner || 'Sistema'}
                                        </span>
                                    </td>
                                )}

                                <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium italic ${
                                    isDarkMode ? 'text-gray-400' : 'text-slate-500'
                                }`}>
                                    {file.date || '--/--/----'}
                                </td>

                                <td className={`px-6 py-4 whitespace-nowrap text-sm font-medium ${
                                    isDarkMode ? 'text-gray-400' : 'text-slate-500'
                                }`}>
                                    {file.size || '---'}
                                </td>

                                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                                    <span className={`flex items-center text-xs ${isDarkMode ? 'text-green-400' : 'text-green-600'}`}>
                                        <div className="w-1.5 h-1.5 rounded-full bg-green-500 mr-2" />
                                        Indexado
                                    </span>
                                </td>

                                <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                                    {!isMultiSelection && (
                                        <div className="flex items-center justify-end space-x-1 opacity-0 group-hover:opacity-100 transition-opacity" onClick={(e) => e.stopPropagation()}>
                                            {/* FUNCIONALIDAD COMPARTIR PENDIENTE
                                            {context === 'personal' && (
                                                <button onClick={() => onShare(fileName)} className={`p-2 rounded-lg ${
                                                    isDarkMode 
                                                    ? 'text-gray-400 hover:text-indigo-400 hover:bg-indigo-900/30' 
                                                    : 'text-slate-400 hover:text-indigo-600 hover:bg-indigo-50'
                                                }`} title="Compartir con departamento">
                                                    <Share2 className="w-5 h-5" />
                                                </button>
                                            )} */}
                                            
                                            <button onClick={() => onDownload(fileName)} className={`p-2 rounded-lg transition-all ${
                                                isDarkMode 
                                                ? 'text-gray-400 hover:text-blue-400 hover:bg-blue-900/30' 
                                                : 'text-slate-400 hover:text-blue-600 hover:bg-blue-50'
                                            }`} title="Descargar">
                                                <Download className="w-5 h-5" />
                                            </button>
                                            
                                            <button onClick={() => onDelete(fileName)} className={`p-2 rounded-lg transition-all ${
                                                isDarkMode 
                                                ? 'text-gray-400 hover:text-red-400 hover:bg-red-900/30' 
                                                : 'text-slate-400 hover:text-red-600 hover:bg-red-50'
                                            }`} title="Eliminar">
                                                <Trash2 className="w-5 h-5" />
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