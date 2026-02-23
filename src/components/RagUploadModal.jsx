import React, { useState, useRef, useEffect } from 'react';
import { Upload, X, File, Tag, Lock, Globe, AlertCircle } from 'lucide-react';

export const RagUploadModal = ({ 
    isOpen, 
    onClose, 
    onUpload, 
    initialDestination,
    availableDepartments = [],
    isDarkMode
}) => {
    // Verificamos si el usuario tiene departamentos
    const hasDepartments = availableDepartments && availableDepartments.length > 0;

    const [dragActive, setDragActive] = useState(false);
    const [files, setFiles] = useState([]);
    
    // Inicializamos destino (por defecto personal)
    const [destination, setDestination] = useState(initialDestination || 'personal');
    
    const [etiquetas, setEtiquetas] = useState("");
    const inputRef = useRef(null);

    // Calculamos si el destino actual es inválido
    const isDestinationInvalid = destination === 'department' && !hasDepartments;

    if (!isOpen) return null;

    const handleDrag = (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (e.type === "dragenter" || e.type === "dragover") {
            setDragActive(true);
        } else if (e.type === "dragleave") {
            setDragActive(false);
        }
    };

    const handleDrop = (e) => {
        e.preventDefault();
        e.stopPropagation();
        setDragActive(false);
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            const newFiles = Array.from(e.dataTransfer.files);
            setFiles(prev => [...prev, ...newFiles]);
        }
    };

    const handleChange = (e) => {
        e.preventDefault();
        if (e.target.files && e.target.files[0]) {
            const newFiles = Array.from(e.target.files);
            setFiles(prev => [...prev, ...newFiles]);
        }
    };

    const removeFile = (idx) => {
        setFiles(prev => prev.filter((_, i) => i !== idx));
    };

    const handleSubmit = () => {
        // Bloqueo adicional por seguridad
        if (isDestinationInvalid) return;

        if (files.length > 0) {
            onUpload(files, destination, etiquetas);
            setFiles([]);
            setEtiquetas("");
            onClose();
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-900/60 backdrop-blur-sm transition-opacity">
            <div className={`rounded-3xl shadow-2xl w-full max-w-2xl border transform transition-all overflow-hidden ${
                isDarkMode 
                ? 'bg-gray-800 border-gray-700' 
                : 'bg-white border-white/20'
            }`}>
                
                {/* Header: Azul en modo claro, Gris oscuro en modo oscuro */}
                <div className={`flex justify-between items-center p-8 text-white ${
                    isDarkMode 
                    ? 'bg-gray-900 border-b border-gray-700' 
                    : 'bg-gradient-to-r from-blue-600 to-indigo-600'
                }`}>
                    <div>
                        <h2 className="text-2xl font-bold">Carga de Documentos</h2>
                        <p className={`mt-1 text-sm ${isDarkMode ? 'text-gray-400' : 'text-blue-100'}`}>
                            Sube archivos para enriquecer la base de conocimientos
                        </p>
                    </div>
                    <button onClick={onClose} className={`p-2 rounded-full transition-all ${
                        isDarkMode 
                        ? 'text-gray-400 hover:text-white hover:bg-gray-700' 
                        : 'text-white/70 hover:text-white bg-white/10 hover:bg-white/20'
                    }`}>
                        <X className="w-6 h-6" />
                    </button>
                </div>

                <div className="p-8 space-y-6">
                    {/* Selector de Destino */}
                    <div className={`p-1.5 rounded-2xl flex relative ${
                        isDarkMode ? 'bg-gray-700' : 'bg-slate-50'
                    }`}>
                        <button
                            type="button"
                            onClick={() => setDestination('personal')}
                            className={`flex-1 flex items-center justify-center py-4 px-6 rounded-xl text-sm font-semibold transition-all duration-200 ${
                                destination === 'personal'
                                ? (isDarkMode ? 'bg-gray-600 text-white shadow-lg ring-1 ring-gray-600' : 'bg-white text-blue-600 shadow-lg ring-1 ring-slate-100')
                                : (isDarkMode ? 'text-gray-400 hover:text-gray-200' : 'text-slate-500 hover:text-slate-700')
                            }`}
                        >
                            <div className="text-left">
                                <div className="flex items-center gap-2">
                                    <Lock className={`w-4 h-4 ${
                                        destination === 'personal' 
                                            ? (isDarkMode ? 'text-blue-400' : 'text-blue-500') 
                                            : 'text-slate-400'
                                    }`} />
                                    <span>Directorio Personal</span>
                                </div>
                                <span className={`block text-xs font-normal mt-1 ml-6 ${
                                    destination === 'personal' 
                                        ? (isDarkMode ? 'text-blue-300' : 'text-blue-400') 
                                        : 'text-slate-400'
                                }`}>
                                    Solo tú tendrás acceso
                                </span>
                            </div>
                        </button>
                        
                        {/* Botón Departamento */}
                        <button
                            type="button"
                            onClick={() => setDestination('department')}
                            className={`flex-1 flex items-center justify-center py-4 px-6 rounded-xl text-sm font-semibold transition-all duration-200 ${
                                destination === 'department'
                                ? (isDarkMode ? 'bg-gray-600 text-white shadow-lg ring-1 ring-gray-600' : 'bg-white text-blue-600 shadow-lg ring-1 ring-slate-100')
                                : (isDarkMode ? 'text-gray-400 hover:text-gray-200' : 'text-slate-500 hover:text-slate-700')
                            }`}
                        >
                            <div className="text-left">
                                <div className="flex items-center gap-2">
                                    <Globe className={`w-4 h-4 ${
                                        destination === 'department' 
                                            ? (isDarkMode ? 'text-indigo-400' : 'text-indigo-500') 
                                            : 'text-slate-400'
                                    }`} />
                                    <span>Departamento</span>
                                </div>
                                <span className={`block text-xs font-normal mt-1 ml-6 ${
                                    destination === 'department' 
                                        ? (isDarkMode ? 'text-blue-300' : 'text-blue-400') 
                                        : 'text-slate-400'
                                }`}>
                                    Compartido con tu equipo
                                </span>
                            </div>
                        </button>
                    </div>

                    {/* MENSAJE DE ERROR */}
                    {isDestinationInvalid && (
                        <div className={`border rounded-xl p-3 flex items-start gap-3 animate-in fade-in slide-in-from-top-2 ${
                            isDarkMode ? 'bg-red-900/20 border-red-800' : 'bg-red-50 border-red-100'
                        }`}>
                            <AlertCircle className={`w-5 h-5 mt-0.5 flex-shrink-0 ${isDarkMode ? 'text-red-400' : 'text-red-500'}`} />
                            <div>
                                <h4 className={`text-sm font-bold ${isDarkMode ? 'text-red-300' : 'text-red-700'}`}>Acceso restringido</h4>
                                <p className={`text-xs mt-0.5 ${isDarkMode ? 'text-red-400' : 'text-red-600'}`}>
                                    No tienes asignado ningún departamento para compartir archivos. Por favor, selecciona "Directorio Personal" o contacta con administración.
                                </p>
                            </div>
                        </div>
                    )}

                    {/* Input de Etiquetas */}
                    <div className={`space-y-2 transition-opacity ${isDestinationInvalid ? 'opacity-50 pointer-events-none' : ''}`}>
                        <label className={`text-[11px] font-bold uppercase tracking-wider flex items-center gap-2 ml-1 ${isDarkMode ? 'text-gray-400' : 'text-slate-400'}`}>
                            <Tag className="w-3.5 h-3.5" /> Etiquetas de indexación (Opcional)
                        </label>
                        <input 
                            type="text" 
                            placeholder="ej: cliente_alpha, legal, urgente..."
                            className={`w-full px-4 py-3 border-none rounded-2xl focus:ring-2 focus:ring-blue-500/20 transition-all shadow-inner ${
                                isDarkMode 
                                ? 'bg-gray-700 text-white placeholder:text-gray-500' 
                                : 'bg-slate-50 text-slate-700 placeholder:text-slate-400'
                            }`}
                            value={etiquetas}
                            onChange={(e) => setEtiquetas(e.target.value)}
                            disabled={isDestinationInvalid}
                        />
                    </div>

                    {/* Dropzone */}
                    <div
                        className={`relative rounded-3xl border-2 border-dashed transition-all duration-300 group ${
                            dragActive
                            ? 'border-blue-500 bg-blue-50/50 scale-[1.01]'
                            : (isDarkMode ? 'border-gray-600 hover:border-gray-500 hover:bg-gray-700/50' : 'border-slate-200 hover:border-blue-300 hover:bg-slate-50/50')
                        } ${files.length > 0 ? 'p-6' : 'p-10'} ${isDestinationInvalid ? 'opacity-50 pointer-events-none' : ''}`}
                        onDragEnter={!isDestinationInvalid ? handleDrag : undefined}
                        onDragLeave={!isDestinationInvalid ? handleDrag : undefined}
                        onDragOver={!isDestinationInvalid ? handleDrag : undefined}
                        onDrop={!isDestinationInvalid ? handleDrop : undefined}
                        onClick={() => !isDestinationInvalid && inputRef.current?.click()}
                    >
                        <input
                            ref={inputRef}
                            type="file"
                            multiple
                            className="hidden"
                            onChange={handleChange}
                            accept=".pdf,.doc,.docx,.txt,.md,.xlsx,.pptx,image/*"
                            disabled={isDestinationInvalid}
                        />
                        <div className={`rounded-full flex items-center justify-center mx-auto group-hover:scale-110 transition-transform duration-300 ${
                            files.length > 0 ? 'w-12 h-12 mb-3' : 'w-20 h-20 mb-6'
                        } ${isDarkMode ? 'bg-gray-700' : 'bg-blue-50'}`}>
                            <Upload className={`${files.length > 0 ? 'h-6 w-6' : 'h-10 w-10'} ${isDarkMode ? 'text-blue-400' : 'text-blue-600'}`} />
                        </div>

                        <div className="text-center">
                            <h3 className={`font-bold ${isDarkMode ? 'text-white' : 'text-slate-800'} ${files.length > 0 ? 'text-lg mb-1' : 'text-xl mb-2'}`}>
                                Selecciona o arrastra tus archivos
                            </h3>
                            <p className={`mx-auto max-w-sm ${isDarkMode ? 'text-gray-400' : 'text-slate-500'} ${files.length > 0 ? 'text-xs mb-3' : 'text-sm mb-6'}`}>
                                Soporta PDF, DOCX, TXT, Excel, PPT e Imágenes.
                            </p>
                            
                            <button
                                type="button"
                                onClick={(e) => {
                                    e.stopPropagation();
                                    if (!isDestinationInvalid) inputRef.current?.click();
                                }}
                                className={`rounded-xl font-semibold shadow-lg transition-all hover:-translate-y-0.5 active:translate-y-0 ${
                                    isDarkMode 
                                    ? 'bg-blue-600 hover:bg-blue-500 text-white shadow-blue-900/30' 
                                    : 'bg-blue-600 hover:bg-blue-700 text-white shadow-blue-500/30'
                                } ${files.length > 0 ? 'px-6 py-2 text-sm' : 'px-8 py-3'}`}
                                disabled={isDestinationInvalid}
                            >
                                Explorar Archivos
                            </button>
                        </div>
                    </div>

                    {/* Lista de archivos a subir */}
                    {files.length > 0 && (
                        <div className={`rounded-2xl p-4 max-h-72 overflow-y-auto custom-scrollbar border ${
                            isDarkMode ? 'bg-gray-700/50 border-gray-600' : 'bg-slate-50 border-slate-100'
                        }`}>
                            <div className="flex justify-between items-center mb-3 px-2">
                                <h4 className={`text-[11px] font-bold uppercase tracking-wider ${isDarkMode ? 'text-gray-300' : 'text-slate-700'}`}>Archivos listos para subir</h4>
                                <span className={`text-xs font-semibold px-2.5 py-1 rounded-lg ${isDarkMode ? 'bg-blue-900/50 text-blue-300' : 'bg-blue-100/80 text-blue-700'}`}>{files.length} archivos</span>
                            </div>
                            <div className="space-y-2">
                                {files.map((file, idx) => (
                                    <div key={idx} className={`flex items-center justify-between p-3 rounded-xl border shadow-sm group transition-colors ${
                                        isDarkMode 
                                        ? 'bg-gray-800 border-gray-600 hover:border-gray-500' 
                                        : 'bg-white border-slate-100 hover:border-blue-200'
                                    }`}>
                                        <div className="flex items-center truncate">
                                            <div className={`p-2 rounded-lg mr-3 ${isDarkMode ? 'bg-gray-700' : 'bg-blue-50'}`}>
                                                <File className={`w-5 h-5 ${isDarkMode ? 'text-blue-400' : 'text-blue-600'}`} />
                                            </div>
                                            <div className="flex flex-col truncate">
                                                <span className={`text-sm font-semibold truncate max-w-[200px] ${isDarkMode ? 'text-gray-200' : 'text-slate-700'}`}>{file.name}</span>
                                                <span className={`text-xs ${isDarkMode ? 'text-gray-400' : 'text-slate-400'}`}>{(file.size / 1024).toFixed(1)} KB</span>
                                            </div>
                                        </div>
                                        <button onClick={(e) => { e.stopPropagation(); removeFile(idx); }} className={`p-2 rounded-lg transition-all ${
                                            isDarkMode ? 'text-gray-400 hover:text-red-400 hover:bg-red-900/20' : 'text-slate-300 hover:text-red-500 hover:bg-red-50'
                                        }`}>
                                            <X className="w-5 h-5" />
                                        </button>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className={`p-6 border-t flex justify-end gap-4 ${
                    isDarkMode ? 'bg-gray-900 border-gray-700' : 'bg-slate-50 border-slate-100'
                }`}>
                    <button
                        onClick={onClose}
                        className={`px-6 py-3 text-sm font-bold rounded-xl transition-colors ${
                            isDarkMode 
                            ? 'text-gray-400 hover:text-white hover:bg-gray-800' 
                            : 'text-slate-500 hover:text-slate-700 hover:bg-slate-200/50'
                        }`}
                    >
                        Cancelar
                    </button>
                    <button
                        onClick={handleSubmit}
                        disabled={files.length === 0 || isDestinationInvalid}
                        className={`px-8 py-3 text-sm font-bold text-white rounded-xl shadow-lg transition-all duration-300 flex items-center gap-2 ${
                            files.length === 0 || isDestinationInvalid
                            ? (isDarkMode ? 'bg-gray-700 text-gray-500 cursor-not-allowed shadow-none' : 'bg-slate-300 cursor-not-allowed shadow-none')
                            : 'bg-gradient-to-r from-blue-600 to-indigo-600 hover:translate-y-[-2px] hover:shadow-blue-500/40'
                        }`}
                    >
                        <Upload className="w-4 h-4" />
                        Subir e Indexar
                    </button>
                </div>
            </div>
        </div>
    );
};