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

    // === HOOKS PARA ANIMACIÓN Y CIERRE ===
    const [visible, setVisible] = useState(false);
    const modalOverlayRef = useRef(null);

    useEffect(() => {
      if (isOpen) {
        setVisible(true);
      }
    }, [isOpen]);

    const handleClose = () => {
      setVisible(false);
      setTimeout(() => {
          // Limpiamos los estados al cerrar
          setFiles([]);
          setEtiquetas("");
          setDestination(initialDestination || 'personal');
          onClose();
      }, 100); // Esperamos a que termine la animación
    };

    // Escuchar la tecla ESC para cerrar
    useEffect(() => {
      const handleKeyDown = (e) => {
        if (e.key === 'Escape' && isOpen) handleClose();
      };
      window.addEventListener('keydown', handleKeyDown);
      return () => window.removeEventListener('keydown', handleKeyDown);
    }, [isOpen]);

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
        if (isDestinationInvalid) return;

        if (files.length > 0) {
            onUpload(files, destination, etiquetas);
            handleClose(); // Usamos handleClose para que también se anime al subir
        }
    };

    return (
        <div 
            ref={modalOverlayRef}
            className={`fixed inset-0 z-50 flex items-center justify-center p-2 sm:p-4 bg-slate-900/60 backdrop-blur-sm transition-opacity duration-100 ${visible ? 'opacity-100' : 'opacity-0'}`}
            onClick={(e) => {
                if (e.target === modalOverlayRef.current) handleClose();
            }}
        >
            <div 
                className={`rounded-2xl md:rounded-3xl shadow-2xl w-full max-w-2xl border flex flex-col overflow-hidden max-h-[95vh] transform transition-all duration-100 ${
                    visible ? 'translate-y-0 scale-100' : '-translate-y-6 scale-95'
                } ${
                    isDarkMode 
                    ? 'bg-gray-800 border-gray-700' 
                    : 'bg-white border-white/20'
                }`}
                onClick={(e) => e.stopPropagation()}
            >
                
                {/* Header */}
                <div className={`flex justify-between items-center p-4 md:p-6 2xl:p-8 text-white shrink-0 ${
                    isDarkMode 
                    ? 'bg-gray-900 border-b border-gray-700' 
                    : 'bg-gradient-to-r from-blue-600 to-indigo-600'
                }`}>
                    <div>
                        <h2 className="text-xl md:text-2xl font-bold">Carga de Documentos</h2>
                        <p className={`mt-1 text-xs md:text-sm ${isDarkMode ? 'text-gray-400' : 'text-blue-100'}`}>
                            Sube archivos para enriquecer la base de conocimientos
                        </p>
                    </div>
                    {/* Botón de cierre modificado con SVG rojo */}
                    <button 
                        onClick={handleClose} 
                        className="bg-red-600 hover:bg-red-700 text-white rounded-full p-1.5 md:p-2 shadow transition-colors shrink-0 ml-2"
                    >
                        <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 md:h-5 md:w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>

                <div className="p-4 md:p-6 2xl:p-8 space-y-4 md:space-y-6 overflow-y-auto scrollbar-hide">
                    {/* Selector de Destino */}
                    <div className={`p-1 md:p-1.5 rounded-xl md:rounded-2xl flex flex-col sm:flex-row gap-1 sm:gap-0 relative ${
                        isDarkMode ? 'bg-gray-700' : 'bg-slate-50'
                    }`}>
                        <button
                            type="button"
                            onClick={() => setDestination('personal')}
                            className={`flex-1 flex items-center justify-center py-3 md:py-4 px-4 md:px-6 rounded-lg md:rounded-xl text-sm font-semibold transition-all duration-200 ${
                                destination === 'personal'
                                ? (isDarkMode ? 'bg-gray-600 text-white shadow-lg ring-1 ring-gray-600' : 'bg-white text-blue-600 shadow-lg ring-1 ring-slate-100')
                                : (isDarkMode ? 'text-gray-400 hover:text-gray-200' : 'text-slate-500 hover:text-slate-700')
                            }`}
                        >
                            <div className="text-left w-full sm:w-auto flex flex-col items-center sm:items-start">
                                <div className="flex items-center gap-2">
                                    <Lock className={`w-4 h-4 ${
                                        destination === 'personal' 
                                            ? (isDarkMode ? 'text-blue-400' : 'text-blue-500') 
                                            : 'text-slate-400'
                                    }`} />
                                    <span>Directorio Personal</span>
                                </div>
                                <span className={`block text-[10px] md:text-xs font-normal mt-0.5 sm:mt-1 sm:ml-6 text-center sm:text-left ${
                                    destination === 'personal' 
                                        ? (isDarkMode ? 'text-blue-300' : 'text-blue-400') 
                                        : 'text-slate-400'
                                }`}>
                                    Solo tú tendrás acceso
                                </span>
                            </div>
                        </button>
                        
                        <button
                            type="button"
                            onClick={() => setDestination('department')}
                            className={`flex-1 flex items-center justify-center py-3 md:py-4 px-4 md:px-6 rounded-lg md:rounded-xl text-sm font-semibold transition-all duration-200 ${
                                destination === 'department'
                                ? (isDarkMode ? 'bg-gray-600 text-white shadow-lg ring-1 ring-gray-600' : 'bg-white text-blue-600 shadow-lg ring-1 ring-slate-100')
                                : (isDarkMode ? 'text-gray-400 hover:text-gray-200' : 'text-slate-500 hover:text-slate-700')
                            }`}
                        >
                            <div className="text-left w-full sm:w-auto flex flex-col items-center sm:items-start">
                                <div className="flex items-center gap-2">
                                    <Globe className={`w-4 h-4 ${
                                        destination === 'department' 
                                            ? (isDarkMode ? 'text-indigo-400' : 'text-indigo-500') 
                                            : 'text-slate-400'
                                    }`} />
                                    <span>Departamento</span>
                                </div>
                                <span className={`block text-[10px] md:text-xs font-normal mt-0.5 sm:mt-1 sm:ml-6 text-center sm:text-left ${
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
                        <div className={`border rounded-xl p-3 flex items-start gap-2 md:gap-3 animate-in fade-in slide-in-from-top-2 ${
                            isDarkMode ? 'bg-red-900/20 border-red-800' : 'bg-red-50 border-red-100'
                        }`}>
                            <AlertCircle className={`w-4 h-4 md:w-5 md:h-5 mt-0.5 flex-shrink-0 ${isDarkMode ? 'text-red-400' : 'text-red-500'}`} />
                            <div>
                                <h4 className={`text-xs md:text-sm font-bold ${isDarkMode ? 'text-red-300' : 'text-red-700'}`}>Acceso restringido</h4>
                                <p className={`text-[10px] md:text-xs mt-0.5 md:mt-1 ${isDarkMode ? 'text-red-400' : 'text-red-600'}`}>
                                    No tienes asignado ningún departamento para compartir archivos. Por favor, selecciona "Directorio Personal" o contacta con administración.
                                </p>
                            </div>
                        </div>
                    )}

                    {/* Input de Etiquetas */}
                    <div className={`space-y-1.5 md:space-y-2 transition-opacity ${isDestinationInvalid ? 'opacity-50 pointer-events-none' : ''}`}>
                        <label className={`text-[10px] md:text-[11px] font-bold uppercase tracking-wider flex items-center gap-1.5 md:gap-2 ml-1 ${isDarkMode ? 'text-gray-400' : 'text-slate-400'}`}>
                            <Tag className="w-3 h-3 md:w-3.5 md:h-3.5" /> Etiquetas de indexación (Opcional)
                        </label>
                        <input 
                            type="text" 
                            placeholder="ej: cliente_alpha, legal, urgente..."
                            className={`w-full px-3 py-2.5 md:px-4 md:py-3 text-sm border-none rounded-xl md:rounded-2xl focus:ring-2 focus:ring-blue-500/20 transition-all shadow-inner ${
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
                        className={`relative rounded-2xl md:rounded-3xl border-2 border-dashed transition-all duration-300 group ${
                            dragActive
                            ? 'border-blue-500 bg-blue-50/50 scale-[1.01]'
                            : (isDarkMode ? 'border-gray-600 hover:border-gray-500 hover:bg-gray-700/50' : 'border-slate-200 hover:border-blue-300 hover:bg-slate-50/50')
                        } ${files.length > 0 ? 'p-4 md:p-6' : 'p-6 md:p-10'} ${isDestinationInvalid ? 'opacity-50 pointer-events-none' : ''}`}
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
                            files.length > 0 ? 'w-10 h-10 md:w-12 md:h-12 mb-2 md:mb-3' : 'w-16 h-16 md:w-20 md:h-20 mb-4 md:mb-6'
                        } ${isDarkMode ? 'bg-gray-700' : 'bg-blue-50'}`}>
                            <Upload className={`${files.length > 0 ? 'h-5 w-5 md:h-6 md:w-6' : 'h-8 w-8 md:h-10 md:w-10'} ${isDarkMode ? 'text-blue-400' : 'text-blue-600'}`} />
                        </div>

                        <div className="text-center">
                            <h3 className={`font-bold ${isDarkMode ? 'text-white' : 'text-slate-800'} ${files.length > 0 ? 'text-base md:text-lg mb-1' : 'text-lg md:text-xl mb-1 md:mb-2'}`}>
                                Selecciona o arrastra tus archivos
                            </h3>
                            <p className={`mx-auto max-w-sm ${isDarkMode ? 'text-gray-400' : 'text-slate-500'} ${files.length > 0 ? 'text-[10px] md:text-xs mb-2 md:mb-3' : 'text-xs md:text-sm mb-4 md:mb-6'}`}>
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
                                } ${files.length > 0 ? 'px-4 py-1.5 md:px-6 md:py-2 text-xs md:text-sm' : 'px-6 py-2 md:px-8 md:py-3 text-sm'}`}
                                disabled={isDestinationInvalid}
                            >
                                Explorar Archivos
                            </button>
                        </div>
                    </div>

                    {/* Lista de archivos a subir */}
                    {files.length > 0 && (
                        <div className={`rounded-xl md:rounded-2xl p-3 md:p-4 max-h-48 md:max-h-72 overflow-y-auto scrollbar-hide border ${
                            isDarkMode ? 'bg-gray-700/50 border-gray-600' : 'bg-slate-50 border-slate-100'
                        }`}>
                            <div className="flex justify-between items-center mb-2 md:mb-3 px-1 md:px-2">
                                <h4 className={`text-[10px] md:text-[11px] font-bold uppercase tracking-wider ${isDarkMode ? 'text-gray-300' : 'text-slate-700'}`}>Archivos listos para subir</h4>
                                <span className={`text-[10px] md:text-xs font-semibold px-2 py-0.5 md:px-2.5 md:py-1 rounded-lg ${isDarkMode ? 'bg-blue-900/50 text-blue-300' : 'bg-blue-100/80 text-blue-700'}`}>{files.length} archivos</span>
                            </div>
                            <div className="space-y-1.5 md:space-y-2">
                                {files.map((file, idx) => (
                                    <div key={idx} className={`flex items-center justify-between p-2 md:p-3 rounded-lg md:rounded-xl border shadow-sm group transition-colors ${
                                        isDarkMode 
                                        ? 'bg-gray-800 border-gray-600 hover:border-gray-500' 
                                        : 'bg-white border-slate-100 hover:border-blue-200'
                                    }`}>
                                        <div className="flex items-center truncate">
                                            <div className={`p-1.5 md:p-2 rounded-md md:rounded-lg mr-2 md:mr-3 shrink-0 ${isDarkMode ? 'bg-gray-700' : 'bg-blue-50'}`}>
                                                <File className={`w-4 h-4 md:w-5 md:h-5 ${isDarkMode ? 'text-blue-400' : 'text-blue-600'}`} />
                                            </div>
                                            <div className="flex flex-col truncate">
                                                <span className={`text-xs md:text-sm font-semibold truncate max-w-[150px] sm:max-w-[200px] ${isDarkMode ? 'text-gray-200' : 'text-slate-700'}`}>{file.name}</span>
                                                <span className={`text-[10px] md:text-xs ${isDarkMode ? 'text-gray-400' : 'text-slate-400'}`}>{(file.size / 1024).toFixed(1)} KB</span>
                                            </div>
                                        </div>
                                        <button onClick={(e) => { e.stopPropagation(); removeFile(idx); }} className={`p-1.5 md:p-2 rounded-lg transition-all shrink-0 ${
                                            isDarkMode ? 'text-gray-400 hover:text-red-400 hover:bg-red-900/20' : 'text-slate-300 hover:text-red-500 hover:bg-red-50'
                                        }`}>
                                            <X className="w-4 h-4 md:w-5 md:h-5" />
                                        </button>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div>

                {/* Footer */}
                <div className={`p-4 md:p-6 border-t flex flex-col sm:flex-row justify-end gap-3 md:gap-4 shrink-0 ${
                    isDarkMode ? 'bg-gray-900 border-gray-700' : 'bg-slate-50 border-slate-100'
                }`}>
                    <button
                        onClick={handleClose}
                        className={`px-4 md:px-6 py-2.5 md:py-3 text-sm font-bold rounded-xl transition-colors w-full sm:w-auto ${
                            isDarkMode 
                            ? 'text-gray-400 hover:text-white hover:bg-gray-800 border border-gray-700 sm:border-transparent' 
                            : 'text-slate-600 bg-white sm:bg-transparent border border-slate-200 sm:border-transparent hover:text-slate-800 hover:bg-slate-200/50'
                        }`}
                    >
                        Cancelar
                    </button>
                    <button
                        onClick={handleSubmit}
                        disabled={files.length === 0 || isDestinationInvalid}
                        className={`px-6 md:px-8 py-2.5 md:py-3 text-sm font-bold text-white rounded-xl shadow-lg transition-all duration-300 flex items-center justify-center gap-2 w-full sm:w-auto ${
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