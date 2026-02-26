import React, { useState, useEffect, useRef, useMemo } from 'react';

export default function PdfViewerModal ({ isOpen, onClose, fileUrl, fileName, page, source }) {
  const [visible, setVisible] = useState(false);
  const modalOverlayRef = useRef(null);

  const fragments = useMemo(() => {
    if (Array.isArray(source?.fragments) && source.fragments.length > 0) {
      return source.fragments;
    }
    if (source?.snippet || source?.page || source?.fragment) {
      return [{
        page: source?.page || page || null,
        fragment: source?.fragment || null,
        snippet: source?.snippet || '',
      }];
    }
    return [];
  }, [source, page]);

  const [activePage, setActivePage] = useState(page || source?.page || null);

  useEffect(() => {
    if (isOpen) {
      setVisible(true);
      setActivePage(page || source?.page || fragments?.[0]?.page || null);
    }
  }, [isOpen, page, source, fragments]);

  const handleClose = () => {
    setVisible(false);
    setTimeout(() => onClose(), 300);
  };

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && isOpen) handleClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen]);

  if (!isOpen) return null;

  const ext = fileName?.split('.').pop().toLowerCase() || '';
  const isPdf = ext === 'pdf';
  const isImage = ['png', 'jpg', 'jpeg', 'gif', 'webp'].includes(ext);
  const isText = ['txt'].includes(ext);
  const isPreviewable = isPdf || isImage || isText;

  const viewerUrl = isPdf && activePage ? `${fileUrl}#page=${activePage}` : fileUrl;

  return (
    <div
      ref={modalOverlayRef}
      className={`fixed inset-0 z-[9999] flex items-center justify-center bg-black/70 backdrop-blur-sm p-2 sm:p-3 2xl:p-4 transition-opacity duration-300 ${visible ? 'opacity-100' : 'opacity-0'}`}
      onClick={(e) => {
        if (e.target === modalOverlayRef.current) handleClose();
      }}
    >
      <div
        className={`bg-white dark:bg-gray-800 w-full max-w-4xl 2xl:max-w-6xl h-full max-h-[95vh] md:max-h-[90vh] 2xl:h-[90vh] 2xl:max-h-none rounded-lg shadow-2xl flex flex-col overflow-hidden border border-gray-200 dark:border-gray-700 transform transition-all duration-300 ${visible ? 'translate-y-0 scale-100' : '-translate-y-6 scale-95'}`}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between px-3 py-2 md:px-3.5 md:py-2.5 2xl:px-4 2xl:py-3 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 shrink-0">
          <div className="flex items-center gap-2 overflow-hidden">
            <h3 className="font-semibold text-sm 2xl:text-base text-gray-700 dark:text-gray-200 truncate" title={fileName}>
              {fileName} {isPdf && activePage && <span className="opacity-60 font-normal ml-1.5 2xl:ml-2 text-xs 2xl:text-sm">(Página {activePage})</span>}
            </h3>
          </div>

          <button
            onClick={handleClose}
            className="bg-red-600 hover:bg-red-700 text-white rounded-full p-1.5 2xl:p-2 shadow transition-colors shrink-0 ml-2"
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4 md:h-4 2xl:h-5 2xl:w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {fragments.length > 0 && (
          <div className="px-3 py-2 md:px-3.5 md:py-2.5 2xl:px-4 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 overflow-y-auto max-h-40">
            <p className="text-[11px] md:text-xs font-semibold text-gray-600 dark:text-gray-300 mb-2 uppercase tracking-wide">
              Evidencias RAG utilizadas
            </p>
            <div className="space-y-1.5">
              {fragments.map((fragment, idx) => {
                const fragmentPage = fragment?.page || null;
                const buttonClass = fragmentPage && fragmentPage === activePage
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/30'
                  : 'border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900';
                return (
                  <button
                    key={`${fragmentPage || 'np'}-${fragment?.fragment || idx}-${idx}`}
                    type="button"
                    onClick={() => fragmentPage && setActivePage(fragmentPage)}
                    className={`w-full text-left p-2 rounded-md border ${buttonClass} hover:border-blue-400 transition-colors`}
                    disabled={!fragmentPage}
                  >
                    <div className="text-[10px] md:text-xs text-gray-600 dark:text-gray-300 font-medium mb-0.5">
                      {fragmentPage ? `Página ${fragmentPage}` : 'Sin página'}{fragment?.fragment ? ` · Fragmento ${fragment.fragment}` : ''}
                    </div>
                    <div className="text-[10px] md:text-xs text-gray-500 dark:text-gray-400 line-clamp-2">
                      {fragment?.snippet || 'Fragmento recuperado del índice RAG.'}
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        <div className="flex-1 bg-gray-100 dark:bg-gray-900 relative flex items-center justify-center overflow-hidden p-2 md:p-3 2xl:p-4">
          {isPreviewable ? (
            isImage ? (
              <img src={viewerUrl} alt={fileName} className="max-w-full max-h-full object-contain rounded shadow" />
            ) : (
              <iframe
                key={viewerUrl}
                src={viewerUrl}
                className="w-full h-full border-none bg-white rounded shadow"
                title="Visor de Documentos"
                allow="fullscreen"
              />
            )
          ) : (
            <div className="flex flex-col items-center text-center p-6 2xl:p-8 bg-white dark:bg-gray-800 rounded-2xl shadow-sm border border-gray-200 dark:border-gray-700 w-full sm:w-[90%] max-w-sm 2xl:max-w-md">
                <i className="fas fa-file-download text-5xl 2xl:text-6xl text-gray-300 dark:text-gray-600 mb-4 2xl:mb-6"></i>
                <h2 className="text-lg 2xl:text-xl font-bold text-gray-800 dark:text-gray-100 mb-2">Previsualización no disponible</h2>
                <p className="text-gray-500 dark:text-gray-400 mb-6 2xl:mb-8 text-xs 2xl:text-sm">
                  Tu navegador no soporta la visualización directa de archivos .{ext.toUpperCase()}. Descárgalo para verlo en tu equipo.
                </p>
                <a
                  href={fileUrl}
                  download={fileName}
                  className="px-5 py-2.5 2xl:px-6 2xl:py-3 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium transition-colors flex items-center justify-center gap-2 shadow-md w-full sm:w-auto text-sm 2xl:text-base"
                >
                    <i className="fas fa-download"></i> Descargar {fileName}
                </a>
            </div>
          )}
        </div>

        <div className="px-3 py-2 2xl:px-4 2xl:py-2 bg-gray-50 dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700 text-[10px] 2xl:text-xs text-gray-500 flex flex-col sm:flex-row justify-between items-center gap-2 sm:gap-0 shrink-0">
           <span className="text-center sm:text-left">{isPreviewable ? 'Vista previa generada localmente' : 'Requiere descarga para visualización'}</span>
           <a href={fileUrl} download={fileName} className="text-blue-600 dark:text-blue-400 hover:underline font-medium text-center sm:text-right">
             Descargar archivo
           </a>
        </div>
      </div>
    </div>
  );
}
