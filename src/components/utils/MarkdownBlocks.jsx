import React, { useState, useRef, useLayoutEffect } from 'react';
import CopyTableCode from './CopyTableCode';

// Función para el bloque de código dentro de un contenedor
export const CodeBlock = ({ children, isDarkMode, ...props }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isOverflowing, setIsOverflowing] = useState(false); // Estado para saber si necesitamos el botón de ampliar o no porque es muy corto
  const codeRef = useRef(null);
  const scrollRef = useRef(null);
  
  const codeText = String(children).replace(/\n$/, '');

  const containerBorder = isDarkMode ? "border-gray-700" : "border-gray-300";
  const headerBg = isDarkMode ? "bg-gray-800" : "bg-gray-200";
  const headerText = isDarkMode ? "text-gray-300" : "text-gray-700";
  const contentBg = isDarkMode ? "#1f2937" : "#f3f4f6";
  const contentColor = isDarkMode ? "#e5e7eb" : "#111827";

  // Medir contenido para ver si activamos el botón "Ampliar"
  useLayoutEffect(() => {
    if (scrollRef.current) {
      // 320px es aprox max-h-80 (20rem * 16px)
      setIsOverflowing(scrollRef.current.scrollHeight > 320);
    }
  }, [children]);

  return (
    <div className={`my-4 rounded-lg border ${containerBorder} overflow-hidden shadow-sm flex flex-col`}>
      {/* HEADER */}
      <div className={`flex items-center justify-between px-3 py-2 text-xs select-none ${headerBg} ${headerText}`}>
        <span className="font-mono font-bold opacity-90">Código</span>
        
        <div className="flex items-center gap-3">
          {/* Botón Ampliar: Solo se muestra si hay overflow */}
          {isOverflowing && (
            <button 
              onClick={() => setIsExpanded(!isExpanded)}
              className="hover:text-blue-500 transition-colors flex items-center gap-1 focus:outline-none"
              type="button"
            >
              <i className={`fas ${isExpanded ? "fa-compress-alt" : "fa-expand-alt"}`}></i>
              <span className="hidden sm:inline">{isExpanded ? "Reducir" : "Ampliar"}</span>
            </button>
          )}

          {/* Botón Copiar: Layout limpio sin absolute raros */}
          <CopyTableCode 
            content={codeText} 
            targetRef={codeRef} 
            type="code" 
            isDarkMode={isDarkMode} 
          />
        </div>
      </div>

      {/* CONTENIDO SCROLLEABLE */}
      <div 
        ref={scrollRef}
        className={`relative transition-all duration-300 ease-in-out scrollbar-hide ${
          isExpanded ? "max-h-none" : "max-h-80"
        } overflow-auto scrollbar-thin scrollbar-thumb-gray-400 scrollbar-track-transparent`}
        style={{ backgroundColor: contentBg }} 
      >
        <table 
          ref={codeRef}
          style={{ width: "100%", borderCollapse: "collapse", border: "none", backgroundColor: contentBg }}
        >
          <tbody>
            <tr>
              <td style={{ padding: "16px", border: "none", color: contentColor }}>
                <pre style={{ margin: 0, fontFamily: "Consolas, Monaco, 'Courier New', monospace", whiteSpace: "pre-wrap", fontSize: "13px", lineHeight: "1.5" }}>
                  <code style={{ fontFamily: "inherit" }} {...props}>{children}</code>
                </pre>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
};

// Función para el bloque de la tabla dentro de un contenedor
export const TableBlock = ({ children, isDarkMode, styles, ...props }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [isOverflowing, setIsOverflowing] = useState(false);
  const tableRef = useRef(null);
  const scrollRef = useRef(null);

  const containerBorder = isDarkMode ? "border-gray-700" : "border-gray-300";
  const headerBg = isDarkMode ? "bg-gray-800" : "bg-gray-200";
  const headerText = isDarkMode ? "text-gray-300" : "text-gray-700";

  useLayoutEffect(() => {
    if (scrollRef.current) {
      setIsOverflowing(scrollRef.current.scrollHeight > 320);
    }
  }, [children]);

  return (
    <div className={`my-4 rounded-lg border ${containerBorder} overflow-hidden shadow-sm bg-opacity-50 ${isDarkMode ? "bg-gray-800" : "bg-white"}`}>
      {/* HEADER */}
      <div className={`flex items-center justify-between px-3 py-2 text-xs select-none ${headerBg} ${headerText}`}>
        <span className="font-bold opacity-90 flex items-center gap-2">
          <i className="fas fa-table"></i> Tabla
        </span>
        
        <div className="flex items-center gap-3">
          {/* Botón Ampliar Condicional */}
          {isOverflowing && (
            <button 
              onClick={() => setIsExpanded(!isExpanded)}
              className="hover:text-blue-500 transition-colors flex items-center gap-1 focus:outline-none"
              type="button"
            >
              <i className={`fas ${isExpanded ? "fa-compress-alt" : "fa-expand-alt"}`}></i>
              <span className="hidden sm:inline">{isExpanded ? "Reducir" : "Ampliar"}</span>
            </button>
          )}

          <CopyTableCode targetRef={tableRef} type="table" isDarkMode={isDarkMode} />
        </div>
      </div>

      {/* CONTENIDO SCROLLEABLE */}
      <div 
        ref={scrollRef}
        className={`overflow-x-auto transition-all duration-300 scrollbar-hide ${
          isExpanded ? "max-h-none" : "max-h-80"
        } scrollbar-thin scrollbar-thumb-gray-400 scrollbar-track-transparent`}
      >
        <table ref={tableRef} style={styles.table} {...props} className="w-full">
          {children}
        </table>
      </div>
    </div>
  );
};