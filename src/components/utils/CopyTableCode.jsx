import React, { useState } from 'react';

export default function CopyTableCode({ content, targetRef, type, isDarkMode }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      let blobHtml = null;
      let blobText = null;

      if (type === 'code') {
        if (targetRef?.current) {
          // A. HTML: Usamos un wrapper neutro. La tabla dentro ya tiene sus estilos (fondo, bordes, 13px, consolas).
          const htmlContent = `<div style="font-family: sans-serif;">${targetRef.current.outerHTML}</div>`;
          
          blobHtml = new Blob([htmlContent], { type: "text/html" });
          blobText = new Blob([content], { type: "text/plain" });
        } else {
          await navigator.clipboard.writeText(content);
          triggerFeedback();
          return;
        }
      } 
      else if (type === 'table' && targetRef?.current) {
        blobHtml = new Blob([targetRef.current.outerHTML], { type: "text/html" });
        blobText = new Blob([targetRef.current.innerText], { type: "text/plain" }); 
      }

      if (blobHtml && blobText) {
        const data = [new ClipboardItem({ "text/html": blobHtml, "text/plain": blobText })];
        await navigator.clipboard.write(data);
        triggerFeedback();
      }

    } catch (err) {
      console.error("Error copiando:", err);
      try {
        await navigator.clipboard.writeText(content || targetRef?.current?.innerText || "");
        triggerFeedback();
      } catch (e) { console.error(e); }
    }
  };

  const triggerFeedback = () => {
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <button
      onClick={handleCopy}
      className={`copy-exclude absolute top-2 right-2 p-1.5 rounded-md text-xs transition-all opacity-0 group-hover:opacity-100 z-10 border shadow-sm ${
        isDarkMode 
          ? "bg-gray-700 text-gray-300 border-gray-600 hover:bg-gray-600 hover:text-white" 
          : "bg-gray-100 text-gray-600 border-gray-300 hover:bg-gray-200 hover:text-black"
      }`}
      title={copied ? "Copiado" : type === 'code' ? "Copiar código" : "Copiar tabla"}
    >
      <i className={`fas ${copied ? "fa-check" : type === 'code' ? "fa-copy" : "fa-table"}`}></i>
      <span className="ml-1 font-sans font-medium">
        {copied ? "Copiado!" : type === 'code' ? "Código" : "Tabla"}
      </span>
    </button>
  );
}