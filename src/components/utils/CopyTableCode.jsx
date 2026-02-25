import React, { useState } from 'react';

export default function CopyTableCode({ content, targetRef, type, isDarkMode }) {
  const [copied, setCopied] = useState(false);

  const triggerFeedback = () => {
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleCopy = async () => {
    try {
      let blobHtml = null;
      let blobText = null;

      if (type === 'code') {
        if (targetRef?.current) {
          // Wrapper HTML para Word
          const htmlContent = `<div style="font-family: Consolas, monospace; color: ${isDarkMode ? 'white' : 'black'}; background: ${isDarkMode ? '#1f2937' : '#f3f4f6'};">${targetRef.current.outerHTML}</div>`;
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
      } else {
        // Fallback simple por si algo falló en la creación de los blobs
        await navigator.clipboard.writeText(content || targetRef?.current?.innerText || "");
        triggerFeedback();
      }

    } catch (err) {
      console.warn("Error copia rica, usando texto plano:", err);
      try {
        await navigator.clipboard.writeText(content || targetRef?.current?.innerText || "");
        triggerFeedback();
      } catch (e) { console.error("Fallo total copia", e); }
    }
  };

  return (
    <button
      onClick={(e) => {
        e.stopPropagation();
        handleCopy();
      }}
      className={`copy-exclude flex items-center justify-center gap-1.5 px-2 py-1 rounded text-xs font-medium transition-all border shadow-sm z-50 ${
        isDarkMode 
          ? "bg-gray-700 text-gray-200 border-gray-600 hover:bg-gray-600 hover:text-white" 
          : "bg-white text-gray-600 border-gray-300 hover:bg-gray-50 hover:text-black"
      } ${copied ? "border-green-500 text-green-500" : ""}`}
      title={copied ? "Copiado" : type === 'code' ? "Copiar código" : "Copiar tabla"}
    >
      <i className={`fas ${copied ? "fa-check" : type === 'code' ? "fa-copy" : "fa-table"}`}></i>
      <span>{copied ? "Copiado!" : type === 'code' ? "Copiar" : "Copiar tabla"}</span>
    </button>
  );
}