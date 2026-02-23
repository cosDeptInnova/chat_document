import React from "react";
import { CheckCircleIcon } from "@heroicons/react/24/solid";
import { useNavigate } from "react-router-dom";
import { openNotetaker } from "../../lib/api";

/**
 * Card de caso de uso.
 *
 * Prioridad de navegación:
 *  1) useCase.href  → ruta interna directa (ej: "/main/text-compare")
 *  2) useCase.tab   → navegación SPA con query params (chat / knowledge)
 *  3) useCase.link  → enlace externo (nueva pestaña)
 *
 * - Si useCase.tab está definido:
 *    · tab === "nuevo"        → navega a /main/chat con los query params adecuados
 *    · tab !== "nuevo"        → navega a /main/knowledge (comportamiento actual)
 *
 * - Si useCase.engine está definido (p.ej. "chatdoc"):
 *    · Se añade chatMode=engine a la query string.
 *    · El componente NuevoChatMainPanel leerá chatMode para decidir
 *      qué endpoints de backend usar:
 *        · chatMode = "modelo"  → /api/modelo/query/llm + /api/modelo/uploadfile/
 *        · chatMode = "chatdoc" → /api/chatdoc/document/upload + /api/chatdoc/document/query
 *
 * - Si useCase.link está definido:
 *    · Abre el enlace en una nueva pestaña.
 */
export default function Card({ useCase, isDarkMode, active, onClick }) {
  const navigate = useNavigate();

    const handleClick = async (evt) => {
    // Evita que algún contenedor padre con onClick navegue al chat por defecto
    evt?.preventDefault?.();
    evt?.stopPropagation?.();

    // Marca esta card como seleccionada en el grid
    onClick && onClick();

        // 0) Acciones especiales (externas / integraciones)
    if (useCase.action) {
      if (useCase.action === "notetaker") {
        try {
          await openNotetaker();
        } catch (e) {
          console.error("No se pudo abrir Notetaker:", e);
          alert("No se pudo abrir Notetaker. Revisa tu sesión o inténtalo más tarde.");
        }
        return;
      }
    }


    // 1) Ruta interna directa (p. ej. comparador)
    if (useCase.href) {
      let target = String(useCase.href || "").trim();

      // Normaliza el caso común donde alguien puso "/text-compare" sin el prefijo "/main/"
      if (target === "/text-compare") {
        target = "/main/text-compare";
      }

      navigate(target);
      return;
    }

    // 2) Navegación SPA basada en "tab"
    if (useCase.tab) {
      const isChatTab = useCase.tab === "nuevo";

      // Ruta interna en función del tab (se mantiene la lógica original)
      const path = isChatTab ? "/main/chat" : "/main/knowledge";

      const params = new URLSearchParams();
      params.set("selectedOption", useCase.tab);

      // Mensaje inicial opcional
      if (useCase.initialMessage) {
        params.set("initialMessage", useCase.initialMessage);
      }

      // Motor del chat:
      // - "chatdoc" para casos como "Hablar con documentos"
      // - por defecto "modelo" (COSMOS general)
      const engine = useCase.engine || "modelo";

      if (isChatTab) {
        // 🔹 Este parámetro es el que leerá NewChatMainPanel
        params.set("chatMode", engine);
      }

      // Navegación interna (misma pestaña)
      navigate(`${path}?${params.toString()}`);
      return;
    }

    // 3) Enlace externo en nueva pestaña
    if (useCase.link) {
      const safeUrl = useCase.link;
      try {
        window.open(safeUrl, "_blank", "noopener,noreferrer");
      } catch (e) {
        console.error("No se pudo abrir el enlace externo:", e);
      }
    }
  };

  return (
    <button
      type="button"
      onClick={handleClick}
      aria-pressed={active}
      className={`
        relative flex flex-col items-start text-left
        w-full h-full p-4 md:p-5 2xl:p-6 rounded-2xl border transition-all duration-300
        active:scale-95 md:hover:scale-105 md:hover:-translate-y-1
        ${active
          ? "bg-blue-100 border-blue-600 text-blue-800 scale-105 shadow-lg"
          : isDarkMode
            ? "bg-gray-800 text-gray-200 border-gray-600 hover:bg-gray-700 hover:scale-105 hover:shadow-2xl"
            : "bg-white text-gray-700 border-gray-300 hover:bg-blue-50 hover:scale-105 hover:shadow-2xl"
        }
      `}
    >
      {/* Icono de check si está activa */}
      {active && (
        <CheckCircleIcon className="absolute top-3 right-4 h-6 w-6 text-blue-600 animate-fadeIn" />
      )}

      {/* Imagen, icono o svg genérico */}
      <div className={`mb-2 rounded-xl inline-flex items-center justify-center transition-colors ${active ? "text-yellow-400" : "text-blue-500"}`}>
        {useCase.image ? (
          <img
            src={useCase.image}
            alt={useCase.title}
            className="w-5 h-5 md:w-6 md:h-6 2xl:w-8 2xl:h-8 object-contain"
          />
        ) : useCase.icon ? (
          <useCase.icon className="w-5 h-5 md:w-6 md:h-6 2xl:w-8 2xl:h-8" />
        ) : (
          <svg
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
            strokeWidth={1.5}
            stroke="currentColor"
            className="w-5 h-5 md:w-6 md:h-6 2xl:w-8 2xl:h-8"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M6 3h9l5 5v13a1 1 0 01-1 1H6a1 1 0 01-1-1V4a1 1 0 011-1z"
            />
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              d="M14 3v5h5"
            />
          </svg>
        )}
      </div>

      {/* Contenido */}
      <div className="flex flex-col flex-1 w-full">
        <h2 className={"font-bold text-sm md:text-md 2xl:text-base mb-1.5 leading-tight"}>
            {useCase.title}
        </h2>
        <p className={"text-xs md:text-xs 2xl:text-md leading-relaxed"}>
            {useCase.description}
        </p>
      </div>
    </button>
  );
}
