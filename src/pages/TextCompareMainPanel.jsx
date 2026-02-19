// src/pages/TextCompareMainPanel.jsx
import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  fetchComparerCsrfToken,
  startTextCompareJob,
  pollTextCompareProgress,
  openResultInNewTab,
  downloadTextCompareReport,
} from "../lib/api"; // bridges del backend
import FilePreviewIcon from "../components/utils/FilePreviewIcon";
import getFileIconClass from "../components/utils/GetFileIcon";

const ALLOWED_EXT = [".pdf", ".doc", ".docx", ".txt"];
const MAX_MB_PER_FILE = 25; // debe cuadrar con EPHEMERAL_MAX_FILE_MB del back
const POLL_MS = 1200;

function extOf(name = "") {
  const idx = name.lastIndexOf(".");
  return idx >= 0 ? name.slice(idx).toLowerCase() : "";
}

function bytesToMB(b) {
  return (b / (1024 * 1024)).toFixed(2);
}

export default function TextCompareMainPanel({ isDarkMode }) {
  // CSRF
  const [csrfReady, setCsrfReady] = useState(false);
  const [csrfError, setCsrfError] = useState(null);

  // Archivos
  const [fileA, setFileA] = useState(null);
  const [fileB, setFileB] = useState(null);

  // Validación/avisos
  const [warn, setWarn] = useState(null);
  const [error, setError] = useState(null);

  // Opciones del comparador
  const [euroMode, setEuroMode] = useState("strict"); // strict | decimal | loose
  const [minEuro, setMinEuro] = useState(""); // string para controlar input
  const [engine, setEngine] = useState("auto"); // <-- AQUÍ, DENTRO DEL COMPONENTE

  // Job
  const [sid, setSid] = useState(null);
  const [progress, setProgress] = useState({
    percent: 0,
    step: "—",
    detail: "",
    status: "idle",
  });
  const [isSubmitting, setIsSubmitting] = useState(false);
  const pollTimer = useRef(null);
  const abortRef = useRef({ aborted: false });

  // UI varias
  const [dragging, setDragging] = useState(false);
  const [showEmbed, setShowEmbed] = useState(false); // ver en iframe (si el back lo permite)
  const iframeRef = useRef(null);

  // ================== CSRF bootstrap ==================
  useEffect(() => {
    let cancelled = false;
    fetchComparerCsrfToken()
      .then(() => {
        if (!cancelled) {
          setCsrfReady(true);
          setCsrfError(null);
        }
      })
      .catch((err) => !cancelled && setCsrfError(err?.message || "Error obteniendo CSRF"));
    return () => {
      cancelled = true;
    };
  }, []);

  // ================== Helpers ==================
  const validateFile = (f) => {
    if (!f) return "Archivo inválido";
    const okExt = ALLOWED_EXT.includes(extOf(f.name));
    if (!okExt)
      return `Extensión no permitida (${extOf(f.name)}). Usa: ${ALLOWED_EXT.join(", ")}`;
    if (f.size > MAX_MB_PER_FILE * 1024 * 1024) {
      return `El archivo "${f.name}" supera ${MAX_MB_PER_FILE} MB (${bytesToMB(
        f.size
      )} MB).`;
    }
    return null;
  };

  const putFile = (f) => {
    if (!f) return;
    const msg = validateFile(f);
    if (msg) {
      setWarn(msg);
      setTimeout(() => setWarn(null), 3500);
      return;
    }
    if (!fileA) setFileA(f);
    else if (!fileB) setFileB(f);
    else {
      setWarn("Solo se admiten dos ficheros: A y B.");
      setTimeout(() => setWarn(null), 2500);
    }
  };

  const stopPolling = () => {
    if (pollTimer.current) {
      clearInterval(pollTimer.current);
      pollTimer.current = null;
    }
  };

  const resetAll = () => {
    setFileA(null);
    setFileB(null);
    setSid(null);
    setProgress({ percent: 0, step: "—", detail: "", status: "idle" });
    setIsSubmitting(false);
    setError(null);
    setWarn(null);
    setShowEmbed(false);
    abortRef.current.aborted = true;
    stopPolling();
  };

  // ================== Drag & Drop ==================
  const onDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const files = Array.from(e.dataTransfer.files || []);
    files.slice(0, 2).forEach(putFile);
  };

  // ================== Enviar comparación ==================
  const handleStartCompare = async () => {
    setError(null);

    if (!csrfReady) {
      setError("CSRF no inicializado.");
      return;
    }
    if (!fileA || !fileB) {
      setWarn("Selecciona los dos ficheros (A y B).");
      setTimeout(() => setWarn(null), 2500);
      return;
    }
    const vA = validateFile(fileA);
    const vB = validateFile(fileB);
    if (vA || vB) {
      setWarn(vA || vB);
      setTimeout(() => setWarn(null), 3500);
      return;
    }

    setIsSubmitting(true);
    abortRef.current.aborted = false;

    try {
      const options = {
        euro_mode: euroMode,
        min_euro: minEuro !== "" ? Number(minEuro) : null,
        engine,
      };

      const { sid: newSid } = await startTextCompareJob({
        fileA,
        fileB,
        options,
      });

      setSid(newSid);
      setProgress({ percent: 5, step: "Empezando", detail: "", status: "running" });

      // Polling de progreso
      stopPolling();
      pollTimer.current = setInterval(async () => {
        try {
          if (abortRef.current.aborted) return;
          const pr = await pollTextCompareProgress(newSid);
          setProgress(pr);
          if (pr.status === "done" || pr.status === "error") {
            stopPolling();
            setIsSubmitting(false);
          }
        } catch (err) {
          console.warn("Error en polling:", err);
        }
      }, POLL_MS);
    } catch (err) {
      console.error("Error iniciando comparación:", err);
      setError(err?.message || "No se pudo iniciar la comparación.");
      setIsSubmitting(false);
    }
  };

  // ================== Acciones sobre el resultado ==================
  const handleOpenResult = () => {
    if (!sid) return;
    openResultInNewTab(sid);
  };

  const handleDownloadPdf = async () => {
    if (!sid) return;
    try {
      await downloadTextCompareReport(sid);
    } catch (err) {
      setWarn("No hay informe PDF disponible aún.");
      setTimeout(() => setWarn(null), 2500);
    }
  };

  const canStart = useMemo(() => {
    return !!fileA && !!fileB && !isSubmitting && csrfReady;
  }, [fileA, fileB, isSubmitting, csrfReady]);

  // ================== Render ==================
  return (
    <div
      className={`relative w-full flex flex-col flex-1 overflow-hidden  ${
        isDarkMode ? "bg-gray-900 text-white" : "bg-white text-gray-900"
      }`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
    >

      <div className="w-full h-full flex flex-col overflow-y-auto scrollbar-hide px-4 md:px-10 2xl:px-20">
        {/* Header */}
        <header className="mb-6 md:mb-8 2xl:mb-12">
          <div className="flex flex-col gap-2">
            <div>
              <h1
                className={`text-xl md:text-2xl 2xl:text-4xl font-extrabold tracking-tight transition-all duration-300 ${
                  isDarkMode ? "text-blue-300" : "text-blue-700"
                }`}
              >
                Comparador de documentos
              </h1>
              <p
                className={`mt-2 text-xs md:text-base 2xl:text-lg max-w-4xl transition-all duration-300 ${
                  isDarkMode ? "text-gray-400" : "text-gray-600"
                }`}
              >
                Detecta cambios reales de contenido entre versiones de contratos, informes
                y documentos ofimáticos (DOC/DOCX/PDF/TXT).
              </p>
            </div>
          </div>
        </header>

        {/* Avisos */}
        <div className="space-y-3 mb-6">
          {!csrfReady && (
            <div
              className={`px-4 py-2 rounded text-sm 2xl:text-base ${
                isDarkMode ? "bg-yellow-700 text-white" : "bg-yellow-100 text-yellow-800"
              }`}
            >
              Inicializando protección CSRF…
            </div>
          )}
          {csrfError && (
            <div
              className={`px-4 py-2 rounded text-sm 2xl:text-base ${
                isDarkMode ? "bg-red-700 text-white" : "bg-red-100 text-red-800"
              }`}
            >
              {csrfError}
            </div>
          )}
          {warn && (
            <div
              className={`px-4 py-2 rounded text-sm 2xl:text-base ${
                isDarkMode ? "bg-amber-700 text-white" : "bg-amber-100 text-amber-800"
              }`}
            >
              {warn}
            </div>
          )}
          {error && (
            <div
              className={`px-4 py-2 rounded text-sm 2xl:text-base ${
                isDarkMode ? "bg-red-700 text-white" : "bg-red-100 text-red-800"
              }`}
            >
              {error}
            </div>
          )}
        </div>

        {/* Selección de archivos */}
        <div
          className={`grid grid-cols-1 md:grid-cols-2 gap-4 2xl:gap-8 ${
            dragging ? (isDarkMode ? "ring-2 ring-blue-400 rounded-xl" : "ring-2 ring-blue-600 rounded-xl") : ""
          }`}
        >
          {/* Lado A */}
          <div
            className={`p-3 2xl:p-6 rounded-xl border transition-all ${
              isDarkMode ? "bg-gray-800 border-gray-700" : "bg-gray-50 border-gray-300"
            }`}
          >
            <div className="flex items-center justify-between mb-3 2xl:mb-5">
              <h2 className="font-semibold text-xs md:text-sm 2xl:text-base">Documento A</h2>
              {fileA && (
                <button
                  onClick={() => setFileA(null)}
                  className={`text-xs 2xl:text-sm px-2 py-1 2xl:px-3 2xl:py-1.5 rounded transition-colors ${
                    isDarkMode ? "bg-gray-700 hover:bg-gray-600" : "bg-gray-200 hover:bg-gray-300"
                  }`}
                >
                  Quitar
                </button>
              )}
            </div>

            <div>
              {!fileA ? (
                <label
                    className={`block w-full h-22 md:h-28 2xl:h-36 border-2 border-dashed rounded-xl cursor-pointer flex items-center justify-center text-center transition-all ${
                    isDarkMode
                        ? "border-gray-600 text-gray-300 hover:bg-gray-700"
                        : "border-gray-300 text-gray-600 hover:bg-gray-100"
                    }`}
                >
                  <input
                    type="file"
                    className="hidden"
                    accept={ALLOWED_EXT.join(",")}
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) putFile(f);
                      e.target.value = null;
                    }}
                  />
                  <div className="p-4">
                    <i className="fas fa-file-upload text-xl md:text-2xl 2xl:text-4xl mb-2 block opacity-80" />
                    <div className="text-sm 2xl:text-lg font-medium">Arrastra o pulsa para subir</div>
                    <div className="text-xs 2xl:text-sm mt-1 opacity-70">
                      PDF, DOC, DOCX o TXT · máx {MAX_MB_PER_FILE} MB
                    </div>
                  </div>
                </label>
              ) : (
                <div
                  className={`flex items-center gap-3 2xl:gap-5 p-2 2xl:p-4 rounded border ${
                    isDarkMode ? "bg-gray-700 border-gray-600" : "bg-white border-gray-200"
                  }`}
                >
                  <div className={`w-12 h-12 2xl:w-16 2xl:h-16 flex items-center justify-center rounded-lg ${
                         isDarkMode ? "bg-gray-800" : "bg-gray-100"
                       }`}
                  >
                    <i
                      className={`fas ${getFileIconClass(extOf(fileA.name).slice(1))} text-2xl 2xl:text-3xl`}
                    />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm 2xl:text-lg font-medium truncate">{fileA.name}</div>
                    <div className="text-xs 2xl:text-sm opacity-70">{bytesToMB(fileA.size)} MB</div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Lado B */}
          <div
            className={`p-3 2xl:p-6 rounded-xl border transition-all ${
              isDarkMode ? "bg-gray-800 border-gray-700" : "bg-gray-50 border-gray-300"
            }`}
          >
            <div className="flex items-center justify-between mb-3 2xl:mb-5">
              <h2 className="font-semibold text-xs md:text-sm 2xl:text-base">Documento B</h2>
              {fileB && (
                <button
                  onClick={() => setFileB(null)}
                  className={`text-xs 2xl:text-sm px-2 py-1 2xl:px-3 2xl:py-1.5 rounded transition-colors ${
                    isDarkMode ? "bg-gray-700 hover:bg-gray-600" : "bg-gray-200 hover:bg-gray-300"
                  }`}
                >
                  Quitar
                </button>
              )}
            </div>

            <div>
              {!fileB ? (
                <label
                  className={`block w-full h-22 md:h-28 2xl:h-36 border-2 border-dashed rounded-xl cursor-pointer flex items-center justify-center text-center transition-all ${
                    isDarkMode
                      ? "border-gray-600 text-gray-300 hover:bg-gray-700"
                      : "border-gray-300 text-gray-600 hover:bg-gray-100"
                  }`}
                >
                  <input
                    type="file"
                    className="hidden"
                    accept={ALLOWED_EXT.join(",")}
                    onChange={(e) => {
                      const f = e.target.files?.[0];
                      if (f) putFile(f);
                      e.target.value = null;
                    }}
                  />
                  <div className="p-4">
                    <i className="fas fa-file-upload text-xl md:text-2xl 2xl:text-4xl mb-2 block opacity-80" />
                    <div className="text-sm 2xl:text-lg font-medium">Arrastra o pulsa para subir</div>
                    <div className="text-xs 2xl:text-sm mt-1 opacity-70">
                      PDF, DOC, DOCX o TXT · máx {MAX_MB_PER_FILE} MB
                    </div>
                  </div>
                </label>
              ) : (
                <div
                  className={`flex items-center gap-3 2xl:gap-5 p-2 2xl:p-4 rounded border ${
                    isDarkMode ? "bg-gray-700 border-gray-600" : "bg-white border-gray-200"
                  }`}
                >
                  <div className={`w-12 h-12 2xl:w-16 2xl:h-16 flex items-center justify-center rounded-lg ${
                         isDarkMode ? "bg-gray-800" : "bg-gray-100"
                       }`}
                  >
                    <i
                      className={`fas ${getFileIconClass(extOf(fileB.name).slice(1))} text-2xl 2xl:text-3xl`}
                    />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm 2xl:text-lg font-medium truncate">{fileB.name}</div>
                    <div className="text-xs 2xl:text-sm opacity-70">{bytesToMB(fileB.size)} MB</div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Acciones */}
        <div className="mt-4 2xl:mt-8 flex flex-wrap gap-2 2xl:gap-4 items-center">
          <button
            disabled={!canStart}
            onClick={handleStartCompare}
            className={`px-5 py-2.5 2xl:px-8 2xl:py-4 rounded-xl font-bold text-sm 2xl:text-lg transition-all shadow-sm active:scale-95 ${
              canStart
                ? isDarkMode
                  ? "bg-blue-500 hover:bg-blue-600 text-white"
                  : "bg-blue-600 hover:bg-blue-700 text-white"
                : isDarkMode
                ? "bg-gray-700 text-gray-400"
                : "bg-gray-200 text-gray-500"
            }`}
            title={csrfReady ? "" : "CSRF no listo"}
          >
            {isSubmitting ? (
              <span className="inline-flex items-center gap-2">
                <i className="fas fa-spinner fa-spin" /> Comparando…
              </span>
            ) : (
              "Comparar Documentos"
            )}
          </button>

          {/* Solo activos si hay SID y terminó */}
          {/* Botones secundarios */}
          <div className="flex gap-2 2xl:gap-4 flex-wrap">
            <button
                disabled={!sid || !(progress.status === "done" || progress.status === "error")}
                onClick={handleOpenResult}
                className={`px-4 py-2.5 2xl:px-6 2xl:py-3 rounded-xl font-semibold text-sm 2xl:text-base transition-colors ${
                sid && (progress.status === "done" || progress.status === "error")
                    ? isDarkMode
                    ? "bg-indigo-500 hover:bg-indigo-600 text-white"
                    : "bg-indigo-600 hover:bg-indigo-700 text-white"
                    : isDarkMode
                    ? "bg-gray-700 text-gray-400"
                    : "bg-gray-200 text-gray-500"
                }`}
            >
                Ver resultado
            </button>

            <button
                disabled={!sid || progress.status !== "done"}
                onClick={handleDownloadPdf}
                className={`px-4 py-2.5 2xl:px-6 2xl:py-3 rounded-xl font-semibold text-sm 2xl:text-base transition-colors ${
                sid && progress.status === "done"
                    ? isDarkMode
                    ? "bg-emerald-500 hover:bg-emerald-600 text-white"
                    : "bg-emerald-600 hover:bg-emerald-700 text-white"
                    : isDarkMode
                    ? "bg-gray-700 text-gray-400"
                    : "bg-gray-200 text-gray-500"
                }`}
            >
                <i className="fas fa-file-pdf mr-2"></i> Descargar Informe (PDF)
            </button>

            <button
                type="button"
                onClick={resetAll}
                className={`px-4 py-2.5 2xl:px-6 2xl:py-3 rounded-xl font-medium text-sm 2xl:text-base border transition-colors ${
                isDarkMode
                    ? "border-gray-600 text-gray-300 hover:bg-gray-800"
                    : "border-gray-300 text-gray-700 hover:bg-gray-100"
                }`}
            >
                Reset
            </button>
          </div>

          <label className="ml-auto inline-flex items-center gap-2 text-sm 2xl:text-base cursor-pointer select-none py-2">
            <input
              type="checkbox"
              checked={showEmbed}
              onChange={() => setShowEmbed((v) => !v)}
              className="w-4 h-4 2xl:w-5 2xl:h-5 text-blue-600 rounded focus:ring-blue-500"
            />
            <span className={isDarkMode ? "text-gray-300" : "text-gray-700"}>
              Ver resultado embebido (iframe)
            </span>
          </label>
        </div>

        {/* Barra de progreso */}
        {sid && (
          <div
            className={`mt-4 2xl:mt-8 p-3 2xl:p-5 rounded-xl border ${
              isDarkMode ? "bg-gray-800 border-gray-700" : "bg-gray-50 border-gray-300"
            }`}
          >
            <div className="flex items-center justify-between mb-2">
              <div className="text-sm 2xl:text-lg font-semibold">
                Progreso: {progress.percent ?? 0}%
              </div>
              <div
                className={`text-xs 2xl:text-sm ${
                  progress.status === "error"
                    ? isDarkMode
                      ? "text-red-300"
                      : "text-red-600"
                    : isDarkMode
                    ? "text-gray-300"
                    : "text-gray-600"
                }`}
              >
                {progress.step}
                {progress.detail ? ` — ${progress.detail}` : ""}
              </div>
            </div>

            <div
              className={`w-full mt-1 h-3 2xl:h-4 rounded-full overflow-hidden ${
                isDarkMode ? "bg-gray-700" : "bg-gray-200"
              }`}
            >
              <div
                className={`h-full transition-all duration-300 ease-out ${
                  progress.status === "error"
                    ? "bg-red-500"
                    : isDarkMode
                    ? "bg-blue-400"
                    : "bg-blue-600"
                }`}
                style={{
                  width: `${Math.min(100, Math.max(0, progress.percent || 0))}%`,
                }}
              />
            </div>

            {progress.status === "error" && (
              <div
                className={`mt-3 text-sm 2xl:text-base ${
                  isDarkMode ? "text-red-300" : "text-red-700"
                }`}
              >
                <i className="fas fa-exclamation-triangle mr-2"></i>
                Se produjo un error en el procesamiento. Abre el resultado para ver el detalle.
              </div>
            )}
          </div>
        )}

        {/* Iframe opcional para ver /resultado/{sid} sin inyectar HTML */}
        {showEmbed && sid && (
          <div className="mt-6 2xl:mt-10 animate-fadeIn">
            <div className="text-sm 2xl:text-base mb-2 opacity-70">
              Vista previa embebida (si el servidor permite `X-Frame-Options` adecuado).
            </div>
            <div className="w-full h-[60vh] 2xl:h-[75vh] border rounded-2xl overflow-hidden shadow-lg">
              <iframe
                ref={iframeRef}
                title="Resultado de la comparación"
                src={`/api/comparador/resultado/${sid}`}
                className="w-full h-full bg-white"
                sandbox="allow-same-origin allow-scripts allow-popups"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}