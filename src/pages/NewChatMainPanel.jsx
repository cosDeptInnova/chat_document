// src/pages/NewChatMainPanel.jsx
import React, { useState, useRef, useEffect } from "react";
import { useSearchParams } from "react-router-dom";
import { FiPhone } from "react-icons/fi";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import getFileIconClass from "../components/utils/GetFileIcon";
import ImagePreviewModal from "../components/utils/ImagePreview";
import FilePreviewIcon from "../components/utils/FilePreviewIcon";
import { useSettings } from "../hooks/useSettings";
import MessageFeedback from "../components/utils/MessageFeedback";
import CopyTableCode from "../components/utils/CopyTableCode";

import {
  sendChatMessage,
  uploadEphemeralFiles,
  fetchConversationDetail,
  uploadChatDocDocument,
  sendChatDocMessage,
  fetchChatdocCsrfToken,
  fetchNlpUploadContext,
  bootstrapWebsearch,
  sendWebSearchMessage,
  rateMessage
} from "../lib/api";


export default function NuevoChatMainPanel({
  isDarkMode,
  initialMessage: initialMessageProp,
  chatId: chatIdProp,
}) {
  // Ajustes de voz desde Settings
  const { volume, speed, tone, language } = useSettings();

  // Parámetros de la URL
  const [searchParams] = useSearchParams();
  const urlInitialMessage = searchParams.get("initialMessage");
  const urlChatId = searchParams.get("chatId");
  const urlChatMode = (searchParams.get("chatMode") || "modelo").toLowerCase();

  // Modo de chat: "modelo" | "chatdoc" | "web"
  const chatMode = ["chatdoc", "web"].includes(urlChatMode) ? urlChatMode : "modelo";


  const effectiveInitialMessage =
    initialMessageProp || urlInitialMessage || "Hola, ¿en qué puedo ayudarte?";

  const initialChatId = chatIdProp || urlChatId || null;

  const [welcomeMessage, setWelcomeMessage] = useState(
    effectiveInitialMessage
  );
  const [fileLimitWarning, setFileLimitWarning] = useState(false);

  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState("");
  const messagesEndRef = useRef(null);
  const [hasStarted, setHasStarted] = useState(false);
  const [isLoadingResponse, setIsLoadingResponse] = useState(false);

  // conversationId real de la BBDD
  const [conversationId, setConversationId] = useState(
    initialChatId ? Number(initialChatId) : null
  );

  // doc_session_id del microservicio chat_document (cuando chatMode = "chatdoc")
  const [docSessionId, setDocSessionId] = useState(null);
  // search_session_id del microservicio web_search (cuando chatMode = "web")
  const [searchSessionId, setSearchSessionId] = useState(null);

  // Gestión de archivos adjuntos (solo para UI del siguiente mensaje)
  const [attachedFiles, setAttachedFiles] = useState([]);

    // 🌐 SOLO WEB: adjuntos procesados que deben ir como contexto en el siguiente prompt
  const [webPendingFiles, setWebPendingFiles] = useState([]); 
  // shape: [{ name: string, ids: string[], text: string }]

    // 🧠 SOLO MODELO: adjuntos procesados que deben ir como contexto en el siguiente prompt
  const [modelPendingFiles, setModelPendingFiles] = useState([]);
  // shape: [{ name: string, ids: string[], text: string }]

  // Si el usuario quita un adjunto mientras se procesa, lo anotamos aquí para no re-inyectarlo
  const modelRemovedDuringUploadRef = useRef(new Set());

  const MAX_MODEL_CONTEXT_CHARS = 15000;

  const clampModelText = (text, max = MAX_MODEL_CONTEXT_CHARS) => {
    const t = String(text ?? "");
    if (t.length <= max) return t;
    return `${t.slice(0, max)}\n\n...[recortado por límite de contexto]`;
  };

  const pickFirstStringModel = (...candidates) =>
    candidates.find((v) => typeof v === "string" && v.trim().length > 0) || "";

  const extractTextForFilenameFromUploadModel = (data, filename) => {
    if (!data) return "";
    const details = Array.isArray(data.details) ? data.details : [];

    const match = details.find((d) => {
      const name = d?.filename || d?.file_name || d?.name;
      return name === filename;
    });

    return pickFirstStringModel(
      match?.extracted_text,
      match?.extractedText,
      match?.text,
      match?.content,
      match?.message
    );
  };

  const buildModelContextBlock = (entries) => {
    const sections = (entries || [])
      .filter((e) => e?.text && String(e.text).trim().length > 0)
      .map((e) => `### ${e.name}\n${e.text}`);

    return clampModelText(sections.join("\n\n"));
  };

  const getModelFileIds = (entries) =>
    (entries || [])
      .flatMap((e) => (Array.isArray(e?.ids) ? e.ids : []))
      .filter(Boolean);

  const mergeModelEntriesByNameLimited = (prev, next, limit = 3) => {
    const map = new Map();
    for (const e of [...(prev || []), ...(next || [])]) {
      if (!e?.name) continue;
      map.set(e.name, e); // el último gana
    }
    return Array.from(map.values()).slice(0, limit);
  };

  // Si el usuario quita un adjunto mientras se procesa, lo anotamos aquí para no re-inyectarlo
  const webRemovedDuringUploadRef = useRef(new Set());

  const MAX_WEB_CONTEXT_CHARS = 15000;

  const clampText = (text, max = MAX_WEB_CONTEXT_CHARS) => {
    const t = String(text ?? "");
    if (t.length <= max) return t;
    return `${t.slice(0, max)}\n\n...[recortado por límite de contexto]`;
  };

  const pickFirstString = (...candidates) =>
    candidates.find((v) => typeof v === "string" && v.trim().length > 0) || "";

  const extractTextForFilenameFromUpload = (data, filename) => {
    if (!data) return "";
    const details = Array.isArray(data.details) ? data.details : [];

    // Intentamos emparejar por filename
    const match = details.find((d) => {
      const name = d?.filename || d?.file_name || d?.name;
      return name === filename;
    });

    // Campos típicos donde suele venir el texto extraído
    return pickFirstString(
      match?.extracted_text,
      match?.extractedText,
      match?.text,
      match?.content,
      match?.message
    );
  };

  const buildWebContextBlock = (entries) => {
    const sections = (entries || [])
      .filter((e) => e?.text && String(e.text).trim().length > 0)
      .map((e) => `### ${e.name}\n${e.text}`);

    return clampText(sections.join("\n\n"));
  };

  const getWebFileIds = (entries) =>
    (entries || []).flatMap((e) => (Array.isArray(e?.ids) ? e.ids : [])).filter(Boolean);

  const mergeWebEntriesByNameLimited = (prev, next, limit = 3) => {
    const map = new Map();
    for (const e of [...(prev || []), ...(next || [])]) {
      if (!e?.name) continue;
      map.set(e.name, e); // el último gana
    }
    return Array.from(map.values()).slice(0, limit);
  };

  const [isProcessingFiles, setIsProcessingFiles] = useState(false);

  const fileInputRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);

  // Modal de imagen
  const [isImageModalOpen, setIsImageModalOpen] = useState(false);
  const [modalImageUrl, setModalImageUrl] = useState(null);

  // Refs / estado para micrófono
  const micBaseRef = useRef("");
  const sessionFinalRef = useRef("");
  const [isListening, setIsListening] = useState(false);
  const recognitionRef = useRef(null);
  const finalTranscriptRef = useRef("");
  const silenceTimerRef = useRef(null);
  const hasSentMessageRef = useRef(false);
  const webHydrationBlockedRef = useRef(new Set());
  const modelHydrationBlockedRef = useRef(new Set());


  // Estado de voz por altavoz
  const [speakingMessageIndex, setSpeakingMessageIndex] = useState(null);
  const [copiedMessageIndex, setCopiedMessageIndex] = useState(null);

  // Preload de vídeos del Voice Agent
  const [videosPreloaded, setVideosPreloaded] = useState(false);
  const [preloadProgress, setPreloadProgress] = useState(0);
  const [preloadError, setPreloadError] = useState(null);

  // Textarea auto-ajustable
  const textareaRef = useRef(null);

    // 🔹 NUEVO: bancos de información (NLP)
  const [nlpDepartments, setNlpDepartments] = useState([]);
  const [scopeLoading, setScopeLoading] = useState(false);
  const [scopeError, setScopeError] = useState(null);
  // Solo se puede seleccionar UNA fuente: "personal" o "dept:<department_directory>"
  const [selectedSources, setSelectedSources] = useState(["personal"]);

  // Helper de selección de bancos (comportamiento tipo radio)
  const toggleSource = (value) => {
    // Siempre dejamos exactamente un valor seleccionado
    setSelectedSources([value]);
  };

  const activeDeptDirs = selectedSources
    .filter((s) => s.startsWith("dept:"))
    .map((s) => s.slice(5));

  const scopeLabel = (() => {
    const getDeptName = (dir) => {
      const dep = nlpDepartments.find(
        (d) => d.department_directory === dir
      );
      return (
        (dep && (dep.name || dep.department_name)) ||
        dir.split("/").slice(-1)[0] ||
        dir
      );
    };

    const first = selectedSources[0];

    // Si la fuente activa es un departamento
    if (first && first.startsWith("dept:")) {
      const dir = first.slice(5);
      return `el departamento ${getDeptName(dir)}`;
    }

    // Fallback / caso "personal"
    return "tu espacio personal";
  })();

  // Efecto: actualizar mensaje de bienvenida si cambian los parámetros
  useEffect(() => {
    setWelcomeMessage(effectiveInitialMessage);
  }, [effectiveInitialMessage]);

  // CSRF para chat_document cuando el modo es chatdoc
  useEffect(() => {
    if (chatMode !== "chatdoc") return;

    fetchChatdocCsrfToken().catch((err) => {
      console.warn("Error inicializando CSRF de chat_document:", err);
    });
  }, [chatMode]);

  // CSRF para web_search cuando el modo es web
  useEffect(() => {
    if (chatMode !== "web") return;
    bootstrapWebsearch().catch((err) => {
      console.warn("Error inicializando CSRF de web_search:", err);
    });
  }, [chatMode]);


  // 🔹 NUEVO: cargar bancos de información (solo en modo "modelo")
  useEffect(() => {
    if (chatMode !== "modelo") return;

    let cancelled = false;
    setScopeLoading(true);

    fetchNlpUploadContext()
      .then((ctx) => {
        if (cancelled) return;
        setNlpDepartments(ctx.departments || []);
        setScopeError(null);
      })
      .catch((err) => {
        if (cancelled) return;
        console.warn("Error obteniendo contexto NLP:", err);
        setScopeError(err?.message || String(err));
      })
      .finally(() => {
        if (!cancelled) setScopeLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [chatMode]);

  // Cargar conversación desde el backend si tenemos conversationId inicial
  useEffect(() => {
    if (!conversationId) return;

    // ✅ En modo web, si marcamos este conversationId como "bloqueado",
    // evitamos que el fetch inicial sobrescriba la respuesta optimista local.
    if (chatMode === "web" && webHydrationBlockedRef.current.has(conversationId)) {
      return;
    }

    if (chatMode === "modelo" && modelHydrationBlockedRef.current.has(conversationId)) {
      return;
    }

    let cancelled = false;

    const loadConversation = async () => {
      try {
        const data = await fetchConversationDetail(conversationId);
        if (cancelled) return;

        const mappedMessages = (data.messages || []).map((m) => {
          const sender = (m.sender || "").toUpperCase();
          const isUser = sender === "USER";
          return {
            id: m.id,
            is_liked: m.is_liked,

            role: isUser ? "user" : "system",
            content: m.content || "",
            files: [],
          };
        });

        //Tras una hidratación real, marcamos esta conversación como ya "conocida"
        if (chatMode === "web") {
          webHydrationBlockedRef.current.add(conversationId);
        }

        setMessages(mappedMessages);
        if (mappedMessages.length > 0) {
          setHasStarted(true);
        }
      } catch (err) {
        console.error("Error cargando conversación:", err);
      }
    };

    loadConversation();
    return () => {
      cancelled = true;
    };
  }, [conversationId, chatMode]);


  // Scroll automático al final
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-crecimiento del textarea
  const adjustTextareaHeight = () => {
    if (!textareaRef.current) return;
    const target = textareaRef.current;
    target.style.height = "auto";
    const maxHeight = 128; // aprox 5 líneas
    const newHeight = Math.min(target.scrollHeight, maxHeight);
    target.style.height = `${newHeight}px`;
  };

  useEffect(() => {
    const raf = requestAnimationFrame(() => {
      adjustTextareaHeight();
    });
    return () => cancelAnimationFrame(raf);
  }, [inputMessage]);

  // --- Llamadas reales al backend ---

  /**
   * Subida de archivos desde el botón de adjuntar.
   *
   * - Modo "modelo": /api/modelo/uploadfile/ (archivos en vuelo).
   * - Modo "chatdoc": /api/chatdoc/document/upload (un solo documento para chatear).
   */
    const uploadFilesToBackend = async (files) => {
    if (!files || files.length === 0) {
      return {
        ids: [],
        message: null,
        details: [],
        conversationId: null,
        docSessionId: null,
        rawData: null,
      };
    }

    try {
      // 🚩 MODO CHAT CON DOCUMENTOS
      if (chatMode === "chatdoc") {
        const [file] = files;
        if (!file) {
          return {
            ids: [],
            message: null,
            details: [],
            conversationId: null,
            docSessionId: null,
            rawData: null,
          };
        }

        const data = await uploadChatDocDocument(file);
        console.log("Respuesta /api/chatdoc/document/upload:", data);

        const docSessionFromUpload =
          data.doc_session_id || data.docSessionId || null;
        const convFromUpload = data?.conversation_id ?? null;

        const message =
          data?.message ||
          `Documento "${data?.file_name || file.name}" procesado y comprendido para chatear.`;

        const details = [
          {
            filename: data?.file_name || file.name,
            message: "Documento preparado para hablar con documentos.",
            error: null,
          },
        ];

        return {
          ids: [],
          message,
          details,
          conversationId: convFromUpload,
          docSessionId: docSessionFromUpload,
          rawData: data,
        };
      }

      // 🧠 MODO MODELO / 🌐 WEB: subida efímera
      const data = await uploadEphemeralFiles(files);
      console.log("Respuesta /api/modelo/uploadfile/:", data);

      const ids =
        (data && data.ephemeral_file_ids) ||
        data.file_ids ||
        data.ids ||
        (Array.isArray(data) ? data : []);

      const message = data?.message || null;
      const details = Array.isArray(data?.details) ? data.details : [];
      const convFromUpload = data?.conversation_id ?? null;

      return {
        ids,
        message,
        details,
        conversationId: convFromUpload,
        docSessionId: null,
        rawData: data,
      };
    } catch (err) {
      console.error(
        "Error en uploadFilesToBackend (modo:",
        chatMode,
        "):",
        err
      );
      return {
        ids: [],
        message: null,
        details: [],
        conversationId: null,
        docSessionId: null,
        rawData: null,
      };
    }
  };

    const sendMessageToBackend = async (text, options = {}) => {
      const {
        currentConversationId = null,
        fileIds = [],
        nlpDepartmentDirectory = null, // solo “modelo”
      } = options;

      // 🚩 MODO CHAT CON DOCUMENTOS
      if (chatMode === "chatdoc") {
        if (!docSessionId) {
          throw new Error(
            "No hay documento activo para hablar. Sube antes un documento en esta sesión."
          );
        }

        const data = await sendChatDocMessage({
          prompt: text,
          docSessionId,
          conversationId: currentConversationId || conversationId,
          mode: null,
        });

        const aiText =
          data.reply ||
          data.response ||
          data.answer ||
          data.content ||
          "Respuesta generada a partir del documento.";

        const newConversationId =
          data.conversation_id || data.chat_id || currentConversationId;

        const aiMessageId = data.id || data.message_id || null;

        return {
          content: aiText,
          conversationId: newConversationId,
          messageId: aiMessageId,
        };
      }

      // 🌐 MODO WEB SEARCH  ✅ (incluye file_ids/files)
      if (chatMode === "web") {
        const convIdToSend = currentConversationId || conversationId;

        const data = await sendWebSearchMessage({
          prompt: text,

          search_session_id: searchSessionId,
          conversation_id: convIdToSend,

          // ✅ Adjuntos (si tu backend los soporta)
          file_ids: fileIds,
          files: fileIds,

          // compatibilidad extra (wrapper camelCase)
          searchSessionId,
          conversationId: convIdToSend,
        });

        const aiText =
          data.reply ||
          data.response ||
          data.answer ||
          data.content ||
          "Respuesta generada tras la búsqueda web.";

        const newConversationId =
          data.conversation_id || data.conversationId || data.chat_id || convIdToSend;

        const newSearchSessionId =
          data.search_session_id || data.searchSessionId || searchSessionId;

        const aiMessageId = data.id || data.message_id || null;

        return {
          content: aiText,
          conversationId: newConversationId,
          searchSessionId: newSearchSessionId,
          messageId: aiMessageId,
        };
      }

      // 🧠 MODO MODELO CLÁSICO
      const payload = {
        message: text,
        conversationId: currentConversationId,
        choice: "C",
        files: fileIds,
      };

      if (nlpDepartmentDirectory) {
        payload.department_directory = nlpDepartmentDirectory;
      }

      const data = await sendChatMessage(payload);

      const aiText =
        data.reply ||
        data.answer ||
        data.content ||
        data.text ||
        "Respuesta generada por IA";

      const newConversationId =
        data.conversation_id || data.chat_id || currentConversationId;

      const aiMessageId = data.id || data.message_id || null;

      return {
        content: aiText,
        conversationId: newConversationId,
        messageId: aiMessageId,
      };
    };


  // Gestión de archivos adjuntos: selección + subida inmediata al backend
  const handleFiles = async (fileList) => {
    const files = Array.from(fileList);
    if (files.length === 0) return;

    if (isProcessingFiles) return;
    setIsProcessingFiles(true);

    // Si el usuario vuelve a adjuntar un archivo con el mismo nombre, lo "des-bloqueamos"
    if (chatMode === "web") {
      files.forEach((f) => webRemovedDuringUploadRef.current.delete(f.name));
    }
    if (chatMode === "modelo") {
      files.forEach((f) => modelRemovedDuringUploadRef.current.delete(f.name));
    }

    // 🔹 Modo documentos (chatdoc): 1 archivo por sesión
    if (chatMode === "chatdoc") {
      const [firstFile] = files;
      if (!firstFile) {
        setIsProcessingFiles(false);
        return;
      }

      setAttachedFiles([firstFile]);

      try {
        const {
          message,
          details,
          conversationId: convFromUpload,
          docSessionId: docSessionFromUpload,
        } = await uploadFilesToBackend([firstFile]);

        if (docSessionFromUpload) {
          setDocSessionId(docSessionFromUpload);
        }

        if (convFromUpload && convFromUpload !== conversationId) {
          setConversationId(convFromUpload);
          setHasStarted(true);
        }

        if (message) {
          setMessages((prev) => [
            ...prev,
            { role: "system", content: `Gestor de Cosmos (documentos): ${message}` },
          ]);
        } else {
          setMessages((prev) => [
            ...prev,
            {
              role: "system",
              content:
                "Gestor de Cosmos (documentos): Documento procesado. Ya puedes hacer preguntas sobre él.",
            },
          ]);
        }

        if (Array.isArray(details) && details.length > 0) {
          const perFileText = details
            .map((f) => (f.error ? `Documento: ${f.filename} - ERROR: ${f.error}` : `Documento: ${f.filename} - Procesado correctamente.`))
            .join("\n");

          setMessages((prev) => [...prev, { role: "system", content: perFileText }]);
        }

        setAttachedFiles([]);
      } catch (err) {
        console.error("Error en handleFiles (chatdoc):", err);
        setMessages((prev) => [
          ...prev,
          {
            role: "system",
            content:
              "Gestor de Cosmos (documentos): Error procesando el documento. Inténtalo de nuevo.",
          },
        ]);
      } finally {
        setIsProcessingFiles(false);
      }
      return;
    }

    // 🔹 Modo modelo/web: máximo 3 archivos
    const newTotal = attachedFiles.length + files.length;
    if (newTotal > 3) {
      setFileLimitWarning(true);
      setTimeout(() => setFileLimitWarning(false), 3000);
      setIsProcessingFiles(false);
      return;
    }

    // Mostramos miniaturas mientras procesa
    setAttachedFiles((prev) => [...prev, ...files]);

    try {
      const { ids, details, conversationId: convFromUpload, rawData } =
        await uploadFilesToBackend(files);

      if (chatMode === "modelo" && convFromUpload) {
        modelHydrationBlockedRef.current.add(convFromUpload);
      }

      //WEB: bloquea hidratación que podría “pisar” mensajes locales
      if (chatMode === "web" && convFromUpload) {
        webHydrationBlockedRef.current.add(convFromUpload);
      }

      if (convFromUpload && convFromUpload !== conversationId) {
        setConversationId(convFromUpload);
        setHasStarted(true);
      }

      // ✅ WEB: capturamos texto extraído para concatenarlo al prompt
      if (chatMode === "web") {
        const entriesFromResponse = files
          .map((f, idx) => {
            if (webRemovedDuringUploadRef.current.has(f.name)) return null;
            const text = clampText(extractTextForFilenameFromUpload(rawData, f.name));
            const id = Array.isArray(ids) ? ids[idx] : null;
            return { name: f.name, ids: id ? [id] : [], text };
          })
          .filter(Boolean);

        let finalEntries = entriesFromResponse;

        const needsFallback =
          finalEntries.length === 0 ||
          finalEntries.every((e) => !e.text || e.text.trim().length === 0);

        if (needsFallback && convFromUpload) {
          try {
            const conv = await fetchConversationDetail(convFromUpload);
            const convMsgs = Array.isArray(conv?.messages) ? conv.messages : [];

            const localUserSet = new Set(
              (messages || [])
                .filter((m) => m?.role === "user")
                .map((m) => String(m?.content || ""))
            );

            const userMsgs = convMsgs
              .map((m) => ({
                id: m?.id ?? 0,
                sender: String(m?.sender || "").toUpperCase(),
                content: String(m?.content || ""),
              }))
              .filter((m) => m.sender === "USER" && m.content.trim().length > 0)
              .filter((m) => !localUserSet.has(m.content))
              .sort((a, b) => (a.id || 0) - (b.id || 0));

            const lastNewUser = userMsgs[userMsgs.length - 1];

            if (lastNewUser?.content) {
              const combined = clampText(lastNewUser.content);
              finalEntries = files
                .filter((f) => !webRemovedDuringUploadRef.current.has(f.name))
                .map((f, idx) => {
                  const id = Array.isArray(ids) ? ids[idx] : null;
                  return { name: f.name, ids: id ? [id] : [], text: combined };
                });
            }
          } catch (e) {
            console.warn("WEB fallback: no pude leer conversación:", e);
          }
        }

        if (finalEntries.length > 0) {
          setWebPendingFiles((prev) => mergeWebEntriesByNameLimited(prev, finalEntries, 3));
        }
      }

      // Mensaje seguro (sin “texto extraído”)
      setMessages((prev) => [
        ...prev,
        {
          role: "system",
          content:
            chatMode === "web"
              ? "Gestor de Cosmos (web): Archivos listos. Se usarán como contexto cuando envíes tu pregunta."
              : "Gestor de Cosmos: Archivos procesados. Ya puedes preguntar sobre ellos.",
        },
      ]);

      if (Array.isArray(details) && details.length > 0) {
        const perFileText = details
          .map((f) => (f.error ? `Archivo: ${f.filename} - ERROR: ${f.error}` : `Archivo: ${f.filename} - Procesado correctamente.`))
          .join("\n");

        setMessages((prev) => [...prev, { role: "system", content: perFileText }]);
      }

      /**
       * ✅ Mantén comportamiento actual:
       * - WEB: no limpies miniaturas tras procesar (así se ve qué se enviará).
       * - MODELO: puedes elegir:
       *   - Si quieres replicar EXACTAMENTE el patrón web, NO limpies aquí.
       *   - Si prefieres comportamiento anterior (limpiar), déjalo como estaba.
       *
       * Como has pedido "los mismos cambios", dejamos que MODELO NO limpie.
       */
      if (chatMode !== "web") {
        setAttachedFiles([]);
      }
    } catch (err) {
      console.error("Error en handleFiles (modelo/web):", err);
      setMessages((prev) => [
        ...prev,
        {
          role: "system",
          content: "Gestor de Cosmos: Error procesando los archivos. Inténtalo de nuevo.",
        },
      ]);
    } finally {
      setIsProcessingFiles(false);
    }
  };


  const handleFilePreview = (file) => {
    if (file.type.startsWith("image/")) {
      const url = URL.createObjectURL(file);
      setModalImageUrl(url);
      setIsImageModalOpen(true);
    } else {
      const url = URL.createObjectURL(file);
      window.open(url, "_blank");
    }
  };

  const closeImageModal = () => {
    setIsImageModalOpen(false);
    if (modalImageUrl?.startsWith("blob:")) {
      URL.revokeObjectURL(modalImageUrl);
    }
    setModalImageUrl(null);
  };

  // Micrófono: inicialización de Web Speech API
  useEffect(() => {
    if (typeof window === "undefined") return;

    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (!SpeechRecognition) {
      console.warn("El micrófono no está disponible en este navegador.");
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = "es-ES";
    recognition.continuous = true;
    recognition.interimResults = true;

    recognition.onstart = () => {
      finalTranscriptRef.current = "";
      sessionFinalRef.current = "";
    };

    recognition.onresult = (event) => {
      if (hasSentMessageRef.current) return;

      clearTimeout(silenceTimerRef.current);

      let interimTranscript = "";
      let latestFinal = "";

      for (let i = event.resultIndex; i < event.results.length; ++i) {
        const result = event.results[i];
        const text = result[0].transcript.trim();

        if (result.isFinal) {
          latestFinal = text;
        } else {
          interimTranscript += text + " ";
        }
      }

      if (latestFinal) {
        sessionFinalRef.current = [sessionFinalRef.current, latestFinal]
          .filter(Boolean)
          .join(" ");
      }

      const liveMic = (sessionFinalRef.current || interimTranscript).trim();
      const base = micBaseRef.current;

      setInputMessage([base, liveMic].filter(Boolean).join(" "));

      silenceTimerRef.current = setTimeout(() => {
        recognition.stop();
        setIsListening(false);

        const finalText = sessionFinalRef.current.trim();
        micBaseRef.current = [micBaseRef.current, finalText]
          .filter(Boolean)
          .join(" ");

        setInputMessage(micBaseRef.current);

        sessionFinalRef.current = "";
        finalTranscriptRef.current = "";
      }, 3000);
    };

    recognition.onend = () => {
      setIsListening(false);
      clearTimeout(silenceTimerRef.current);
    };

    recognitionRef.current = recognition;
  }, []);


  const copyMessageToClipboard = async (text, index) => {
    const value = String(text ?? "");

    try {
      // 1. Buscamos el div HTML del mensaje
      const elementId = `message-content-${index}`;
      const element = document.getElementById(elementId);

      if (element && typeof ClipboardItem !== "undefined" && navigator.clipboard?.write) {
        try {
          // A. Clonamos el nodo para no afectar lo que ve el usuario
          const clone = element.cloneNode(true);

          // B. Borramos los botones "Copiar código/tabla" usando la clase copy-exclude
          const buttonsToRemove = clone.querySelectorAll('.copy-exclude');
          buttonsToRemove.forEach(btn => btn.remove());

          // C. Preparamos el HTML limpio
          const htmlContent = `<div style="color: black; background: white; font-family: sans-serif;">${clone.innerHTML}</div>`;
          
          const htmlBlob = new Blob([htmlContent], { type: "text/html" });
          const textBlob = new Blob([value], { type: "text/plain" });

          const data = [
            new ClipboardItem({
              "text/html": htmlBlob,
              "text/plain": textBlob,
            }),
          ];

          await navigator.clipboard.write(data);

          setCopiedMessageIndex(index);
          window.setTimeout(() => setCopiedMessageIndex(null), 1200);
          return; 

        } catch (richErr) {
          console.warn("Fallo copia rica (HTML), usando fallback:", richErr);
        }
      }

      // 2. Fallback clásico (Texto plano)
      if (typeof navigator !== "undefined" && navigator.clipboard?.writeText) {
        await navigator.clipboard.writeText(value);
      } else {
        const textarea = document.createElement("textarea");
        textarea.value = value;
        textarea.setAttribute("readonly", "");
        textarea.style.position = "fixed";
        textarea.style.left = "-9999px";
        textarea.style.top = "-9999px";
        document.body.appendChild(textarea);
        textarea.select();
        document.execCommand("copy");
        document.body.removeChild(textarea);
      }

      setCopiedMessageIndex(index);
      window.setTimeout(() => setCopiedMessageIndex(null), 1200);

    } catch (err) {
      console.warn("No se pudo copiar al portapapeles:", err);
    }
  };

  // Renderiza Markdown SOLO para mensajes del bot (role === "system") con estilos en línea para que al copiar se mantenga el formato
  const renderMessageContent = (msg, isDarkMode) => {
    const isUser = msg.role === "user";

    if (isUser) {
      return <div className="whitespace-pre-wrap">{msg.content}</div>;
    }

    const styles = {
      block: {
        // MODO CLARO: Usamos gray-100 (#f3f4f6) en vez de gray-50 para que se note el fondo en Word
        backgroundColor: isDarkMode ? "#1f2937" : "#f3f4f6", 
        color: isDarkMode ? "#e5e7eb" : "#111827",
        // MODO CLARO: Borde un poco más oscuro (#d1d5db) para definir bien la caja
        border: `1px solid ${isDarkMode ? "#374151" : "#d1d5db"}`,
      },
      codeInline: {
        backgroundColor: isDarkMode ? "#374151" : "#f3f4f6",
        color: isDarkMode ? "#f3f4f6" : "#be185d",
        padding: "2px 5px",
        borderRadius: "4px",
        fontFamily: "Consolas, Monaco, 'Courier New', monospace",
        fontSize: "13px", // Píxeles fijos para evitar que se haga enano en Word
        border: `1px solid ${isDarkMode ? "#4b5563" : "#d1d5db"}`
      },
      table: {
        borderCollapse: "collapse",
        width: "100%",
        fontFamily: "Arial, sans-serif"
      },
      th: {
        border: `1px solid ${isDarkMode ? "#6b7280" : "#9ca3af"}`,
        padding: "8px",
        fontWeight: "bold",
        backgroundColor: isDarkMode ? "#374151" : "#e5e7eb",
        color: isDarkMode ? "#fff" : "#000"
      },
      td: {
        border: `1px solid ${isDarkMode ? "#6b7280" : "#9ca3af"}`,
        padding: "8px",
        color: isDarkMode ? "#d1d5db" : "#1f2937"
      }
    };

    return (
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ ...props }) => <p className="m-0 whitespace-pre-wrap mb-2 last:mb-0" {...props} />,
          ul: ({ ...props }) => <ul className="list-disc pl-5 my-1" {...props} />,
          ol: ({ ...props }) => <ol className="list-decimal pl-5 my-1" {...props} />,
          li: ({ ...props }) => <li className="my-0.5" {...props} />,
          strong: ({ ...props }) => <strong className="font-bold" {...props} />,
          em: ({ ...props }) => <em className="italic" {...props} />,
          a: ({ ...props }) => (
            <a className="underline break-words text-blue-500" target="_blank" rel="noreferrer" {...props} />
          ),

          // --- BLOQUE DE CÓDIGO ---
          code: ({ inline, children, ...props }) => {
            if (inline) {
              return <code style={styles.codeInline} {...props}>{children}</code>;
            }
            
            const CodeBlockWrapper = () => {
              const codeRef = React.useRef(null);
              const codeText = String(children).replace(/\n$/, '');

              return (
                <div className="relative group my-3">
                  <CopyTableCode 
                    content={codeText} 
                    targetRef={codeRef} 
                    type="code" 
                    isDarkMode={isDarkMode} 
                  />
                  
                  {/* Contenedor con la tabla. El ref va directo a la tabla. */}
                  <div style={{ overflowX: "auto" }}>
                    <table 
                      ref={codeRef} 
                      style={{ 
                        width: "100%", 
                        borderCollapse: "collapse", 
                        border: styles.block.border, 
                        backgroundColor: styles.block.backgroundColor 
                      }}
                    >
                      <tbody>
                        <tr>
                          <td style={{ padding: "12px", border: "none", color: styles.block.color }}>
                            {/* fontSize: 13px fijo para que Word lo respete */}
                            <pre style={{ margin: 0, fontFamily: "Consolas, Monaco, 'Courier New', monospace", whiteSpace: "pre-wrap", fontSize: "13px" }}>
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

            return <CodeBlockWrapper />;
          },

          // --- TABLAS ---
          table: ({ children, ...props }) => {
            const TableWrapper = () => {
              const tableRef = React.useRef(null);
              return (
                <div className="relative group my-3">
                  <CopyTableCode targetRef={tableRef} type="table" isDarkMode={isDarkMode} />
                  <div className="overflow-x-auto">
                    <table ref={tableRef} style={styles.table} {...props}>
                      {children}
                    </table>
                  </div>
                </div>
              );
            };
            return <TableWrapper />;
          },
          th: ({ ...props }) => <th style={styles.th} {...props} />,
          td: ({ ...props }) => <td style={styles.td} {...props} />,
        }}
      >
        {msg.content}
      </ReactMarkdown>
    );
  };

  // Voz por altavoz (lectura de mensajes)
  const speakMessage = (text, index) => {
    if (typeof window === "undefined" || !window.speechSynthesis) return;

    const utterance = new SpeechSynthesisUtterance(text);

    utterance.volume = volume / 100;
    utterance.rate = speed;
    utterance.pitch = (tone + 10) / 20;

    const languageMap = {
      es: "es-ES",
      en: "en-US",
      fr: "fr-FR",
      de: "de-DE",
      it: "it-IT",
      pt: "pt-PT",
      ja: "ja-JP",
      zh: "zh-CN",
    };

    utterance.lang = languageMap[language] || "es-ES";

    setSpeakingMessageIndex(index);

    utterance.onend = () => setSpeakingMessageIndex(null);

    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(utterance);
  };

  // Preload temprano de vídeos para el Voice Agent
  useEffect(() => {
    const preloadVoiceVideos = () => {
      console.log(
        "🎬 PRELOAD TEMPRANO: Iniciando carga de videos en background..."
      );

      const videos = [
        "/attached_assets/cosmos-speaking.mp4",
        "/attached_assets/cosmos-listening.mp4",
      ];

      let loadedCount = 0;
      const totalVideos = videos.length;

      videos.forEach((videoSrc) => {
        const videoEl = document.createElement("video");
        videoEl.src = videoSrc;
        videoEl.preload = "auto";
        videoEl.muted = true;

        let hasLoaded = false;

        const onLoaded = () => {
          if (hasLoaded) return;
          hasLoaded = true;

          loadedCount++;
          const progress = Math.round((loadedCount / totalVideos) * 100);
          setPreloadProgress(progress);

          console.log(
            `✅ Video precargado (${loadedCount}/${totalVideos}): ${videoSrc}`
          );

          if (loadedCount === totalVideos) {
            setVideosPreloaded(true);
            setPreloadError(null);
            console.log(
              "🚀 TODOS los videos precargados - UI de voz será INSTANTÁNEA"
            );
          }
        };

        const onError = (e) => {
          if (hasLoaded) return;
          hasLoaded = true;

          const errorMsg = `Error precargando: ${videoSrc}`;
          console.error(`❌ ${errorMsg}`, e);
          setPreloadError(errorMsg);

          loadedCount++;
          const progress = Math.round((loadedCount / totalVideos) * 100);
          setPreloadProgress(progress);

          if (loadedCount === totalVideos) {
            console.log("⚠️ Preload completado con algunos errores");
          }
        };

        videoEl.addEventListener("canplaythrough", onLoaded);
        videoEl.addEventListener("loadeddata", onLoaded);
        videoEl.addEventListener("error", onError);

        const timeoutId = setTimeout(() => {
          if (!hasLoaded) {
            console.warn(
              `⏰ Timeout precargando: ${videoSrc} - pero continuando...`
            );
          }
        }, 10000);

        videoEl.addEventListener("canplaythrough", () =>
          clearTimeout(timeoutId)
        );
        videoEl.addEventListener("error", () => clearTimeout(timeoutId));

        videoEl.load();
      });
    };

    preloadVoiceVideos();
  }, []);

  // Abrir Voice Agent en nueva pestaña
  const handleOpenVoiceAgent = () => {
    localStorage.setItem("voiceAgentDarkMode", isDarkMode ? "true" : "false");
    const url = `${window.location.origin}/voice-agent`;
    window.open(url, "_blank", "noopener,noreferrer");
  };

  // --- Envío de mensaje desde el UI ---
    const handleSend = async () => {
    if (isProcessingFiles) return;
    const cleanMessage = inputMessage.replace(/\s+/g, "");
    if (!cleanMessage && attachedFiles.length === 0) return;

    if (chatMode === "chatdoc" && !docSessionId && attachedFiles.length === 0) {
      setMessages((prev) => [
        ...prev,
        {
          role: "system",
          content:
            "Antes de hacer preguntas en modo 'Hablar con documentos', adjunta primero un archivo con el documento.",
        },
      ]);
      return;
    }

    if (isListening && recognitionRef.current) {
      recognitionRef.current.stop();
      setIsListening(false);
    }

    if (!hasStarted) setHasStarted(true);

    const messageToSend = inputMessage.trim();

    setInputMessage("");
    finalTranscriptRef.current = "";
    hasSentMessageRef.current = true;

    const filesSnapshot = [...attachedFiles];
    setAttachedFiles([]);

    setTimeout(() => {
      hasSentMessageRef.current = false;
    }, 100);

    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }

    setIsLoadingResponse(true);

    const userMessage = {
      role: "user",
      content: messageToSend,
      files: filesSnapshot.length > 0 ? filesSnapshot : undefined,
    };
    setMessages((prev) => [...prev, userMessage]);

    // 🔹 Prompt final a backend
    let finalTextForBackend = messageToSend;

    // 🔹 MODELO: inyección de dept (igual que antes)
    if (chatMode === "modelo" && activeDeptDirs.length > 0) {
      const deptStrings = activeDeptDirs.map((dir) => {
        const dep = nlpDepartments.find((d) => d.department_directory === dir);
        const shortName =
          (dep && (dep.name || dep.department_name)) ||
          dir.split("/").slice(-1)[0] ||
          dir;
        return `${shortName} (${dir})`;
      });

      const scopeHint = `Contexto departamentos seleccionados: ${deptStrings.join(", ")}. `;
      finalTextForBackend = `${scopeHint}${messageToSend}`;
    }

    // 🔹 MODELO: derivar department_directory (igual que antes)
    let nlpDepartmentDirectory = null;
    if (chatMode === "modelo" && activeDeptDirs.length > 0) {
      nlpDepartmentDirectory = activeDeptDirs[0];
    }

    //WEB: concatenar el texto extraído de adjuntos al prompt del usuario
    let webFileIdsToSend = [];
    if (chatMode === "web") {
      const ctx = buildWebContextBlock(webPendingFiles);
      webFileIdsToSend = getWebFileIds(webPendingFiles);

      if (ctx && ctx.trim().length > 0) {
        finalTextForBackend = [
          messageToSend,
          "",
          "Contexto de archivos adjuntos (extraído):",
          ctx,
        ]
          .filter(Boolean)
          .join("\n");
      }
    }


    try {
      const {
        content: aiContent,
        conversationId: newConvId,
        searchSessionId: newSearchSessionId,
        messageId: newMessageId,
      } = await sendMessageToBackend(finalTextForBackend, {
        currentConversationId: conversationId,
        nlpDepartmentDirectory,
        fileIds: chatMode === "web" ? webFileIdsToSend : [],
      });


      if (chatMode === "web" && newConvId) {
        webHydrationBlockedRef.current.add(newConvId);
      }
      if (chatMode === "modelo" && newConvId) {
        modelHydrationBlockedRef.current.add(newConvId);
      }

      if (newConvId && newConvId !== conversationId) {
        setConversationId(newConvId);
      }


      if (newSearchSessionId && newSearchSessionId !== searchSessionId) {
        setSearchSessionId(newSearchSessionId);
      }

      const aiResponse = {
        id: newMessageId,
        role: "system",
        content: aiContent,
        is_liked: null,
      };

      setMessages((prev) => [...prev, aiResponse]);

      if (chatMode === "web") {
        setWebPendingFiles([]);
        webRemovedDuringUploadRef.current.clear();
      }
    } 
    
    catch (err) {
      console.error("Error al enviar mensaje al backend:", err);

      const fallbackResponse = {
        role: "system",
        content:
          "Lo siento, ha ocurrido un error al conectar con el asistente. Inténtalo de nuevo en unos segundos.",
      };
      setMessages((prev) => [...prev, fallbackResponse]);
    } finally {
      setIsLoadingResponse(false);
      finalTranscriptRef.current = "";
      clearTimeout(silenceTimerRef.current);
    }
  };


  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();

      if (textareaRef.current) {
        textareaRef.current.focus();
      }

      if (inputMessage.trim() !== "" || attachedFiles.length > 0) {
        handleSend();
      }
    }
  };

  return (
    <div
      className={`relative w-full flex flex-col flex-1 overflow-hidden items-center justify-center ${
        isDarkMode ? "bg-gray-900 text-white" : "bg-white text-gray-900"
      }`}
      onDragOver={(e) => {
        e.preventDefault();
        setIsDragging(true);
      }}
      onDragLeave={() => setIsDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setIsDragging(false);
        handleFiles(e.dataTransfer.files);
      }}
    >
      <div
        className="flex flex-col w-full max-w-4xl 2xl:max-w-6xl h-full relative bg-transparent 
        px-2 sm:px-3 md:px-4 
        pt-2 sm:pt-3 md:pt-4 
        pb-6 md:pb-8 lg:pb-4"
      >
        {/* Contenedor de mensajes con scroll */}
        <div className="relative flex-1 overflow-y-auto scrollbar-hide pr-1 sm:pr-2 pb-2 z-0 max-h-[calc(100vh-10rem)] sm:max-h-[calc(100vh-12rem)]">
          {/* Lista de mensajes */}
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full">
              <h1
                className={`text-xl sm:text-2xl md:text-3xl font-extrabold text-center mb-3 sm:mb-4 md:mb-6 ${
                  isDarkMode ? "text-blue-400" : "text-blue-700"
                }`}
              >
                {welcomeMessage}
              </h1>
              {/* Indicador de modo (opcional, pero útil para los usuarios) */}
              {chatMode === "chatdoc" && (
                <p
                  className={`text-xs sm:text-sm mt-2 ${
                    isDarkMode ? "text-gray-300" : "text-gray-600"
                  }`}
                >
                  Estás en modo <strong>“Hablar con documentos”</strong>. Primero
                  adjunta un documento y luego realiza tus preguntas.
                </p>
              )}
            </div>
          ) : (
            <div className="flex flex-col space-y-1.5 sm:space-y-2 md:space-y-2.5">
              {messages.map((msg, idx) => {
                const isUser = msg.role === "user";
                const messageClass = isUser
                  ? isDarkMode
                    ? "bg-blue-100 border border-blue-600 text-blue-800 self-end"
                    : "bg-blue-500 text-white self-end"
                  : isDarkMode
                  ? "bg-gray-500 border border-white-600 text-white self-start"
                  : "bg-gray-200 text-gray-800 self-start";

                const alignment = isUser ? "self-end" : "self-start";

                return (
                  <div key={idx} className="flex flex-col gap-0.5 sm:gap-1">
                    {/* Archivos adjuntos del mensaje */}
                    {msg.files && msg.files.length > 0 && (
                      <div
                        className={`flex gap-3 flex-wrap p-2 rounded-lg shadow-sm border 
                        ${
                          isDarkMode
                            ? "bg-white border-gray-300"
                            : "bg-gray-100 border-gray-300"
                        } 
                        ${alignment}`}
                      >
                        {msg.files.map((file, fileIdx) => {
                          const isImage = file.type.startsWith("image/");
                          const fileUrl = URL.createObjectURL(file);
                          const extension = file.name
                            .split(".")
                            .pop()
                            .toLowerCase();

                          return (
                            <div
                              key={fileIdx}
                              className="w-14 h-14 md:w-20 md:h-20 rounded border overflow-hidden flex items-center justify-center bg-white shadow"
                            >
                              <div
                                className="w-14 h-14 md:w-20 md:h-20 rounded border overflow-hidden flex items-center justify-center bg-white shadow cursor-pointer"
                                onClick={() => handleFilePreview(file)}
                              >
                                {isImage ? (
                                  <img
                                    src={fileUrl}
                                    alt={file.name}
                                    className="object-cover w-full h-full"
                                  />
                                ) : (
                                  <div className="text-xs text-center px-2">
                                    <i
                                      className={`fas ${getFileIconClass(
                                        extension
                                      )} text-2xl mb-1`}
                                    />
                                    <p
                                      className={`truncate ${
                                        isDarkMode
                                          ? "text-black"
                                          : "text-gray-800"
                                      }`}
                                    >
                                      {file.name}
                                    </p>
                                  </div>
                                )}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}

                    {/* Burbuja de mensaje */}
                    <div
                      className={`chat-message ${
                        msg.role === "user" ? "user" : "system"
                      } 
                      ${messageClass} p-1.5 md:p-3 lg:p-3 rounded-lg text-xs md:text-xs 2xl:text-sm leading-snug`}
                    >
                      {/* Contenido (usuario = texto plano, bot = Markdown) */}
                      <div 
                        className="overflow-x-auto"
                        id={`message-content-${idx}`}
                      >
                        {renderMessageContent(msg, isDarkMode)}
                      </div>

                      {/* Acciones: Copiar + Voz (una sola vez) */}
                      <div className="mt-1 flex items-center gap-2">
                        {/* Copiar */}
                        <button
                          type="button"
                          onClick={() => copyMessageToClipboard(msg.content, idx)}
                          className="text-[10px] sm:text-xs"
                          title="Copiar al portapapeles"
                          aria-label="Copiar al portapapeles"
                        >
                          <i
                            className={`fas ${
                              copiedMessageIndex === idx ? "fa-check" : "fa-copy"
                            } ${
                              isDarkMode
                                ? isUser
                                  ? "text-blue-900 hover:text-white"
                                  : "text-white hover:text-blue-300"
                                : "text-gray-700 hover:text-black"
                            }`}
                          />
                        </button>

                        {/* Voz */}
                        <button
                          type="button"
                          onClick={() => {
                            if (speakingMessageIndex === idx) {
                              window.speechSynthesis?.cancel();
                              setSpeakingMessageIndex(null);
                            } else {
                              speakMessage(msg.content, idx);
                            }
                          }}
                          className="text-[10px] sm:text-2xs text-blue-400 hover:text-blue-600"
                          title="Leer en voz alta"
                          aria-label="Leer en voz alta"
                        >
                          <i
                            className={`fas ${
                              speakingMessageIndex === idx ? "fa-stop" : "fa-volume-up"
                            } ${
                              isDarkMode
                                ? isUser
                                  ? "text-blue-900 hover:text-white"
                                  : "text-white hover:text-blue-300"
                                : "text-gray-700 hover:text-black"
                            }`}
                          />
                        </button>

                        {/* Componente de Feedback (Like/Dislike) */}
                        {/* Solo se muestra si el mensaje NO es del usuario y ADEMÁS si existe msg.id */}
                        {msg.role !== "user" && msg.id && ( 
                          <MessageFeedback 
                            messageId={msg.id} 
                            initialLiked={msg.is_liked} 
                            isDarkMode={isDarkMode} 
                          />
                        )}
                      </div>
                    </div>
                  </div>
                );
              })}

              {/* Cargando respuesta */}
            {isLoadingResponse && (
                <div className="flex items-center space-x-1 self-start text-gray-400 text-xs sm:text-sm animate-pulse px-2 py-1 message-animate">
                <i className="fas fa-spinner fa-spin"></i>
                <span>Pensando...</span>
                </div>
            )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* Contenedor del input + advertencia */}
        <div className="w-full flex flex-col items-center relative pb-5 md:pb-5">
          {/* Input con animación */}
          <div
            className={`w-full max-w-4xl 2xl:max-w-6xl transition-all duration-700 ease-in-out transform ${
              hasStarted ? "translate-y-0" : "-translate-y-4"
            }`}
          >
            <div
              className={`flex flex-col gap-1 md:gap-1 p-2 md:p-2 rounded-xl shadow-lg transition ${
                isDarkMode ? "bg-gray-800" : "bg-white"
              }`}
            >
              {isProcessingFiles && (
                <div
                  className={`flex items-center gap-2 text-xs px-2 py-1 rounded ${
                    isDarkMode ? "bg-gray-700 text-gray-200" : "bg-gray-100 text-gray-700"
                  }`}
                >
                  <i className="fas fa-spinner fa-spin" />
                  <span>Cosmos está procesando y entendiendo el/los documento(s)…</span>
                </div>
              )}

              {/* Archivos adjuntos (UI input) */}
              {attachedFiles.length > 0 && (
                <div className="flex flex-wrap gap-3 mt-1">
                  {attachedFiles.map((file, idx) => (
                    <div
                      key={idx}
                      className={`relative w-22 h-22 md:w-22 md:h-22 rounded-xl border overflow-hidden shadow-md flex items-center justify-center p-1 ${
                        isDarkMode
                          ? "bg-gray-700 border-gray-600"
                          : "bg-gray-100 border-gray-300"
                      }`}
                    >
                      <FilePreviewIcon
                        file={file}
                        onPreview={() => handleFilePreview(file)}
                      />
                      <button
                        onClick={() => {
                          const fileToRemove = attachedFiles[idx];

                          setAttachedFiles((prev) => prev.filter((_, i) => i !== idx));

                          if (chatMode === "web" && fileToRemove?.name) {
                            webRemovedDuringUploadRef.current.add(fileToRemove.name);
                            setWebPendingFiles((prev) => prev.filter((e) => e.name !== fileToRemove.name));
                          }
                        }}

                        className="absolute top-0 right-0 p-1 bg-red-500 rounded-bl-xl hover:bg-red-700 text-white text-xs"
                      >

                        <svg
                          xmlns="http://www.w3.org/2000/svg"
                          className="h-5 w-5"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                          strokeWidth={2}
                        >
                          <path
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            d="M6 18L18 6M6 6l12 12"
                          />
                        </svg>
                      </button>
                    </div>
                  ))}
                </div>
              )}

              {/* Input de texto */}
              <div className="flex">
                <textarea
                  ref={textareaRef}
                  placeholder={
                    chatMode === "chatdoc"
                      ? "Pregunta lo que necesites sobre el documento cargado"
                      : chatMode === "web"
                      ? "Pregunta lo que necesites (buscaré en la web y citaré fuentes)"
                      : "Pregunta lo que necesites"
                  }

                  rows={1}
                  className={`
                    w-full px-2 py-2 rounded-xl resize-none leading-relaxed
                    focus:outline-none transition
                    md:py-2 md:max-h-[7rem]
                    max-h-[6rem] min-h-[2rem]
                    overflow-y-auto
                    scrollbar-thin
                    text-sm md:text-sm
                    ${
                      isDarkMode
                        ? "bg-gray-700 text-gray-100 placeholder-gray-400"
                        : "bg-gray-100 text-gray-900 placeholder-gray-500"
                    }
                  `}
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onInput={adjustTextareaHeight}
                  onKeyDown={handleKeyDown}
                />
              </div>

              {/* Botones */}
              <div className="flex justify-between items-center">
                <div className="flex gap-1 md:gap-2 items-center">
                  {/* Adjuntar archivos */}
                  <button
                    disabled={isProcessingFiles}
                    className={`p-2 md:p-2 rounded-full transition ${
                      isProcessingFiles
                        ? "opacity-50 cursor-not-allowed"
                        : isDarkMode
                        ? "hover:bg-gray-600"
                        : "hover:bg-gray-200"
                    }`}
                    onClick={() => {
                      if (isProcessingFiles) return;
                      fileInputRef.current?.click();
                    }}
                  >

                    <i
                      className={`fas fa-paperclip ${
                        isDarkMode ? "text-gray-300" : "text-gray-600"
                      }`}
                    ></i>
                    <input
                      type="file"
                      multiple={chatMode !== "chatdoc"} // en modo documento, solo un archivo tiene sentido
                      ref={fileInputRef}
                      disabled={isProcessingFiles}
                      onChange={(e) => {
                        handleFiles(e.target.files);
                        e.target.value = null;
                      }}
                      style={{ display: "none" }}
                    />
                  </button>

                  {/* 🔹 NUEVO: bancos de información (solo modo "modelo") */}
                  {chatMode === "modelo" && (
                    <div className="mt-1 flex flex-col gap-1">
                      <div className="flex items-center justify-between">
                        <span
                          className={`text-[11px] sm:text-xs ${
                            isDarkMode ? "text-gray-300" : "text-gray-600"
                          }`}
                        >
                          {scopeLoading
                            ? "Cargando bancos de información..."
                            : `Miraré aquí lo que me preguntes: ${scopeLabel}`}
                        </span>
                        {scopeError && (
                          <span className="text-[10px] text-red-400">
                            {scopeError}
                          </span>
                        )}
                      </div>
                      <div className="flex flex-wrap gap-1.5">
                        {/* Chip personal */}
                        <button
                          type="button"
                          onClick={() => toggleSource("personal")}
                          className={`px-2 py-1 rounded-full text-[11px] sm:text-xs border transition ${
                            selectedSources.includes("personal")
                              ? isDarkMode
                                ? "bg-blue-600 text-white border-blue-500"
                                : "bg-blue-500 text-white border-blue-500"
                              : isDarkMode
                              ? "bg-gray-700 text-gray-200 border-gray-500 hover:border-blue-400"
                              : "bg-gray-100 text-gray-700 border-gray-300 hover:border-blue-400"
                          }`}
                        >
                          Personal
                        </button>

                        {/* Chips de departamentos */}
                        {nlpDepartments.map((dep) => {
                          const dir = dep.department_directory;
                          if (!dir) return null;
                          const shortName =
                            (dep.name || dep.department_name) ||
                            dir.split("/").slice(-1)[0] ||
                            dir;
                          const value = `dept:${dir}`;
                          const selected = selectedSources.includes(value);

                          return (
                            <button
                              key={dir}
                              type="button"
                              onClick={() => toggleSource(value)}
                              className={`px-2 py-1 rounded-full text-[11px] sm:text-xs border transition ${
                                selected
                                  ? isDarkMode
                                    ? "bg-indigo-500 text-white border-indigo-400"
                                    : "bg-indigo-600 text-white border-indigo-500"
                                  : isDarkMode
                                  ? "bg-gray-700 text-gray-200 border-gray-500 hover:border-indigo-400"
                                  : "bg-gray-100 text-gray-700 border-gray-300 hover:border-indigo-400"
                              }`}
                            >
                              {shortName}
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>

                <div className="flex gap-1 md:gap-2 items-center">
                  {/* Botón del teléfono para acceder a la ventana del chat con el agente por voz */}
                  <button
                    onClick={handleOpenVoiceAgent}
                    className={`
                      relative flex items-center justify-center
                      w-9 h-9 md:w-10 md:h-10 rounded-full border-none cursor-pointer
                      transition-all duration-300 ease-in-out
                      shadow-lg hover:shadow-xl hover:scale-110 active:scale-95
                      ${videosPreloaded 
                        ? isDarkMode 
                          ? 'bg-blue-400 hover:bg-blue-500'
                          : 'bg-blue-700 hover:bg-blue-800'
                        : isDarkMode 
                          ? 'bg-red-600 hover:bg-red-700' 
                          : 'bg-red-500 hover:bg-red-600'
                      }
                    `}
                    title={
                      videosPreloaded
                        ? "Pulsa para hablar con un agente"
                        : `La función hablar con un agente está cargando ${preloadProgress}%`
                    }
                  >
                    {/* Icono del teléfono en blanco */}
                    <FiPhone 
                      size={20} 
                      className="text-white transition-colors duration-300"
                    />
                    
                    {/* Indicador de estado de preload */}
                    {!videosPreloaded && (
                      <div className="absolute -top-1 -right-1">
                        <div className="relative flex">
                          <div className="absolute animate-ping w-3 h-3 bg-amber-400 rounded-full opacity-75" />
                          <div className="relative w-3 h-3 bg-amber-500 rounded-full border border-white" />
                        </div>
                      </div>
                    )}
                    
                    {videosPreloaded && (
                      <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full border border-white" />
                    )}
                  </button>

                  {/* Micrófono */}
                  <button
                    onClick={() => {
                      if (!recognitionRef.current) return;

                      if (isListening) {
                        recognitionRef.current.stop();
                        setIsListening(false);
                      } else {
                        micBaseRef.current = inputMessage.trim();
                        finalTranscriptRef.current = "";
                        sessionFinalRef.current = "";

                        recognitionRef.current.start();
                        setIsListening(true);
                      }

                      if (textareaRef.current) textareaRef.current.focus();
                    }}
                    className={`p-2 rounded-full transition ${
                      isDarkMode ? "hover:bg-gray-600" : "hover:bg-gray-200"
                    }`}
                  >
                    <i
                      className={`fas ${
                        isListening ? "fa-circle-stop" : "fa-microphone"
                      } ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}
                    ></i>
                  </button>

                  {/* Enviar */}
                  <button
                    onClick={handleSend}
                    disabled={
                      isProcessingFiles ||
                      (inputMessage.trim() === "" && attachedFiles.length === 0)
                    }
                    className={`p-2 rounded-xl transition ${
                      isProcessingFiles ||
                      (inputMessage.trim() === "" && attachedFiles.length === 0)
                        ? isDarkMode
                          ? "bg-gray-700 text-gray-400 opacity-60 cursor-not-allowed"
                          : "bg-gray-200 text-gray-500 opacity-60 cursor-not-allowed"
                        : "bg-blue-500 hover:bg-blue-600 text-white"
                    }`}
                  >
                    {inputMessage.trim() !== "" || attachedFiles.length > 0 ? (
                      <i className="fas fa-paper-plane"></i>
                    ) : (
                      <i className="fa fa-arrow-up" aria-hidden="true"></i>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Frase de advertencia */}
          <p
            className={`absolute bottom-0 text-center text-[9.2px] sm:text-xs md:text-xs ${
              isDarkMode ? "text-gray-400" : "text-gray-500"
            }`}
          >
            COSMOS puede cometer errores. Considera verificar la información
            importante.
          </p>
        </div>
      </div>

      <ImagePreviewModal
        isOpen={isImageModalOpen}
        onClose={closeImageModal}
        imageUrl={modalImageUrl}
        isDarkMode={isDarkMode}
      />

      {/* Aviso límite de archivos */}
      {fileLimitWarning && (
        <div
          className="mt-2 flex justify-center
            fixed  bottom-28 z-50
            pointer-events-none"
        >
          <div
            className={`
              inline-block px-4 py-2 rounded-lg shadow text-sm font-medium
              ${isDarkMode ? "bg-red-600 text-white" : "bg-red-500 text-white"}
              animate-fadeIn
              pointer-events-auto
            `}
          >
            Solo puedes adjuntar hasta 3 archivos.
          </div>
        </div>
      )}
    </div>
  );
}