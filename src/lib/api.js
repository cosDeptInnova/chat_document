// src/lib/api.js

// Base del API del modelo de negocio.
// Detrás de Nginx, /api/modelo/* → FastAPI modelo_negocio
const BASE_API_URL =
  process.env.REACT_APP_MODELO_API_BASE || "/api/modelo";

// Base del API de chat_document.
// Detrás de Nginx, /api/chatdoc/* → FastAPI chat_document (puerto 8100 en el host)
const CHATDOC_API_BASE =
  process.env.REACT_APP_CHATDOC_API_BASE || "/api/chatdoc";

// Base del API NLP/RAG.
// Detrás de Nginx, /api/nlp/* → FastAPI NLP (puerto 5000 en el host)
const NLP_API_BASE =
  process.env.REACT_APP_NLP_API_BASE || "/api/nlp";

// Base del API del COMPARADOR de documentos.
// Detrás de Nginx, /api/comparador/* → FastAPI comparador (puerto 8007 en el host)
const COMPARATOR_API_BASE =
  process.env.REACT_APP_COMPARATOR_API_BASE || "/api/comparador";


const WEBSEARCH_API_BASE =
  process.env.REACT_APP_WEBSEARCH_API_BASE || "/api/websearch";

const LEGALSEARCH_API_BASE =
  process.env.REACT_APP_LEGALSEARCH_API_BASE || "/api/legalsearch";

// CSRF web_search (csrftoken_websearch)
function getWebsearchCsrfToken() {
  return getCookieValue("csrftoken_websearch");
}
// --- Utilidades genéricas de cookies ---

function getCookieValue(name) {
  const decoded = decodeURIComponent(document.cookie || "");
  const parts = decoded.split("; ");
  const prefix = `${name}=`;
  for (const part of parts) {
    if (part.startsWith(prefix)) {
      return part.substring(prefix.length);
    }
  }
  return null;
}

// CSRF modelo_negocio (csrftoken_app)
function getCsrfToken() {
  return getCookieValue("csrftoken_app");
}

// CSRF chat_document (csrftoken_chatdoc)
function getChatdocCsrfToken() {
  return getCookieValue("csrftoken_chatdoc");
}

// CSRF NLP (csrftoken_nlp)
function getNlpCsrfToken() {
  return getCookieValue("csrftoken_nlp");
}

// CSRF COMPARADOR (usa mismo nombre que modelo: csrftoken_app)
function getComparerCsrfToken() {
  return getCookieValue("csrftoken_app");
}

// --- Fetch genérico para modelo_negocio ---

async function apiFetch(path, { method = "GET", headers = {}, body } = {}) {
  const url = `${BASE_API_URL}${path}`;
  const opts = {
    method,
    credentials: "include", // muy importante para SSO / cookies
    headers: {
      ...headers,
    },
  };

  if (body instanceof FormData) {
    // NO poner Content-Type, el navegador se encarga
    opts.body = body;
  } else if (body !== undefined) {
    opts.body = JSON.stringify(body);
    opts.headers["Content-Type"] = "application/json";
  }

  // CSRF solo en métodos "peligrosos"
  if (["POST", "PUT", "PATCH", "DELETE"].includes(method.toUpperCase())) {
    const csrf = getCsrfToken();
    if (csrf) {
      opts.headers["X-CSRFToken"] = csrf;
    }
  }

  const res = await fetch(url, opts);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Error ${res.status} en ${url}: ${text}`);
  }

  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    return res.json();
  }
  return res.text();
}

// --- Fetch genérico para chat_document ---

async function chatdocFetch(
  path,
  { method = "GET", headers = {}, body } = {},
) {
  const url = `${CHATDOC_API_BASE}${path}`;
  const opts = {
    method,
    credentials: "include",
    headers: {
      ...headers,
    },
  };

  if (body instanceof FormData) {
    opts.body = body;
  } else if (body !== undefined) {
    opts.body = JSON.stringify(body);
    opts.headers["Content-Type"] = "application/json";
  }

  if (["POST", "PUT", "PATCH", "DELETE"].includes(method.toUpperCase())) {
    const csrf = getChatdocCsrfToken();
    if (csrf) {
      opts.headers["X-CSRFToken"] = csrf;
    }
  }

  const res = await fetch(url, opts);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Error ${res.status} en ${url}: ${text}`);
  }

  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    return res.json();
  }
  return res.text();
}

// --- Fetch genérico para NLP / RAG ---

async function nlpFetch(
  path,
  { method = "GET", headers = {}, body } = {},
) {
  const url = `${NLP_API_BASE}${path}`;
  const opts = {
    method,
    credentials: "include", // necesario para enviar cookies de sesión / CSRF
    headers: {
      ...headers,
    },
  };

  if (body instanceof FormData) {
    // NO poner Content-Type, el navegador se encarga
    opts.body = body;
  } else if (body !== undefined) {
    opts.body = JSON.stringify(body);
    opts.headers["Content-Type"] = "application/json";
  }

  // En el servicio NLP usamos el mismo esquema CSRF (double-submit cookie).
  const csrf = getNlpCsrfToken();
  if (csrf) {
    opts.headers["X-CSRFToken"] = csrf;
  }

  const res = await fetch(url, opts);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Error ${res.status} en ${url}: ${text}`);
  }

  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    return res.json();
  }
  return res.text();
}

async function websearchFetch(
  path,
  { method = "GET", headers = {}, body } = {},
) {
  const url = `${WEBSEARCH_API_BASE}${path}`;
  const opts = {
    method,
    credentials: "include",
    headers: {
      ...headers,
    },
  };

  if (body instanceof FormData) {
    opts.body = body;
  } else if (body !== undefined) {
    opts.body = JSON.stringify(body);
    opts.headers["Content-Type"] = "application/json";
  }

  // CSRF para métodos no-idempotentes
  if (["POST", "PUT", "PATCH", "DELETE"].includes(method.toUpperCase())) {
    const csrf = getWebsearchCsrfToken();
    if (csrf) {
      opts.headers["X-CSRFToken"] = csrf;
    }
  }

  const res = await fetch(url, opts);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Error ${res.status} en ${url}: ${text}`);
  }

  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    return res.json();
  }
  return res.text();
}

async function legalsearchFetch(
  path,
  { method = "GET", headers = {}, body } = {},
) {
  const url = `${LEGALSEARCH_API_BASE}${path}`;
  const opts = {
    method,
    credentials: "include",
    headers: {
      ...headers,
    },
  };

  if (body instanceof FormData) {
    opts.body = body;
  } else if (body !== undefined) {
    opts.body = JSON.stringify(body);
    opts.headers["Content-Type"] = "application/json";
  }

  if (["POST", "PUT", "PATCH", "DELETE"].includes(method.toUpperCase())) {
    const csrf = getWebsearchCsrfToken();
    if (csrf) {
      opts.headers["X-CSRFToken"] = csrf;
    }
  }

  const res = await fetch(url, opts);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Error ${res.status} en ${url}: ${text}`);
  }

  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    return res.json();
  }
  return res.text();
}

// --- Fetch genérico para COMPARADOR ---

async function comparerFetch(
  path,
  { method = "GET", headers = {}, body } = {},
) {
  const url = `${COMPARATOR_API_BASE}${path}`;
  const opts = {
    method,
    credentials: "include",
    headers: {
      ...headers,
    },
  };

  if (body instanceof FormData) {
    // NO poner Content-Type, el navegador se encarga
    opts.body = body;
  } else if (body !== undefined) {
    opts.body = JSON.stringify(body);
    opts.headers["Content-Type"] = "application/json";
  }

  // CSRF para métodos no-idempotentes
  if (["POST", "PUT", "PATCH", "DELETE"].includes(method.toUpperCase())) {
    const csrf = getComparerCsrfToken();
    if (csrf) {
      opts.headers["X-CSRFToken"] = csrf;
    }
  }

  const res = await fetch(url, opts);
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Error ${res.status} en ${url}: ${text}`);
  }

  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    return res.json();
  }
  return res.text();
}

/* === Endpoints genéricos / helpers === */

/**
 * Inicializa el token CSRF del micro de modelo_negocio.
 *
 * GET /csrf-token (detrás de nginx: /api/modelo/csrf-token)
 * → pone cookie "csrftoken_app" y devuelve { csrf_token: "..." }.
 */
export async function fetchCsrfToken() {
  try {
    return await apiFetch("/csrf-token", { method: "GET" });
  } catch (e) {
    console.error("Error al obtener CSRF token del modelo:", e);
    throw e;
  }
}

/**
 * Inicializa el token CSRF del micro de chat_document.
 *
 * GET /csrf-token (detrás de nginx: /api/chatdoc/csrf-token)
 * → pone cookie "csrftoken_chatdoc" y devuelve { csrf_token: "..." }.
 */
export async function fetchChatdocCsrfToken() {
  try {
    return await chatdocFetch("/csrf-token", { method: "GET" });
  } catch (e) {
    console.error("Error al obtener CSRF token de chat_document:", e);
    throw e;
  }
}

/**
 * Inicializa el token CSRF del micro NLP/RAG.
 *
 * GET /csrf-token (detrás de nginx: /api/nlp/csrf-token)
 * → pone cookie "csrftoken_nlp" y devuelve { csrf_token: "..." }.
 */
export async function fetchNlpCsrfToken() {
  try {
    return await nlpFetch("/csrf-token", { method: "GET" });
  } catch (e) {
    console.error("Error al obtener CSRF token del NLP:", e);
    throw e;
  }
}

/**
 * Función de bootstrap usada por MainLayout.jsx.
 *
 * - Llama a /csrf-token para asegurarse de que la cookie CSRF está presente.
 * - Llama a /me para recuperar los datos del usuario autenticado.
 */
export async function bootstrapModelo() {
  try {
    try {
      await fetchCsrfToken();
    } catch (e) {
      console.warn(
        "bootstrapModelo: no se pudo inicializar CSRF del modelo, continuo:",
        e,
      );
    }

    const me = await fetchMe();

    return {
      ok: true,
      user: me,
    };
  } catch (error) {
    console.error("bootstrapModelo: error durante el bootstrap:", error);
    return {
      ok: false,
      user: null,
      error: error?.message || String(error),
    };
  }
}

/**
 * Bootstrap específico de chat_document.
 *
 * De momento solo se encarga de inicializar el CSRF de chat_document.
 */
export async function bootstrapChatdoc() {
  try {
    await fetchChatdocCsrfToken();
    return { ok: true };
  } catch (error) {
    console.error(
      "bootstrapChatdoc: error inicializando CSRF de chat_document:",
      error,
    );
    return {
      ok: false,
      error: error?.message || String(error),
    };
  }
}

/**
 * Bootstrap específico para el micro NLP/RAG.
 *
 * Inicializa solo su CSRF (csrftoken_nlp). El contexto de subida
 * se obtiene con `fetchNlpUploadContext`.
 */
export async function bootstrapNlp() {
  try {
    await fetchNlpCsrfToken();
    return { ok: true };
  } catch (error) {
    console.error(
      "bootstrapNlp: error inicializando CSRF del NLP:",
      error,
    );
    return {
      ok: false,
      error: error?.message || String(error),
    };
  }
}

/* === Endpoints modelo_negocio existentes === */

export async function fetchMe() {
  // GET /me
  return apiFetch("/me", { method: "GET" });
}

export async function fetchConversations() {
  // GET /conversations → [{ id, title, created_at }]
  return apiFetch("/conversations", { method: "GET" });
}

export async function deleteConversation(conversationId){
  return apiFetch(`/conversations/${conversationId}`, { method: "DELETE"});
}

export async function rateMessage(messageId, isLiked) {
  // PUT /messages/{id}/feedback
  // body: { is_liked: true } (o false/null)
  return apiFetch(`/messages/${messageId}/feedback`, { 
    method: "PUT", 
    body: { is_liked: isLiked } 
  });
}

export async function toggleFavoriteConversation(conversationId, isFavorite) {
  // PATCH /conversations/{id}/favorite
  // Body: { is_favorite: true/false }
  return apiFetch(`/conversations/${conversationId}/favorite`, {
    method: "PATCH",
    body: { is_favorite: isFavorite }
  });
}

export async function fetchConversationDetail(id) {
  // GET /conversations/{id} → { id, created_at, messages: [...] }
  return apiFetch(`/conversations/${id}`, { method: "GET" });
}

export async function sendChatMessage({
  message,
  conversationId,
  choice = "C",
  files = [],
}) {
  // POST /query/llm
  return apiFetch("/query/llm", {
    method: "POST",
    body: {
      message, // campo que el backend espera
      choice, // "C" (COSMOS) o "R" (rápido)
      conversation_id: conversationId || null,
      files, // lista de IDs efímeros (si usas /uploadfile)
    },
  });
}

export async function uploadEphemeralFiles(files) {
  // POST /uploadfile/ con multipart
  const formData = new FormData();
  for (const f of files) {
    formData.append("files", f);
  }
  return apiFetch("/uploadfile/", {
    method: "POST",
    body: formData,
  });
}

export async function resetContext() {
  // POST /context/reset
  return apiFetch("/context/reset", { method: "POST" });
}

export async function logout() {
  // POST /logout del micro modelo_negocio
  return apiFetch("/logout", { method: "POST" });
}

/* === Endpoints específicos de chat_document === */

/**
 * Sube un documento al microservicio chat_document
 * → POST /api/chatdoc/document/upload
 */
export async function uploadChatDocDocument(file) {
  const formData = new FormData();
  formData.append("file", file);
  return chatdocFetch("/document/upload", {
    method: "POST",
    body: formData,
  });
}

/**
 * Envía una consulta al microservicio chat_document
 * → POST /api/chatdoc/document/query
 */
export async function sendChatDocMessage({
  prompt,
  docSessionId,
  conversationId = null,
  mode = null,
}) {
  return chatdocFetch("/document/query", {
    method: "POST",
    body: {
      prompt,
      doc_session_id: docSessionId,
      conversation_id: conversationId,
      mode,
    },
  });
}

/* === Endpoints específicos del micro NLP/RAG === */

/**
 * Recupera el contexto de subida al NLP:
 *   { role, departments, user_directory }
 *
 * GET /api/nlp/upload_context → /upload_context en el micro NLP.
 */
export async function fetchNlpUploadContext() {
  return nlpFetch("/upload_context", { method: "GET" });
}

/**
 * Sube archivos al micro NLP, reutilizando la lógica antigua de /upload_file:
 *   - files: array de File
 *   - department: string department_directory o null (privado)
 *
 * POST /api/nlp/upload_file → /upload_file en el micro NLP.
 */
export async function uploadRagFiles({ files, department = null }) {
  const formData = new FormData();
  files.forEach((f) => formData.append("files", f));
  if (department) {
    formData.append("department", department);
  }
  return nlpFetch("/upload_file", {
    method: "POST",
    body: formData,
  });
}

/**
 * Lista archivos subidos (privados o de un departamento concreto).
 *
 * GET /api/nlp/list_files[?department=...]
 */
export async function listRagFiles({ department = null } = {}) {
  const params = new URLSearchParams();
  if (department) params.append("department", department);
  const qs = params.toString();
  const path = qs ? `/list_files?${qs}` : "/list_files";
  return nlpFetch(path, { method: "GET" });
}

/**
 * Borra archivos seleccionados.
 *
 * DELETE /api/nlp/delete_files
 * body: { filenames: [...], department: string | null }
 */
export async function deleteRagFiles({ filenames, department = null }) {
  return nlpFetch("/delete_files", {
    method: "DELETE",
    body: {
      filenames,
      department,
    },
  });
}

/**
 * Lanza el procesamiento e indexado de los ficheros del usuario
 * en su directorio privado.
 *
 * GET /api/nlp/process_user_files[?client_tag=...&scan_dir=...]
 */
export async function processUserRagFiles({ clientTag, scanDir } = {}) {
  const params = new URLSearchParams();
  if (clientTag) params.append("client_tag", clientTag);
  if (scanDir) params.append("scan_dir", scanDir);

  const qs = params.toString();
  const path = qs ? `/process_user_files?${qs}` : "/process_user_files";

  return nlpFetch(path, { method: "GET" });
}

/**
 * Lanza el procesamiento e indexado de los ficheros departamentales
 * (solo rol Supervisor).
 *
 * GET /api/nlp/process_department_files
 */
export async function processDepartmentRagFiles() {
  return nlpFetch("/process_department_files", { method: "GET" });
}

/**
 * Buscador RAG simple (equivalente al JS antiguo de /search).
 *
 * POST /api/nlp/search
 */
export async function searchRag({ query, topK = 5, topKContext = 3 }) {
  return nlpFetch("/search", {
    method: "POST",
    body: {
      query,
      top_k: topK,
      top_k_context: topKContext,
    },
  });
}

/* === Endpoints del COMPARADOR de textos (via /api/comparador) === */

/**
 * Inicializa el token CSRF del micro comparador.
 *
 * GET /api/comparador/csrf-token
 * → pone cookie "csrftoken_app" y devuelve { csrf_token: "..." }.
 */
export async function fetchComparerCsrfToken() {
  return comparerFetch("/csrf-token", { method: "GET" });
}

/**
 * Inicia un job de comparación de textos.
 * POST /api/comparador/comparar
 * FormData:
 *  - file_a: File
 *  - file_b: File
 *  - euro_mode: "strict" | "decimal" | "loose" (opcional)
 *  - min_euro: número (opcional)
 */
export async function startTextCompareJob({ fileA, fileB, options = {} }) {
  const form = new FormData();
  form.append("file_a", fileA);
  form.append("file_b", fileB);

  if (options?.euro_mode) {
    form.append("euro_mode", String(options.euro_mode));
  }
  if (options?.min_euro !== null && options?.min_euro !== undefined && options?.min_euro !== "") {
    form.append("min_euro", String(options.min_euro));
  }
  if (options?.engine) {
    form.append("engine", String(options.engine));
  }
  if (options?.soffice) {
    form.append("soffice", String(options.soffice));
  }

  return comparerFetch("/comparar", { method: "POST", body: form });
}

/**
 * Consulta el progreso del job.
 * GET /api/comparador/progress/{sid}
 */
export async function pollTextCompareProgress(sid) {
  return comparerFetch(`/progress/${encodeURIComponent(sid)}`, { method: "GET" });
}

/**
 * Abre el HTML de resultado en una nueva pestaña, servido por el backend.
 * GET visual → /api/comparador/resultado/{sid}
 */
export function openResultInNewTab(sid) {
  const url = `${COMPARATOR_API_BASE}/resultado/${encodeURIComponent(sid)}`;
  window.open(url, "_blank", "noopener,noreferrer");
}

/**
 * Descarga el informe PDF (si está disponible) de /api/comparador/descargar/{sid}/informe.pdf
 */
export async function downloadTextCompareReport(sid) {
  const url = `${COMPARATOR_API_BASE}/descargar/${encodeURIComponent(sid)}/informe.pdf`;
  const res = await fetch(url, {
    method: "GET",
    credentials: "include",
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Error ${res.status} al descargar informe: ${text}`);
  }

  const blob = await res.blob();

  // Intenta recuperar filename del Content-Disposition
  const cd = res.headers.get("content-disposition") || "";
  let filename = `informe_${sid}.pdf`;
  const m = /filename\*?=(?:UTF-8''|")?([^";\n]+)/i.exec(cd);
  if (m && m[1]) {
    try {
      filename = decodeURIComponent(m[1].replace(/"/g, ""));
    } catch {
      filename = m[1].replace(/"/g, "");
    }
  }

  const href = window.URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = href;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  window.URL.revokeObjectURL(href);
}


export async function fetchWebsearchCsrfToken() {
  try {
    return await websearchFetch("/csrf-token", { method: "GET" });
  } catch (e) {
    console.error("Error al obtener CSRF token de web_search:", e);
    throw e;
  }
}

/**
 * Bootstrap específico de web_search.
 */
export async function bootstrapWebsearch() {
  try {
    await fetchWebsearchCsrfToken();
    return { ok: true };
  } catch (error) {
    console.error("bootstrapWebsearch: error inicializando CSRF:", error);
    return { ok: false, error: error?.message || String(error) };
  }
}

/**
 * Envía una consulta al microservicio web_search.
 * → POST /api/websearch/search/query
 *
 * Devuelve:
 *  {
 *    reply, response, conversation_id, search_session_id,
 *    sources: [{ title, url, snippet, content }]
 *  }
 */
export async function sendWebSearchMessage({
  prompt,
  searchSessionId = null,
  conversationId = null,
  topK = null,
  maxIters = null,
}) {
  return websearchFetch("/search/query", {
    method: "POST",
    body: {
      prompt,
      search_session_id: searchSessionId,
      conversation_id: conversationId,
      top_k: topK,
      max_iters: maxIters,
    },
  });
}

export async function fetchLegalsearchCsrfToken() {
  try {
    return await legalsearchFetch("/csrf-token", { method: "GET" });
  } catch (e) {
    console.error("Error al obtener CSRF token de legal_search:", e);
    throw e;
  }
}

export async function bootstrapLegalsearch() {
  try {
    await fetchLegalsearchCsrfToken();
    return { ok: true };
  } catch (error) {
    console.error("bootstrapLegalsearch: error inicializando CSRF:", error);
    return { ok: false, error: error?.message || String(error) };
  }
}

export async function uploadLegalSearchFiles({
  files,
  searchSessionId,
  conversationId = null,
}) {
  const formData = new FormData();
  (files || []).forEach((f) => formData.append("files", f));

  const params = new URLSearchParams();
  if (searchSessionId) params.set("search_session_id", searchSessionId);
  if (conversationId !== null && conversationId !== undefined) {
    params.set("conversation_id", String(conversationId));
  }

  const qs = params.toString();
  const path = qs ? `/search/uploadfile?${qs}` : "/search/uploadfile";

  return legalsearchFetch(path, {
    method: "POST",
    body: formData,
  });
}

export async function sendLegalSearchMessage({
  prompt,
  searchSessionId = null,
  conversationId = null,
  attachedFileIds = [],
  topK = null,
  maxIters = null,
}) {
  return legalsearchFetch("/search/query", {
    method: "POST",
    body: {
      prompt,
      search_session_id: searchSessionId,
      conversation_id: conversationId,
      attached_file_ids: attachedFileIds,
      top_k: topK,
      max_iters: maxIters,
    },
  });
}

export async function fetchNotetakerSsoUrl() {
  const url = `/api/notetaker/sso-url`;

  const opts = {
    method: "POST",
    credentials: "include",
    headers: {},
  };

  // CSRF: el backend valida CSRF como modelo_negocio (csrftoken_app)
  const csrf = getCsrfToken();
  if (csrf) {
    opts.headers["X-CSRFToken"] = csrf;
  }

  const res = await fetch(url, opts);

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Error ${res.status} en ${url}: ${text}`);
  }

  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) {
    return res.json();
  }
  // no debería ocurrir, pero por seguridad:
  const text = await res.text();
  return { url: text };
}

/**
 * Acción directa: abre Notetaker.
 */
export async function openNotetaker() {
  const { url } = await fetchNotetakerSsoUrl();
  if (!url) throw new Error("No se recibió URL de Notetaker.");
  window.location.href = url; // redirección full page (otro origen)
}

