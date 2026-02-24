# utils.py
import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from utils import clean_text, extract_text_from_csv, extract_text_from_doc,  extract_text_from_docx, extract_text_from_pdf, extract_text_from_pptx, extract_text_from_txt, extract_text_from_xlsx
import requests  # ya lo usas en otros sitios

import logging
import requests
from typing import Optional
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

OCR_APP_URL = os.getenv("OCR_APP_URL", "http://localhost:8010")
# -------------------------------------------------------------------
# Integración avanzada con Docling para PDF/Office/HTML/MD
# -------------------------------------------------------------------
try:
    from docling.document_converter import DocumentConverter
    from docling_core.transforms.chunker import HierarchicalChunker
    _HAS_DOCLING = True
except Exception:
    DocumentConverter = None  # type: ignore
    HierarchicalChunker = None  # type: ignore
    _HAS_DOCLING = False
    logging.warning(
        "[utils] Docling no está instalado; se usarán extractores clásicos "
        "para documentos (PDF/DOCX/PPTX/HTML/MD)."
    )


@dataclass
class DocumentExtractionResult:
    """
    Resultado normalizado de la extracción de un documento.

    - text: texto concatenado del documento (ya limpio y truncado).
    - chunks: lista de trozos lógicos pensados para RAG / contexto.
    - metadata: metadatos técnicos útiles para auditoría y debug.
    """
    text: str
    chunks: List[str]
    metadata: Dict[str, Any]


class EphemeralDocumentProcessor:
    """
    Procesador avanzado para documentos efímeros:
    - Usa Docling cuando está disponible (PDF, DOCX, PPTX, HTML, MD...).
    - Recurre a tus extractores clásicos (extract_text_from_*) como fallback.
    - Aplica recortes defensivos por tamaño y chunking.
    - Deja un hook para encadenar uMiner u otros normalizadores avanzados.
    """

    def __init__(
        self,
        max_chars_per_file: int = 80_000,
        max_chars_per_chunk: int = 4_000,
        use_docling: bool = True,
    ) -> None:
        self.max_chars_per_file = max_chars_per_file
        self.max_chars_per_chunk = max_chars_per_chunk
        self.use_docling = use_docling and _HAS_DOCLING
        self._converter: Optional[DocumentConverter] = None  # type: ignore
        self._chunker: Optional[HierarchicalChunker] = None  # type: ignore

    def _ensure_docling(self) -> None:
        if not self.use_docling:
            return

        if self._converter is None:
            try:
                self._converter = DocumentConverter()
            except Exception as exc:
                logging.warning(
                    "[utils] Falló inicialización de Docling; se desactiva su uso. Detalle: %s",
                    exc,
                )
                self.use_docling = False

        if self._chunker is None and self.use_docling:
            try:
                self._chunker = HierarchicalChunker()
            except Exception as exc:
                logging.warning(
                    "[utils] Falló inicialización de HierarchicalChunker; se desactiva chunking Docling. Detalle: %s",
                    exc,
                )
                self._chunker = None

    def process_file(
        self,
        file_path: str,
        filename: str,
        mime_type: Optional[str] = None,
    ) -> DocumentExtractionResult:
        """
        Punto de entrada principal: extrae texto y metadatos de un fichero local.

        Usa Docling si está disponible para PDF/DOCX/PPTX/HTML/MD y
        recurre a los extractores clásicos de utils como fallback.
        """
        ext = (os.path.splitext(filename)[1] or "").lower()
        raw_text: str = ""

        # 1) Intento con Docling cuando tiene sentido (PDF/Office/HTML/MD).
        if self.use_docling and ext in {
            ".pdf",
            ".docx",
            ".doc",
            ".pptx",
            ".ppt",
            ".html",
            ".htm",
            ".md",
        }:
            try:
                self._ensure_docling()
                if self._converter is not None:
                    doc = self._converter.convert(file_path).document  # type: ignore[attr-defined]
                    if self._chunker is not None:
                        chunks = [
                            chunk.text for chunk in self._chunker.chunk(doc)  # type: ignore[union-attr]
                        ]
                        raw_text = "\n\n".join(chunks)
                    else:
                        # API Docling >= 0.13: render_as_markdown; fallback a export_to_markdown
                        if hasattr(doc, "render_as_markdown"):
                            raw_text = doc.render_as_markdown()  # type: ignore[call-arg]
                        elif hasattr(doc, "export_to_markdown"):
                            raw_text = doc.export_to_markdown()  # type: ignore[call-arg]
                        else:
                            raw_text = str(doc)
            except Exception as exc:
                logging.warning(
                    "[utils] Docling no pudo procesar '%s'. Se usará extractor clásico. Detalle: %s",
                    filename,
                    exc,
                )
                raw_text = ""

        # 2) Fallback: extractores clásicos de utils.py.
        if not raw_text:
            # Estas funciones ya existen en tu utils.py
            if ext == ".pdf":
                raw_text = extract_text_from_pdf(file_path)         # noqa: F821
            elif ext == ".docx":
                raw_text = extract_text_from_docx(file_path)        # noqa: F821
            elif ext == ".doc":
                raw_text = extract_text_from_doc(file_path)         # noqa: F821
            elif ext == ".xlsx":
                raw_text = extract_text_from_xlsx(file_path)        # noqa: F821
            elif ext == ".csv":
                raw_text = extract_text_from_csv(file_path)         # noqa: F821
            elif ext == ".pptx":
                raw_text = extract_text_from_pptx(file_path)        # noqa: F821
            elif ext == ".txt":
                raw_text = extract_text_from_txt(file_path)         # noqa: F821
            elif ext in {".png", ".jpg", ".jpeg"}:
                # Para imágenes se debe usar extract_text_from_image_file()
                raise ValueError(
                    "Para imágenes usa extract_text_from_image_file() en lugar de process_file()."
                )
            else:
                raise ValueError(
                    f"Extensión de archivo no soportada para extracción: '{ext}'"
                )

        text = clean_text(raw_text or "")                          # noqa: F821
        if not text:
            raise ValueError("No se pudo extraer texto significativo del documento.")

        # 3) Recorte defensivo por tamaño
        if len(text) > self.max_chars_per_file:
            text = text[: self.max_chars_per_file]

        # 4) Chunking sencillo (fallback si no tenemos chunking Docling)
        chunks: List[str] = []
        offset = 0
        while offset < len(text):
            end = min(offset + self.max_chars_per_chunk, len(text))
            slice_ = text[offset:end]
            # Intentar cortar en salto de línea para preservar párrafos
            last_nl = slice_.rfind("\n")
            if last_nl > 200:  # no cortar demasiado pronto
                end = offset + last_nl
                slice_ = text[offset:end]
            chunks.append(slice_.strip())
            offset = end

        metadata: Dict[str, Any] = {
            "filename": filename,
            "extension": ext,
            "mime_type": mime_type,
            "num_chars": len(text),
            "num_chunks": len(chunks),
            "engine": "docling"
            if self.use_docling
            and ext
            in {".pdf", ".docx", ".doc", ".pptx", ".ppt", ".html", ".htm", ".md"}
            else "classic",
        }

        # Hook de post-procesado: aquí puedes encadenar uMiner u otras librerías
        # para limpieza avanzada, detección de PII, normalización, etc.
        #
        # if self.umin_pipeline:
        #     text = self.umin_pipeline.normalize(text)
        #     chunks = [self.umin_pipeline.normalize(ch) for ch in chunks]
        #     metadata["postprocessor"] = "uMiner"

        return DocumentExtractionResult(text=text, chunks=chunks, metadata=metadata)


# Instancia global reutilizable (evita re-cargar modelos Docling en cada petición)
EPHEMERAL_DOC_PROCESSOR = EphemeralDocumentProcessor()


def process_ephemeral_file(
    file_path: str,
    filename: str,
    mime_type: Optional[str] = None,
) -> DocumentExtractionResult:
    """
    Helper de alto nivel para el endpoint /uploadfile/.
    """
    return EPHEMERAL_DOC_PROCESSOR.process_file(
        file_path=file_path,
        filename=filename,
        mime_type=mime_type,
    )


def extract_text_from_image_file(
    file_path: str,
    filename: str,
    mime_type: str = "image/png",
    timeout: int = 120,
    request_id: Optional[str] = None,
) -> str:
    """
    Extrae texto de una imagen usando el microservicio OCR (OCR_APP_URL).

    - Envía el archivo al endpoint /upload_and_extract.
    - Propaga un X-Request-Id opcional para trazabilidad.
    - Maneja errores de red y de respuesta de forma robusta.

    Devuelve:
      - Texto ya limpio (clean_text).
    Lanza:
      - HTTPException 503 si el OCR no está disponible.
      - HTTPException 502 si el OCR responde con error lógico.
    """
    headers = {}
    if request_id:
        headers["X-Request-Id"] = request_id

    try:
        with open(file_path, "rb") as imgf:
            resp = requests.post(
                f"{OCR_APP_URL}/upload_and_extract",
                files={"file": (filename, imgf, mime_type)},
                headers=headers,
                timeout=timeout,
            )
    except requests.exceptions.RequestException as exc:
        # Red / timeout / DNS / conexión
        logger.error(
            "Error de red al llamar al microservicio OCR",
            extra={
                "file": filename,
                "mime_type": mime_type,
                "error": str(exc),
                "event": "ocr_call_network_error",
                "request_id": request_id,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Servicio OCR no disponible en este momento.",
        ) from exc

    # Si el OCR responde con código HTTP de error
    if not resp.ok:
        logger.error(
            "Respuesta HTTP no exitosa desde el microservicio OCR",
            extra={
                "file": filename,
                "mime_type": mime_type,
                "status_code": resp.status_code,
                "body": resp.text[:1000],  # recorte defensivo
                "event": "ocr_call_http_error",
                "request_id": request_id,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Error al procesar la imagen en el servicio OCR.",
        )

    try:
        data = resp.json()
    except ValueError as exc:
        logger.error(
            "La respuesta del OCR no es JSON válido",
            extra={
                "file": filename,
                "mime_type": mime_type,
                "body": resp.text[:1000],
                "event": "ocr_invalid_json",
                "request_id": request_id,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Respuesta no válida del servicio OCR.",
        ) from exc

    # El micro OCR devuelve "status": "ok" cuando todo va bien
    if data.get("status") != "ok":
        logger.error(
            "El microservicio OCR devolvió un estado de error lógico",
            extra={
                "file": filename,
                "mime_type": mime_type,
                "ocr_status": data.get("status"),
                "meta": data.get("meta", {}),
                "event": "ocr_logical_error",
                "request_id": request_id,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="El servicio OCR no pudo extraer texto de la imagen.",
        )

    text_raw = data.get("extracted_text", "") or ""
    text_clean = clean_text(text_raw)

    logger.info(
        "Texto extraído correctamente desde el microservicio OCR",
        extra={
            "file": filename,
            "mime_type": mime_type,
            "length": len(text_clean),
            "event": "ocr_call_success",
            "request_id": request_id,
        },
    )

    return text_clean

