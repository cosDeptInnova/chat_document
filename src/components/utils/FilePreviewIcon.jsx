import React, { useEffect, useMemo, useState } from "react";
import getFileIconClass from "./GetFileIcon";

const IMAGE_EXTS = ["jpg", "jpeg", "png", "gif", "webp"];

export default function FilePreviewIcon({ file, onPreview }) {
  const extension = useMemo(
    () => (file?.name || "").split(".").pop().toLowerCase(),
    [file]
  );

  const isImage = useMemo(() => {
    return (
      IMAGE_EXTS.includes(extension) ||
      (file && typeof file.type === "string" && file.type.startsWith("image/"))
    );
  }, [extension, file]);

  // URL solo para miniatura y preview
  const [thumbUrl, setThumbUrl] = useState(null);

  useEffect(() => {
    if (!isImage || !file) {
      setThumbUrl(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setThumbUrl(url);
    return () => {
      URL.revokeObjectURL(url);
    };
  }, [isImage, file]);

  if (isImage && thumbUrl) {
    return (
      <button
        type="button"
        className="cursor-pointer focus:outline-none flex items-center justify-center w-20 h-20 bg-gray-200 rounded overflow-hidden"
        onClick={() => onPreview(thumbUrl)} // Le pasamos la url al modal
        title="Ver vista previa"
      >
        <img
          src={thumbUrl}
          alt={file.name}
          className="w-full h-full object-cover"
        />
      </button>
    );
  }

  // Archivos no imagen → icono
  const iconClass = getFileIconClass(extension);
  return (
    <div
      className="flex items-center justify-center w-20 h-20 bg-gray-200 rounded cursor-pointer"
      onClick={() => onPreview(null)} // Al ser un archivo que no se abre, ponemos null para que no lo haga
      title={file?.name}
    >
      <i className={`fas ${iconClass} text-3xl`} aria-hidden="true" />
    </div>
  );
}