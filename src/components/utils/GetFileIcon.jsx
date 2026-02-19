import React from 'react';

export default function getFileIconClass(extension) {
  switch (extension) {
    case "pdf":
      return "fa-file-pdf text-red-500";
    case "doc":
    case "docx":
      return "fa-file-word text-blue-500";
    case "xls":
    case "xlsx":
      return "fa-file-excel text-green-500";
    case "ppt":
    case "pptx":
      return "fa-file-powerpoint text-orange-500";
    case "txt":
      return "fa-file-alt text-gray-400";
    default:
      return "fa-file text-gray-400";
  }
}