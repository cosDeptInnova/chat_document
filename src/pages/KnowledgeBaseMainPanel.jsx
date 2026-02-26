import React, { useState, useEffect } from "react";
import { Search, Folder, Users, Upload } from 'lucide-react';
// Importamos los componentes
import { RagFileList } from '../components/RagFileList';
import { RagUploadModal } from '../components/RagUploadModal';

// Importamos TU librería de API original
import {
  bootstrapNlp,
  fetchNlpUploadContext,
  uploadRagFiles,
  processUserRagFiles,
  processDepartmentRagFiles,
  listRagFiles,
  deleteRagFiles,
  downloadRagFile
} from "../lib/api";

//FUNCIONES DE FORMATEO PARA LA FECHA Y EL TAMAÑO  ===
const formatBytes = (bytes) => {
    if (bytes === 0) return '0 KB';
    if (!bytes || isNaN(bytes)) return '---';

    const k = 1024;
    const decimals = 2; // Número de decimales

    // 1. Si es mayor o igual a 1 GB
    if (bytes >= k * k * k) {
        return `${(bytes / (k * k * k)).toFixed(decimals)} GB`;
    }
    
    // 2. Si es mayor o igual a 1 MB
    else if (bytes >= k * k) {
        return `${(bytes / (k * k)).toFixed(decimals)} MB`;
    }
    
    // 3. Para todo lo demás (menor a 1 MB), mostramos KB
    else {
        return `${(bytes / k).toFixed(decimals)} KB`;
    }
};

const formatDate = (dateString) => {
    if (!dateString) return '--/--/----';
    try {
        return new Date(dateString).toLocaleDateString('es-ES', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit',
            hour: '2-digit',
            minute: '2-digit'
        });
    } catch (e) {
        return dateString;
    }
};

export default function BaseConocimientosMainPanel({ isDarkMode }) {
  // === ESTADOS DE UI ===
  const [activeTab, setActiveTab] = useState('personal'); 
  const [searchQuery, setSearchQuery] = useState('');
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [sortConfig, setSortConfig] = useState({ key: 'name', direction: 'asc' });
  
  // === ESTADOS DE DATOS ===
  const [role, setRole] = useState(null);
  const [departments, setDepartments] = useState([]);
  const [selectedDepartment, setSelectedDepartment] = useState("");
  
  const [existingFiles, setExistingFiles] = useState([]); 
  const [existingFilesLoading, setExistingFilesLoading] = useState(false);
  const [selectedExistingFiles, setSelectedExistingFiles] = useState(new Set());

  // === 1. INICIALIZACIÓN ===
  useEffect(() => {
    async function init() {
      try { await bootstrapNlp(); } catch (e) { console.warn("CSRF:", e); }

      try {
        const ctx = await fetchNlpUploadContext();
        setRole(ctx.role || null);
        
        const userDepts = ctx.departments || [];
        setDepartments(userDepts);
        
        if (userDepts.length > 0) {
          setSelectedDepartment(userDepts[0].department_directory);
        }
      } catch (e) {
        console.error("Error cargando contexto:", e);
      }
    }
    init();
  }, []);

  useEffect(() => {
    loadExistingFiles();
  }, [activeTab, selectedDepartment]);

  // === 2. FUNCIONES DE CARGA ===
  const loadExistingFiles = async () => {
    setExistingFilesLoading(true);
    try {
      if (activeTab === 'department' && departments.length === 0) {
          setExistingFiles([]);
          setExistingFilesLoading(false);
          return;
      }

      const departmentValue = (activeTab === 'department' && role === "Supervisor") 
        ? selectedDepartment 
        : null; 
      
      const res = await listRagFiles({
        department: departmentValue,
      });

      const files = Array.isArray(res?.files) ? res.files : [];
      setExistingFiles(files);
      setSelectedExistingFiles(new Set());
    } catch (e) {
      console.error("Error listando archivos:", e);
      setExistingFiles([]);
    } finally {
      setExistingFilesLoading(false);
    }
  };

  // === 3. FUNCIÓN DE SUBIDA ===
  const handleUpload = async (filesToUpload, destination, labels) => {
    try {
      let departmentValue = null;
      if (destination === 'department') {
         if (role === "Supervisor") {
             departmentValue = selectedDepartment;
         } else {
             departmentValue = null; 
         }
      }

      await uploadRagFiles({ files: filesToUpload, department: departmentValue });

      if (destination === 'personal') {
        await processUserRagFiles({ clientTag: labels });
      } else {
        await processDepartmentRagFiles();
      }

      alert("Archivos subidos e indexados correctamente.");
      loadExistingFiles();
      setIsUploadModalOpen(false);
      
    } catch (err) {
      console.error("Error subida:", err);
      alert(`Error: ${err.message || "Fallo en la subida"}`);
    }
  };

  // === 4. FUNCIONES DE BORRADO ===
  const handleDelete = async (filename) => {
    if (!window.confirm(`¿Borrar "${filename}"?`)) return;
    try {
      setExistingFilesLoading(true);
      const departmentValue = (activeTab === 'department' && role === "Supervisor") ? selectedDepartment : null;
      const result = await deleteRagFiles({ filenames: [filename], department: departmentValue });
      await loadExistingFiles();

      const issues = [];
      if (Array.isArray(result?.not_found) && result.not_found.length > 0) {
        issues.push(`No encontrados: ${result.not_found.join(", ")}`);
      }
      if (Array.isArray(result?.errors) && result.errors.length > 0) {
        issues.push(`Errores: ${result.errors.map(item => item.file).join(", ")}`);
      }
      if (issues.length > 0) {
        alert(`Borrado parcial. ${issues.join(" | ")}`);
      }
    } catch (e) {
      alert("Error al borrar: " + e.message);
      setExistingFilesLoading(false);
    }
  };

  const handleBulkDelete = async () => {
    const filenames = Array.from(selectedExistingFiles);
    if (filenames.length === 0) return;
    if (!window.confirm(`¿Seguro que quieres borrar ${filenames.length} archivos?`)) return;

    try {
      setExistingFilesLoading(true);
      const departmentValue = (activeTab === 'department' && role === "Supervisor") ? selectedDepartment : null;
      const result = await deleteRagFiles({ filenames: filenames, department: departmentValue });
      await loadExistingFiles();

      const issues = [];
      if (Array.isArray(result?.not_found) && result.not_found.length > 0) {
        issues.push(`No encontrados: ${result.not_found.join(", ")}`);
      }
      if (Array.isArray(result?.errors) && result.errors.length > 0) {
        issues.push(`Errores: ${result.errors.map(item => item.file).join(", ")}`);
      }
      if (issues.length > 0) {
        alert(`Borrado parcial. ${issues.join(" | ")}`);
      }
    } catch (e) {
      alert("Error borrando: " + e.message);
      setExistingFilesLoading(false);
    }
  };

  const handleSelectFile = (id) => {
    setSelectedExistingFiles(prev => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  // === 5. PREPARACIÓN DE DATOS ===
  const filteredFilesRaw = existingFiles.filter(item => {
    // Soportamos si item es string (viejo) o objeto (nuevo backend)
    const name = typeof item === 'string' ? item : item.name;
    return name.toLowerCase().includes(searchQuery.toLowerCase());
  });

  const sortedFilesRaw = [...filteredFilesRaw].sort((a, b) => {
    const nameA = typeof a === 'string' ? a : a.name;
    const nameB = typeof b === 'string' ? b : b.name;
    const res = nameA.localeCompare(nameB);
    return sortConfig.direction === 'asc' ? res : -res;
  });

  // Convertimos los datos del backend a lo que la tabla espera
  const filesForTable = sortedFilesRaw.map(item => {
    // Si el backend aun devuelve strings planos (legacy)
    if (typeof item === 'string') {
        return {
            name: item,
            date: '--/--/----', 
            size: '---', 
            type: 'application/' + item.split('.').pop(),
            owner: activeTab === 'personal' ? 'Yo' : (selectedDepartment || 'Departamento'), 
            id: item
        };
    }
    
    // Si el backend devuelve objetos completos
    return {
        name: item.name,
        // Usamos las funciones formateadoras de arriba
        date: formatDate(item.date || item.created_at), 
        size: formatBytes(item.size), 
        type: 'application/' + item.name.split('.').pop(),
        owner: item.owner || (activeTab === 'personal' ? 'Yo' : (selectedDepartment || 'Departamento')), 
        id: item.name
    };
  });

  // === FUNCIÓN DE DESCARGA ===
  const handleDownload = async (filename) => {
    try {
      const departmentValue = (activeTab === 'department' && role === "Supervisor") 
        ? selectedDepartment 
        : null;

      await downloadRagFile({ 
        filename: filename, 
        department: departmentValue 
      });
      
      // Mostrar notificación de éxito
      alert("Descarga iniciada"); 
    } catch (e) {
      console.error("Error en descarga:", e);
      alert("Error al descargar el archivo: " + e.message);
    }
  };

  // === 6. RENDERIZADO ===
  return (
    <div className={`w-full h-screen flex flex-col p-4 md:p-8 font-sans overflow-hidden ${isDarkMode ? "bg-gray-900" : "bg-white"}`}>
      
      {/* Header Fijo */}
      <div className="flex-none mb-6 flex flex-col md:flex-row md:items-center md:justify-between gap-4 max-w-6xl mx-auto w-full">
        <div>
          <h1 className={`text-[28px] font-bold tracking-tight ${isDarkMode ? "text-blue-400" : "text-blue-700"}`}>
            Base de Conocimientos
          </h1>
          <p className={`mt-2 text-[15px] font-medium ${isDarkMode ? "text-gray-300" : "text-gray-600"}`}>
            Gestiona los documentos indexados en Qdrant
          </p>
        </div>
        <button
          onClick={() => setIsUploadModalOpen(true)}
          className="flex items-center justify-center px-5 py-2.5 bg-[#2563eb] hover:bg-blue-700 text-white rounded-2xl font-semibold shadow-md hover:scale-[1.02]"
        >
          <Upload className="w-4 h-4 mr-2" />
          Subir Archivos
        </button>
      </div>

      {/* CARD PRINCIPAL */}
      <div className={`flex-1 flex flex-col min-h-0 mb-12 rounded-[32px] shadow-xl border overflow-hidden max-w-6xl mx-auto w-full ${isDarkMode ? 'bg-gray-800 border-gray-700 shadow-none' : 'bg-white border-slate-100'}`}>
        
        {/* Toolbar Superior */}
        <div className={`flex-none p-6 flex flex-col xl:flex-row gap-6 justify-between items-center border-b transition-colors duration-300 ${isDarkMode ? 'border-gray-700' : 'border-slate-100'}`}>
          <div className={`flex p-1.5 rounded-2xl self-start md:self-auto w-full md:w-auto transition-colors duration-300 ${isDarkMode ? 'bg-gray-700' : 'bg-slate-100/80'}`}>
            <button
              onClick={() => setActiveTab('personal')}
              className={`flex-1 flex items-center px-6 py-2.5 rounded-xl text-sm font-semibold transition-all min-w-0 ${
                activeTab === 'personal'
                  ? (isDarkMode ? 'bg-gray-600 text-white shadow-sm' : 'bg-white text-blue-600 shadow-sm')
                  : (isDarkMode ? 'text-gray-300 hover:text-white' : 'text-slate-500 hover:text-slate-700')
              }`}
            >
              <Folder className="w-4 h-4 mr-2 shrink-0" /> 
              <span className="block truncate md:overflow-visible md:whitespace-normal md:text-clip">Personal</span>
            </button>
            <button
              onClick={() => setActiveTab('department')}
              className={`flex-1 flex items-center px-6 py-2.5 rounded-xl text-sm font-semibold transition-all min-w-0 ${
                activeTab === 'department'
                  ? (isDarkMode ? 'bg-gray-600 text-white shadow-sm' : 'bg-white text-blue-600 shadow-sm')
                  : (isDarkMode ? 'text-gray-300 hover:text-white' : 'text-slate-500 hover:text-slate-700')
              }`}
            >
              <Users className="w-4 h-4 mr-2 shrink-0" /> 
              <span className="block truncate md:overflow-visible md:whitespace-normal md:text-clip">Departamento</span>
            </button>
          </div>

          {/* Selector de departamento */}
          {activeTab === 'department' && role === 'Supervisor' && departments.length > 0 && (
             <div className="w-full md:w-64">
                <select 
                    value={selectedDepartment} 
                    onChange={(e) => setSelectedDepartment(e.target.value)}
                    className={`w-full p-2.5 rounded-xl text-sm border font-medium outline-none focus:ring-2 focus:ring-blue-500/20 transition-colors duration-300 ${
                        isDarkMode 
                        ? 'bg-gray-700 border-gray-600 text-white' 
                        : 'bg-slate-50 border-slate-200 text-slate-700'
                    }`}
                >
                    {departments.map(d => (
                        <option key={d.department_directory} value={d.department_directory}>
                            📂 {d.department_directory}
                        </option>
                    ))}
                </select>
             </div>
          )}

          {/* Buscador */}
          <div className="relative w-full md:w-80 group">
            <Search className={`absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 ${isDarkMode ? 'text-gray-400' : 'text-slate-400'}`} />
            <input
              type="text"
              placeholder="Filtrar archivos..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className={`w-full pl-12 pr-4 py-3 border-none rounded-2xl focus:outline-none focus:ring-2 focus:ring-blue-500/20 ${
                  isDarkMode 
                  ? 'bg-gray-700 text-white placeholder:text-gray-400' 
                  : 'bg-slate-50 text-slate-700 placeholder:text-slate-400'
              }`}
            />
          </div>
        </div>

        {/* ZONA DE CONTENIDO */}
        <div className={`flex-1 overflow-auto scrollbar-hide relative ${isDarkMode ? 'bg-gray-800' : 'bg-white/50'}`}>
            {activeTab === 'department' && departments.length === 0 ? (
                <div className={`h-full flex flex-col items-center justify-center p-8 ${isDarkMode ? 'text-gray-500' : 'text-slate-400'}`}>
                    <Users className="w-16 h-16 mb-4 opacity-20" />
                    <p className="font-medium text-lg text-center">No perteneces a ningún departamento</p>
                    <p className="text-sm mt-2 text-center max-w-md">Contacta con un administrador si crees que es un error.</p>
                </div>
            ) : (
                <div className="min-w-[800px] h-full">
                    <RagFileList 
                        files={filesForTable} 
                        loading={existingFilesLoading}
                        selectedFiles={selectedExistingFiles}
                        onSelectFile={handleSelectFile}
                        onDelete={handleDelete}
                        onDownload={handleDownload}
                        onShare={(name) => alert(`Compartiendo ${name}... (Pendiente)`)}
                        context={activeTab}
                        sortConfig={sortConfig}
                        onSort={(key) => setSortConfig(prev => ({ 
                            key, 
                            direction: prev.key === key && prev.direction === 'asc' ? 'desc' : 'asc' 
                        }))}
                        isDarkMode={isDarkMode}
                    />
                </div>
            )}
        </div>

        {/* Footer Fijo */}
        <div className={`flex-none border-t p-4 flex flex-col md:flex-row justify-between items-center gap-4 text-sm h-auto md:h-20 ${isDarkMode ? 'border-gray-700 bg-gray-800' : 'border-slate-100 bg-slate-50/50'}`}>
            {selectedExistingFiles.size > 0 ? (
                <div className="flex flex-wrap gap-3 animate-in fade-in slide-in-from-bottom-2 justify-center md:justify-start">
                    <span className={`font-bold flex items-center mr-2 ${isDarkMode ? 'text-white' : 'text-slate-700'}`}>
                        {selectedExistingFiles.size} seleccionados
                    </span>
                    <button onClick={handleBulkDelete} className={`px-4 py-2 rounded-xl font-bold border ${isDarkMode ? 'bg-red-900/30 text-red-400 border-red-800 hover:bg-red-900/50' : 'bg-red-50 text-red-600 border-red-200 hover:bg-red-100'}`}>
                        Borrar seleccionados
                    </button>
                    {activeTab === 'personal' && (
                        <button onClick={() => alert("Compartir masivo pendiente")} className={`px-4 py-2 rounded-xl font-bold border ${isDarkMode ? 'bg-blue-900/30 text-blue-400 border-blue-800 hover:bg-blue-900/50' : 'bg-indigo-50 text-indigo-600 border-indigo-200 hover:bg-indigo-100'}`}>
                            Compartir
                        </button>
                    )}
                </div>
            ) : (
                <span className={`italic text-center md:text-left w-full md:w-auto ${isDarkMode ? 'text-gray-500' : 'text-slate-500'}`}>
                    {activeTab === 'department' && departments.length === 0 
                        ? 'Sin acceso a departamentos' 
                        : `Mostrando ${filesForTable.length} archivos`
                    }
                </span>
            )}
        </div>
      </div>

      {/* Modal de Subida */}
      <RagUploadModal 
        isOpen={isUploadModalOpen} 
        onClose={() => setIsUploadModalOpen(false)} 
        onUpload={handleUpload}
        initialDestination={activeTab}
        availableDepartments={departments}
        isDarkMode={isDarkMode}
      />
    </div>
  );
}