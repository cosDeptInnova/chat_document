import { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { securityApi, type SSLCertificateSummary } from '../services/api';
import { XMarkIcon, CheckCircleIcon } from '@heroicons/react/24/outline';

interface ImportProgress {
  step: string;
  status: 'pending' | 'loading' | 'completed' | 'error';
}

export const AdminSecurity: React.FC = () => {
  useAuth();
  const [certificateSummary, setCertificateSummary] = useState<SSLCertificateSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const [showImportModal, setShowImportModal] = useState(false);
  
  // Estados para el formulario de importación
  const [certificateFile, setCertificateFile] = useState<File | null>(null);
  const [privateKeyFile, setPrivateKeyFile] = useState<File | null>(null);
  const [keystorePassword, setKeystorePassword] = useState('');
  const [intermediateMode, setIntermediateMode] = useState<'manual' | 'automatic'>('automatic');
  const [intermediateFiles, setIntermediateFiles] = useState<File[]>([]);
  const [importing, setImporting] = useState(false);
  
  // Estados para mostrar campos dinámicamente
  const [showPrivateKey, setShowPrivateKey] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [showIntermediate, setShowIntermediate] = useState(false);
  const [analyzingCertificate, setAnalyzingCertificate] = useState(false);
  
  // Estados para el popup de progreso
  const [showProgressModal, setShowProgressModal] = useState(false);
  const [progressSteps, setProgressSteps] = useState<ImportProgress[]>([
    { step: 'Cargando archivos', status: 'pending' },
    { step: 'Prevalidación de archivos cargados', status: 'pending' },
    { step: 'Extrayendo datos de archivos cargados', status: 'pending' },
    { step: 'Compilando cadena de certificados', status: 'pending' },
    { step: 'Verificando cadena de certificados', status: 'pending' },
    { step: 'Guardando cambios y limpiando', status: 'pending' },
  ]);

  // Cargar resumen del certificado
  useEffect(() => {
    loadCertificateSummary();
  }, []);

  // Resetear campos cuando se abre el modal
  useEffect(() => {
    if (showImportModal) {
      setCertificateFile(null);
      setPrivateKeyFile(null);
      setKeystorePassword('');
      setIntermediateFiles([]);
      setShowPrivateKey(false);
      setShowPassword(false);
      setShowIntermediate(false);
      setError(null);
      setSuccess(null);
    }
  }, [showImportModal]);

  const loadCertificateSummary = async () => {
    try {
      setLoading(true);
      setError(null);
      const summary = await securityApi.getSSLCertificate();
      setCertificateSummary(summary);
    } catch (err: any) {
      if (err.response?.status === 404) {
        setCertificateSummary(null);
      } else {
        setError(err.response?.data?.detail || err.message || 'Error al cargar certificado');
      }
    } finally {
      setLoading(false);
    }
  };

  const analyzeCertificate = async (file: File) => {
    setAnalyzingCertificate(true);
    setError(null);
    
    try {
      const fileExt = file.name.toLowerCase().split('.').pop() || '';
      
      // Determinar si requiere clave privada
      const requiresPrivateKey = ['crt', 'cer', 'pem'].includes(fileExt);
      
      if (requiresPrivateKey) {
        setShowPrivateKey(true);
        // Por defecto, asumimos que puede requerir contraseña
        // Esto se determinará cuando se cargue la clave privada
      } else {
        // Para .pfx, .keystore, .jks, siempre requieren contraseña
        if (['pfx', 'keystore', 'jks'].includes(fileExt)) {
          setShowPrivateKey(false);
          setShowPassword(true);
        }
      }
      
      // Mostrar sección de certificados intermedios después de analizar
      setShowIntermediate(true);
      
    } catch (err) {
      console.error('Error analizando certificado:', err);
      setError('Error al analizar el certificado');
    } finally {
      setAnalyzingCertificate(false);
    }
  };

  const analyzePrivateKey = async (file: File) => {
    // Intentar leer el archivo para determinar si está encriptado
    try {
      const text = await file.text();
      // Si contiene "ENCRYPTED" o "Proc-Type: 4,ENCRYPTED", requiere contraseña
      const isEncrypted = text.includes('ENCRYPTED') || text.includes('Proc-Type: 4,ENCRYPTED');
      setShowPassword(isEncrypted);
    } catch (err) {
      // Si no se puede leer como texto, probablemente es binario y puede requerir contraseña
      // Por seguridad, mostramos el campo de contraseña
      setShowPassword(true);
    }
  };

  const handleCertificateSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setCertificateFile(file);
      await analyzeCertificate(file);
    }
  };

  const handlePrivateKeySelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setPrivateKeyFile(file);
      await analyzePrivateKey(file);
    }
  };

  const updateProgressStep = (stepIndex: number, status: ImportProgress['status']) => {
    setProgressSteps(prev => {
      const newSteps = [...prev];
      newSteps[stepIndex] = { ...newSteps[stepIndex], status };
      return newSteps;
    });
  };

  const handleImport = async () => {
    if (!certificateFile) {
      setError('Debe seleccionar un archivo de certificado');
      return;
    }

    // Validaciones
    const certExt = certificateFile.name.toLowerCase().split('.').pop();
    if (['crt', 'cer', 'pem'].includes(certExt || '') && !privateKeyFile) {
      setError('Se requiere la clave privada para certificados .crt/.cer/.pem');
      return;
    }

    if (showPassword && !keystorePassword) {
      setError('Se requiere la contraseña del keystore');
      return;
    }

    setImporting(true);
    setError(null);
    setSuccess(null);
    setShowProgressModal(true);
    
    // Resetear progreso
    setProgressSteps(prev => prev.map(step => ({ ...step, status: 'pending' as const })));

    try {
      // Paso 1: Cargando archivos
      updateProgressStep(0, 'loading');
      await new Promise(resolve => setTimeout(resolve, 500));
      updateProgressStep(0, 'completed');

      // Paso 2: Prevalidación
      updateProgressStep(1, 'loading');
      await new Promise(resolve => setTimeout(resolve, 600));
      updateProgressStep(1, 'completed');

      // Paso 3: Extrayendo datos
      updateProgressStep(2, 'loading');
      await new Promise(resolve => setTimeout(resolve, 700));
      updateProgressStep(2, 'completed');

      // Paso 4: Compilando cadena (solo si es automático)
      if (intermediateMode === 'automatic') {
        updateProgressStep(3, 'loading');
        await new Promise(resolve => setTimeout(resolve, 1000));
        updateProgressStep(3, 'completed');
      } else {
        updateProgressStep(3, 'completed');
      }

      // Paso 5: Verificando cadena
      updateProgressStep(4, 'loading');
      await new Promise(resolve => setTimeout(resolve, 800));
      updateProgressStep(4, 'completed');

      // Realizar la importación real
      const result = await securityApi.importSSLCertificate(
        certificateFile,
        privateKeyFile || undefined,
        keystorePassword || undefined,
        intermediateMode,
        intermediateMode === 'manual' && intermediateFiles.length > 0 ? intermediateFiles : undefined
      );

      // Paso 6: Guardando cambios
      updateProgressStep(5, 'loading');
      await new Promise(resolve => setTimeout(resolve, 500));
      updateProgressStep(5, 'completed');

      // Mostrar mensaje de éxito
      setSuccess(result.message || 'Certificado importado exitosamente');
      
      // Esperar 3 segundos y cerrar
      setTimeout(() => {
        setShowProgressModal(false);
        setShowImportModal(false);
        setImporting(false);
        
        // Limpiar formulario
        setCertificateFile(null);
        setPrivateKeyFile(null);
        setKeystorePassword('');
        setIntermediateFiles([]);
        setShowPrivateKey(false);
        setShowPassword(false);
        setShowIntermediate(false);
        
        // Recargar resumen
        loadCertificateSummary();
      }, 3000);

    } catch (err: any) {
      // Marcar todos los pasos como error
      setProgressSteps(prev => prev.map(step => ({ ...step, status: 'error' as const })));
      setError(err.response?.data?.detail || err.message || 'Error al importar certificado');
      setImporting(false);
    }
  };

  const formatDate = (dateString: string) => {
    try {
      const date = new Date(dateString);
      return date.toLocaleString('es-ES', {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return dateString;
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-slate-50">Ajustes de Seguridad</h1>
          <p className="mt-2 text-gray-600 dark:text-slate-400">Gestiona certificados SSL y configuración de seguridad</p>
        </div>
      </div>

      {error && (
        <div className="bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <p className="text-sm text-red-800 dark:text-red-400">
            <strong>Error:</strong> {error}
          </p>
        </div>
      )}

      {success && (
        <div className="bg-green-50 dark:bg-green-900/30 border border-green-200 dark:border-green-800 rounded-lg p-4">
          <p className="text-sm text-green-800 dark:text-green-400">
            <strong>Éxito:</strong> {success}
          </p>
        </div>
      )}

      {/* Resumen de Certificado SSL */}
      <div className="bg-white dark:bg-slate-800 rounded-lg shadow transition-colors">
        <div className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold text-gray-900 dark:text-slate-50">
              Resumen de Certificado SSL
            </h2>
            <button
              onClick={() => setShowImportModal(true)}
              className="px-4 py-2 bg-primary-600 text-white rounded-lg font-semibold hover:bg-primary-700 transition-colors"
            >
              Modificar Certificado SSL
            </button>
          </div>

          {certificateSummary ? (
            <div className="space-y-6">
              {certificateSummary.imported_by && (
                <div className="text-sm text-gray-600 dark:text-slate-400">
                  <p>
                    <strong>Última importación por:</strong> {certificateSummary.imported_by}
                  </p>
                  {certificateSummary.imported_at && (
                    <p>
                      <strong>En:</strong> {formatDate(certificateSummary.imported_at)}
                    </p>
                  )}
                </div>
              )}

              <div>
                <h3 className="text-lg font-medium text-gray-900 dark:text-slate-50 mb-3">
                  Certificado Principal
                </h3>
                <div className="bg-gray-50 dark:bg-slate-700 rounded-lg p-4 space-y-2">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <p className="text-sm font-medium text-gray-700 dark:text-slate-300">Nombre del asunto:</p>
                      <p className="text-sm text-gray-900 dark:text-slate-50">{certificateSummary.certificate.subject_name}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-gray-700 dark:text-slate-300">Nombre alternativo del sujeto:</p>
                      <p className="text-sm text-gray-900 dark:text-slate-50">
                        {certificateSummary.certificate.subject_alternative_names.join(', ') || 'N/A'}
                      </p>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-gray-700 dark:text-slate-300">Válido desde:</p>
                      <p className="text-sm text-gray-900 dark:text-slate-50">{formatDate(certificateSummary.certificate.valid_from)}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-gray-700 dark:text-slate-300">Válido hasta:</p>
                      <p className="text-sm text-gray-900 dark:text-slate-50">{formatDate(certificateSummary.certificate.valid_to)}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-gray-700 dark:text-slate-300">Algoritmo de firmas:</p>
                      <p className="text-sm text-gray-900 dark:text-slate-50">{certificateSummary.certificate.signature_algorithm}</p>
                    </div>
                    <div>
                      <p className="text-sm font-medium text-gray-700 dark:text-slate-300">Algoritmo de claves:</p>
                      <p className="text-sm text-gray-900 dark:text-slate-50">{certificateSummary.certificate.key_algorithm}</p>
                    </div>
                    {certificateSummary.certificate.key_size && (
                      <div>
                        <p className="text-sm font-medium text-gray-700 dark:text-slate-300">Tamaño de clave:</p>
                        <p className="text-sm text-gray-900 dark:text-slate-50">{certificateSummary.certificate.key_size} Bits</p>
                      </div>
                    )}
                    <div>
                      <p className="text-sm font-medium text-gray-700 dark:text-slate-300">Nombre del emisor:</p>
                      <p className="text-sm text-gray-900 dark:text-slate-50">{certificateSummary.certificate.issuer_name}</p>
                    </div>
                  </div>
                </div>
              </div>

              {certificateSummary.chain && certificateSummary.chain.length > 0 && (
                <div className="space-y-4">
                  {certificateSummary.chain.map((cert, index) => (
                    <div key={index}>
                      <h3 className="text-lg font-medium text-gray-900 dark:text-slate-50 mb-3">
                        Certificado {index === 0 ? 'Intermedio' : 'Raíz'} ({cert.issuer_name})
                      </h3>
                      <div className="bg-gray-50 dark:bg-slate-700 rounded-lg p-4 space-y-2">
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <p className="text-sm font-medium text-gray-700 dark:text-slate-300">Nombre del asunto:</p>
                            <p className="text-sm text-gray-900 dark:text-slate-50">{cert.subject_name}</p>
                          </div>
                          <div>
                            <p className="text-sm font-medium text-gray-700 dark:text-slate-300">Válido hasta:</p>
                            <p className="text-sm text-gray-900 dark:text-slate-50">{formatDate(cert.valid_to)}</p>
                          </div>
                          <div>
                            <p className="text-sm font-medium text-gray-700 dark:text-slate-300">Nombre del emisor:</p>
                            <p className="text-sm text-gray-900 dark:text-slate-50">{cert.issuer_name}</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          ) : (
            <div className="text-center py-8">
              <p className="text-gray-600 dark:text-slate-400 mb-4">
                No hay certificado SSL configurado actualmente.
              </p>
              <button
                onClick={() => setShowImportModal(true)}
                className="px-4 py-2 bg-primary-600 text-white rounded-lg font-semibold hover:bg-primary-700 transition-colors"
              >
                Importar Certificado SSL
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Modal de Importación */}
      {showImportModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white dark:bg-slate-800 rounded-lg shadow-xl max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto transition-colors">
            <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-slate-700">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-slate-50">
                Importación de Certificados SSL
              </h2>
              <button
                onClick={() => !importing && setShowImportModal(false)}
                disabled={importing}
                className="text-gray-400 hover:text-gray-500 dark:hover:text-slate-300 disabled:opacity-50"
              >
                <XMarkIcon className="h-6 w-6" />
              </button>
            </div>

            <div className="p-6 space-y-6">
              {/* Certificado SSL - Siempre visible */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
                  Certificado SSL <span className="text-red-500">*</span>
                </label>
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={certificateFile?.name || '.cer / .crt / .p7b / .pfx / .keystore / .jks'}
                    readOnly
                    className="flex-1 px-4 py-2 border border-gray-300 dark:border-slate-600 rounded-lg bg-gray-50 dark:bg-slate-700 text-gray-500 dark:text-slate-400"
                  />
                  <label className="px-4 py-2 bg-gray-100 dark:bg-slate-700 text-gray-700 dark:text-slate-300 rounded-lg hover:bg-gray-200 dark:hover:bg-slate-600 cursor-pointer">
                    Examinar
                    <input
                      type="file"
                      accept=".cer,.crt,.p7b,.pfx,.keystore,.jks,.pem"
                      onChange={handleCertificateSelect}
                      disabled={importing || analyzingCertificate}
                      className="hidden"
                    />
                  </label>
                </div>
                {analyzingCertificate && (
                  <p className="mt-2 text-sm text-blue-600 dark:text-blue-400">Analizando certificado...</p>
                )}
              </div>

              {/* Keystore/Clave Privada - Solo si se requiere */}
              {showPrivateKey && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
                    Keystore/clave privada original <span className="text-red-500">*</span>
                  </label>
                  <div className="flex gap-2">
                    <input
                      type="text"
                      value={privateKeyFile?.name || '.keystore / .key'}
                      readOnly
                      className="flex-1 px-4 py-2 border border-gray-300 dark:border-slate-600 rounded-lg bg-gray-50 dark:bg-slate-700 text-gray-500 dark:text-slate-400"
                    />
                    <label className="px-4 py-2 bg-gray-100 dark:bg-slate-700 text-gray-700 dark:text-slate-300 rounded-lg hover:bg-gray-200 dark:hover:bg-slate-600 cursor-pointer">
                      Examinar
                      <input
                        type="file"
                        accept=".key,.keystore,.pem"
                        onChange={handlePrivateKeySelect}
                        disabled={importing}
                        className="hidden"
                      />
                    </label>
                  </div>
                </div>
              )}

              {/* Contraseña de Keystore - Solo si se requiere */}
              {showPassword && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
                    Contraseña de Keystore <span className="text-red-500">*</span>
                  </label>
                  <input
                    type="password"
                    value={keystorePassword}
                    onChange={(e) => setKeystorePassword(e.target.value)}
                    disabled={importing}
                    className="w-full px-4 py-2 border border-gray-300 dark:border-slate-600 rounded-lg bg-white dark:bg-slate-700 text-gray-900 dark:text-slate-50 focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                    placeholder="Contraseña del keystore (si aplica)"
                  />
                </div>
              )}

              {/* Certificados Intermedios - Solo si se requiere */}
              {showIntermediate && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-slate-300 mb-2">
                    Certificado(s) intermedio/raiz
                  </label>
                  <div className="space-y-3">
                    <div className="flex gap-4">
                      <label className="flex items-center">
                        <input
                          type="radio"
                          value="automatic"
                          checked={intermediateMode === 'automatic'}
                          onChange={(e) => setIntermediateMode(e.target.value as 'manual' | 'automatic')}
                          disabled={importing}
                          className="mr-2"
                        />
                        <span className="text-sm text-gray-700 dark:text-slate-300">Automático</span>
                      </label>
                      <label className="flex items-center">
                        <input
                          type="radio"
                          value="manual"
                          checked={intermediateMode === 'manual'}
                          onChange={(e) => setIntermediateMode(e.target.value as 'manual' | 'automatic')}
                          disabled={importing}
                          className="mr-2"
                        />
                        <span className="text-sm text-gray-700 dark:text-slate-300">Manual</span>
                      </label>
                    </div>

                    {intermediateMode === 'automatic' ? (
                      <div className="p-4 bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-800 rounded-lg">
                        <p className="text-sm text-blue-700 dark:text-blue-300">
                          Se detectarán y descargarán automáticamente los certificados intermedios. Asegúrese de que el servidor esté conectado a Internet.
                        </p>
                      </div>
                    ) : (
                      <>
                        <div className="p-4 bg-blue-50 dark:bg-blue-900/30 border border-blue-200 dark:border-blue-800 rounded-lg">
                          <p className="text-sm text-blue-700 dark:text-blue-300">
                            Elija los certificados intermedios y cárguelos manualmente.
                          </p>
                        </div>
                        <div className="flex gap-2">
                          <input
                            type="text"
                            value={intermediateFiles.length > 0 ? `${intermediateFiles.length} archivo(s) seleccionado(s)` : '.cer / .crt'}
                            readOnly
                            className="flex-1 px-4 py-2 border border-gray-300 dark:border-slate-600 rounded-lg bg-gray-50 dark:bg-slate-700 text-gray-500 dark:text-slate-400"
                          />
                          <label className="px-4 py-2 bg-gray-100 dark:bg-slate-700 text-gray-700 dark:text-slate-300 rounded-lg hover:bg-gray-200 dark:hover:bg-slate-600 cursor-pointer">
                            Examinar
                            <input
                              type="file"
                              accept=".cer,.crt,.pem"
                              multiple
                              onChange={(e) => {
                                const files = Array.from(e.target.files || []);
                                setIntermediateFiles([...intermediateFiles, ...files]);
                              }}
                              disabled={importing}
                              className="hidden"
                            />
                          </label>
                        </div>
                        {intermediateFiles.length > 0 && (
                          <div className="space-y-2">
                            {intermediateFiles.map((file, index) => (
                              <div key={index} className="flex items-center justify-between p-2 bg-gray-50 dark:bg-slate-700 rounded">
                                <span className="text-sm text-gray-700 dark:text-slate-300">{file.name}</span>
                                <button
                                  onClick={() => setIntermediateFiles(intermediateFiles.filter((_, i) => i !== index))}
                                  disabled={importing}
                                  className="text-red-600 dark:text-red-400 hover:text-red-800 dark:hover:text-red-300"
                                >
                                  <XMarkIcon className="h-5 w-5" />
                                </button>
                              </div>
                            ))}
                          </div>
                        )}
                      </>
                    )}
                  </div>
                </div>
              )}

              {error && (
                <div className="p-4 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-800 rounded-lg text-red-700 dark:text-red-400">
                  {error}
                </div>
              )}
            </div>

            <div className="flex justify-end gap-3 p-6 border-t border-gray-200 dark:border-slate-700">
              <button
                onClick={() => setShowImportModal(false)}
                disabled={importing}
                className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-slate-300 bg-gray-100 dark:bg-slate-700 rounded-md hover:bg-gray-200 dark:hover:bg-slate-600 disabled:opacity-50"
              >
                Cancelar
              </button>
              <button
                onClick={handleImport}
                disabled={importing || !certificateFile || analyzingCertificate}
                className="px-4 py-2 text-sm font-medium text-white bg-primary-600 rounded-md hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {importing ? 'Importando...' : 'Importar'}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Modal de Progreso */}
      {showProgressModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-[60]">
          <div className="bg-white dark:bg-slate-800 rounded-lg shadow-xl max-w-md w-full mx-4 transition-colors">
            <div className="p-6">
              <h2 className="text-xl font-semibold text-gray-900 dark:text-slate-50 mb-6">
                Importando Certificado SSL
              </h2>
              
              <div className="space-y-4">
                {progressSteps.map((step, index) => (
                  <div key={index} className="flex items-center gap-3">
                    {step.status === 'completed' && (
                      <CheckCircleIcon className="h-5 w-5 text-green-500 flex-shrink-0" />
                    )}
                    {step.status === 'loading' && (
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-primary-600 flex-shrink-0"></div>
                    )}
                    {step.status === 'pending' && (
                      <div className="h-5 w-5 rounded-full border-2 border-gray-300 dark:border-slate-600 flex-shrink-0"></div>
                    )}
                    {step.status === 'error' && (
                      <div className="h-5 w-5 rounded-full bg-red-500 flex-shrink-0"></div>
                    )}
                    <span className={`text-sm ${
                      step.status === 'completed' 
                        ? 'text-green-600 dark:text-green-400' 
                        : step.status === 'loading'
                        ? 'text-primary-600 dark:text-primary-400'
                        : step.status === 'error'
                        ? 'text-red-600 dark:text-red-400'
                        : 'text-gray-500 dark:text-slate-400'
                    }`}>
                      {step.step}
                    </span>
                  </div>
                ))}
              </div>

              {progressSteps.every(step => step.status === 'completed') && (
                <div className="mt-6 p-4 bg-green-50 dark:bg-green-900/30 border border-green-200 dark:border-green-800 rounded-lg">
                  <p className="text-sm font-medium text-green-800 dark:text-green-400 text-center">
                    Importación correcta
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
