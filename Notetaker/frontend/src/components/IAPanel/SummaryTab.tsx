import React, { useState, useEffect, useRef } from 'react';
import { ClockIcon, ExclamationTriangleIcon, ClipboardDocumentIcon, CheckIcon } from '@heroicons/react/24/outline';
import { getSpeakerColor } from '../../utils/speakerColors';

interface SummaryTabProps {
  summaryData: any;
  loading: boolean;
}

export const SummaryTab: React.FC<SummaryTabProps> = ({ summaryData, loading }) => {
  const [copied, setCopied] = useState(false);
  const summaryContainerRef = useRef<HTMLDivElement>(null);
  // Estado para la barra de progreso (siempre inicializar, aunque solo se use en 'processing')
  const [progress, setProgress] = useState(30);

  // Simular progreso gradual solo cuando está procesando (actualizar cada 5 segundos)
  // IMPORTANTE: Este hook debe estar ANTES de cualquier return condicional
  useEffect(() => {
    if (summaryData && summaryData.status === 'processing') {
      // Resetear progreso cuando empieza a procesar
      setProgress(30);

      const interval = setInterval(() => {
        setProgress((prev) => {
          // Incrementar progreso gradualmente hasta 90% (nunca llegar a 100% hasta que termine)
          if (prev < 90) {
            return Math.min(prev + 2, 90);
          }
          return prev;
        });
      }, 5000);
      return () => clearInterval(interval);
    } else {
      // Resetear progreso cuando no está procesando
      setProgress(30);
    }
  }, [summaryData?.status]);

  // Estado para controlar la visualización temporal del error
  const [showError, setShowError] = useState(false);
  const [isRestoring, setIsRestoring] = useState(false);

  // Efecto para manejar el error temporal y volver al resumen original
  useEffect(() => {
    if (summaryData && summaryData.status === 'failed') {
      // Solo intentar restaurar si hay datos previos (toon o insights)
      const hasPreviousData = summaryData.toon || (summaryData.insights && Object.keys(summaryData.insights).length > 0);

      if (hasPreviousData) {
        setShowError(true);
        // Mostrar error por 5 segundos
        const timer = setTimeout(() => {
          setIsRestoring(true);
          // Pequeña pausa para mostrar mensaje de restauración
          setTimeout(() => {
            setShowError(false);
            setIsRestoring(false);
          }, 1500);
        }, 5000);
        return () => clearTimeout(timer);
      }
    } else {
      // Si cambia el estado (ej. a pending o processing), resetear error
      setShowError(false);
      setIsRestoring(false);
    }
  }, [summaryData?.status]);

  // Estado de carga inicial
  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[200px]">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  // Estado no disponible (solo si no estamos en modo restauracion/error temporal)
  if ((!summaryData || summaryData.status === 'not_available') && !showError) {
    return (
      <div className="text-center py-8">
        <p className="text-gray-600 dark:text-slate-400">{summaryData?.message || 'Esta reunión no tiene análisis disponible aún'}</p>
      </div>
    );
  }

  // Función auxiliar para formatear tiempo en segundos a texto legible
  const formatTime = (seconds: number): string => {
    if (seconds < 60) {
      return `${Math.round(seconds)} seg`;
    } else if (seconds < 3600) {
      const minutes = Math.floor(seconds / 60);
      const secs = Math.round(seconds % 60);
      if (secs === 0) {
        return `${minutes} min`;
      }
      return `${minutes} min ${secs} seg`;
    } else {
      const hours = Math.floor(seconds / 3600);
      const minutes = Math.floor((seconds % 3600) / 60);
      if (minutes === 0) {
        return `${hours} h`;
      }
      return `${hours} h ${minutes} min`;
    }
  };

  // Estado pendiente (en cola)
  if (summaryData.status === 'pending') {
    const queueEst = summaryData.queue_estimation;
    return (
      <div className="text-center py-8">
        <ClockIcon className="h-12 w-12 text-gray-400 dark:text-slate-500 mx-auto mb-4" />
        <p className="text-gray-700 dark:text-slate-300 font-medium mb-2">El análisis está en cola</p>
        <p className="text-sm text-gray-600 dark:text-slate-400 mb-4">
          {summaryData.message || 'Se procesará en breve...'}
        </p>

        {queueEst && (
          <div className="mt-6 space-y-3 text-left max-w-md mx-auto bg-gray-50 dark:bg-slate-700/50 rounded-lg p-4">
            {queueEst.meetings_ahead > 0 && (
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600 dark:text-slate-400">Reuniones por delante:</span>
                <span className="text-sm font-medium text-gray-900 dark:text-slate-200">
                  {queueEst.meetings_ahead} {queueEst.meetings_ahead === 1 ? 'reunión' : 'reuniones'}
                </span>
              </div>
            )}
            {queueEst.estimated_wait_seconds > 0 && (
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600 dark:text-slate-400">Tiempo estimado de espera:</span>
                <span className="text-sm font-medium text-gray-900 dark:text-slate-200">
                  {formatTime(queueEst.estimated_wait_seconds)}
                </span>
              </div>
            )}
            {queueEst.estimated_processing_seconds > 0 && (
              <div className="flex justify-between items-center">
                <span className="text-sm text-gray-600 dark:text-slate-400">Tiempo estimado de procesamiento:</span>
                <span className="text-sm font-medium text-gray-900 dark:text-slate-200">
                  {formatTime(queueEst.estimated_processing_seconds)}
                </span>
              </div>
            )}
            {queueEst.meetings_ahead === 0 && queueEst.estimated_wait_seconds === 0 && (
              <div className="text-sm text-gray-600 dark:text-slate-400 text-center">
                Tu reunión es la siguiente en procesarse
              </div>
            )}
          </div>
        )}
      </div>
    );
  }

  // Estado procesando
  if (summaryData.status === 'processing') {
    const queueEst = summaryData.queue_estimation;
    // Calcular progreso estimado basado en tiempo transcurrido y estimado
    const estimatedTotal = queueEst?.estimated_processing_seconds || 300; // Default 5 min

    return (
      <div className="text-center py-8">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto mb-4"></div>
        <p className="text-gray-700 dark:text-slate-300 font-medium mb-2">Analizando la reunión</p>
        <p className="text-sm text-gray-600 dark:text-slate-400 mb-4">
          {summaryData.message || 'Este proceso puede tardar unos minutos, los resultados aparecerán automáticamente.'}
        </p>

        {/* Barra de progreso */}
        <div className="mt-4 w-full max-w-md mx-auto">
          <div className="w-full bg-gray-200 dark:bg-slate-700 rounded-full h-2.5 mb-2">
            <div
              className="bg-primary-600 h-2.5 rounded-full transition-all duration-500 ease-out"
              style={{ width: `${progress}%` }}
            ></div>
          </div>
          <p className="text-xs text-gray-500 dark:text-slate-400">
            {progress}% completado
          </p>
        </div>

        {queueEst && queueEst.estimated_processing_seconds > 0 && (
          <div className="mt-4 text-sm text-gray-600 dark:text-slate-400">
            Tiempo estimado restante: {formatTime(estimatedTotal * (1 - progress / 100))}
          </div>
        )}
      </div>
    );
  }

  // Estado fallido (o mostrando error temporal)
  if (summaryData.status === 'failed') {
    // Si estamos mostrando error o no hay datos previos para restaurar
    const hasPreviousData = summaryData.toon || (summaryData.insights && Object.keys(summaryData.insights).length > 0);

    if (showError || !hasPreviousData) {
      if (isRestoring) {
        return (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-600 mx-auto mb-4"></div>
            <p className="text-gray-700 dark:text-slate-300 font-medium">Volviendo al resumen original...</p>
          </div>
        );
      }

      return (
        <div className="text-center py-8">
          <ExclamationTriangleIcon className="h-12 w-12 text-red-500 dark:text-red-400 mx-auto mb-4" />
          <p className="text-gray-700 dark:text-slate-300 font-medium mb-2">Error al procesar</p>
          <p className="text-sm text-gray-600 dark:text-slate-400 mb-2">
            {summaryData.message || 'No se pudo completar el análisis. Inténtalo de nuevo más tarde.'}
          </p>
          {summaryData.error && (
            <p className="text-xs text-red-600 dark:text-red-400 mt-2">{summaryData.error}</p>
          )}
          {hasPreviousData && (
            <p className="text-xs text-gray-500 dark:text-slate-500 mt-4">
              Restaurando resumen anterior en unos segundos...
            </p>
          )}
        </div>
      );
    }
  }

  // Estado completado - mostrar datos
  const insights = summaryData.insights || {};
  const actionItems = insights.action_items || [];
  const participationPercent = insights.participation_percent || {};
  const allSpeakers = Object.keys(participationPercent);

  // Extraer puntos clave del toon (markdown) o usar summary
  let toonText = summaryData.toon || insights.summary || '';

  // Funcion para eliminar la seccion "Tareas asignadas" del toon
  // (las tareas se muestran en una seccion separada con mejor formato)
  const removeTareasSection = (text: string): string => {
    const lines = text.split('\n');
    const result: string[] = [];
    let inTareasSection = false;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const trimmedLine = line.trim().toLowerCase();

      // Detectar inicio de seccion de tareas (varias variantes)
      if (trimmedLine.startsWith('# tareas asignadas') ||
        trimmedLine.startsWith('## tareas asignadas') ||
        trimmedLine.startsWith('### tareas asignadas') ||
        trimmedLine === '# tareas' ||
        trimmedLine === '## tareas' ||
        trimmedLine === '### tareas') {
        inTareasSection = true;
        continue;
      }

      // Detectar fin de seccion (siguiente seccion con # o fin del texto)
      if (inTareasSection) {
        if (line.trim().startsWith('#') && !trimmedLine.includes('tareas')) {
          // Nueva seccion que no es de tareas, salir de la seccion de tareas
          inTareasSection = false;
          result.push(line);
        }
        // Si estamos en seccion de tareas, no añadir la linea
        continue;
      }

      // Si no estamos en seccion de tareas, añadir la linea
      result.push(line);
    }

    return result.join('\n');
  };

  // Eliminar seccion de tareas del toon (se muestran aparte)
  toonText = removeTareasSection(toonText);

  // Funcion para eliminar la linea resumen de "Senales" que aparece antes de "Carga y papeles"
  // (las senales se muestran desarrolladas en la seccion "Senales de reunion (metricas)")
  const removeSignalsShortLine = (text: string): string => {
    return text
      // Eliminar linea que empieza con "Senales:" y contiene colaboracion/decision/conflicto
      .replace(/^Se[ñn]ales:\s*colaboraci[oó]n\s+\d+\/\d+.*$/gim, '')
      // Limpiar lineas vacias duplicadas que puedan quedar
      .replace(/\n{3,}/g, '\n\n');
  };

  // Eliminar linea resumen de senales (se muestran desarrolladas mas abajo)
  toonText = removeSignalsShortLine(toonText);

  // Función para calcular roles dinámicamente basándose en porcentajes de participación
  // Sistema Híbrido del documento: combina umbrales absolutos y relativos
  const assignMeetingRoles = (participants: Array<{ name: string; percentage: number }>): Array<{ name: string; percentage: number; role: string }> => {
    if (participants.length === 0) return [];
    
    // Clonar y ordenar por porcentaje descendente
    const sorted = [...participants].sort((a, b) => b.percentage - a.percentage);
    const n = sorted.length;
    
    // Si solo hay 1 participante
    if (n === 1) {
      return [{ ...sorted[0], role: 'driver' }];
    }
    
    // Contador de roles asignados (para seguir el sistema híbrido)
    const rolesAssigned = { 'driver': 0, 'co-driver': 0, 'contributor': 0, 'assistant': 0 };
    
    // Crear array de resultados con roles
    const result: Array<{ name: string; percentage: number; role: string }> = [];
    
    // Procesar cada participante en orden
    for (let i = 0; i < n; i++) {
      const p = sorted[i];
      const pct = p.percentage;
      let role: string;
      
      // DRIVER: El primero, si tiene > 40% o es claramente dominante
      if (i === 0) {
        role = 'driver';
        rolesAssigned['driver'] += 1;
      }
      // CO-DRIVER: El segundo, si tiene > 25% y la diferencia con el primero < 25%
      else if (i === 1 && rolesAssigned['co-driver'] === 0) {
        const diff = sorted[0].percentage - pct;
        if (pct >= 25 && diff < 25) {
          role = 'co-driver';
          rolesAssigned['co-driver'] += 1;
        } else if (pct >= 12) {
          role = 'contributor';
          rolesAssigned['contributor'] += 1;
        } else {
          role = 'assistant';
          rolesAssigned['assistant'] += 1;
        }
      }
      // CONTRIBUTOR: 3º-Nº si tiene > 12%
      else if (pct >= 12) {
        role = 'contributor';
        rolesAssigned['contributor'] += 1;
      }
      // ASSISTANT: El resto
      else {
        role = 'assistant';
        rolesAssigned['assistant'] += 1;
      }
      
      // Crear nuevo objeto con el rol asignado
      result.push({ ...p, role });
    }
    
    return result;
  };

  // Función para extraer métricas del markdown original
  const extractMetricsFromMarkdown = (text: string): { 
    participationPercent: Record<string, number>,
    talkTime: Record<string, number>, 
    turns: Record<string, number>, 
    responsivity: Record<string, number> 
  } => {
    const participationPercent: Record<string, number> = {};
    const talkTime: Record<string, number> = {};
    const turns: Record<string, number> = {};
    const responsivity: Record<string, number> = {};
    
    const lines = text.split('\n');
    let inCargaSection = false;
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      
      // Detectar inicio de sección "Carga y Roles" o "Carga y papeles"
      if (line.includes('Carga y Roles') || line.includes('Carga y papeles')) {
        inCargaSection = true;
        continue;
      }
      
      // Detectar fin de sección
      if (inCargaSection && (line.startsWith('##') || line.startsWith('###'))) {
        break;
      }
      
      // Procesar líneas de participantes dentro de la sección
      if (inCargaSection && (line.startsWith('-') || line.startsWith('*'))) {
        // Formato esperado de Cosmos: - **Nombre**: Role — XX.X% · YYYs · Z turnos
        // Ejemplo: - **Elena Diaz Barcenilla**: Colaborador — 43.2% · 718s · 66 turnos
        
        let name: string | null = null;
        
        // Patrón principal: - **Nombre**: (puede haber texto después antes del porcentaje)
        const nameMatch = line.match(/^[-*]\s*\*\*([^*]+)\*\*\s*:/);
        if (nameMatch && nameMatch[1]) {
          name = nameMatch[1].trim();
        } else {
          // Patrón alternativo: - Nombre: (sin negrita)
          const altNameMatch = line.match(/^[-*]\s*([^:]+?)\s*:/);
          if (altNameMatch && altNameMatch[1]) {
            name = altNameMatch[1].trim();
          }
        }
        
        // Si aún no tenemos nombre, intentar extraerlo antes del primer porcentaje
        if (!name) {
          const beforePercentMatch = line.match(/^[-*]\s*(.+?)(?=\s*\d+\.?\d*%)/);
          if (beforePercentMatch && beforePercentMatch[1]) {
            name = beforePercentMatch[1].trim();
            // Limpiar formato markdown y caracteres especiales
            name = name.replace(/\*\*/g, '').replace(/\*/g, '').replace(/[:—]+$/, '').trim();
          }
        }
        
        if (name && name.length > 0) {
          // Extraer porcentaje: XX.X% o XX% (buscar el primer porcentaje en la línea)
          const percentMatch = line.match(/(\d+\.?\d*)\s*%/);
          if (percentMatch) {
            participationPercent[name] = parseFloat(percentMatch[1]);
          }
          
          // Extraer segundos: YYYs (buscar patrón de segundos después de ·)
          const secondsMatch = line.match(/·\s*(\d+)\s*s\b/);
          if (secondsMatch) {
            talkTime[name] = parseInt(secondsMatch[1], 10);
          } else {
            // Intentar sin el · por si acaso
            const secondsMatchAlt = line.match(/(\d+)\s*s\b/);
            if (secondsMatchAlt) {
              talkTime[name] = parseInt(secondsMatchAlt[1], 10);
            }
          }
          
          // Extraer turnos: Z turnos
          const turnsMatch = line.match(/(\d+)\s+turnos/);
          if (turnsMatch) {
            turns[name] = parseInt(turnsMatch[1], 10);
          }
          
          // Extraer responsividad: Responsividad AA/100 o resp AA/100 (puede no estar presente)
          const respMatch = line.match(/(?:Responsividad|resp)\s*(\d+)\/(\d+)/);
          if (respMatch) {
            responsivity[name] = parseInt(respMatch[1], 10);
          }
        }
      }
    }
    
    return { participationPercent, talkTime, turns, responsivity };
  };

  // Función para formatear la sección "Carga y Roles" usando datos de insights
  const formatCargaYRolesSection = (text: string): string => {
    const lines = text.split('\n');
    let cargaSectionStartIndex = -1;
    let cargaSectionEndIndex = -1;
    
    // Buscar inicio y fin de la sección "Carga y Roles"
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const lineLower = line.toLowerCase();
      
      // Detectar inicio de sección "Carga y Roles" o "Carga y papeles" (case insensitive)
      // Puede venir como ## Carga y Roles o ### Carga y roles
      if ((lineLower.includes('carga y roles') || lineLower.includes('carga y papeles')) && cargaSectionStartIndex === -1) {
        cargaSectionStartIndex = i;
        // Continuar para buscar el fin de la sección
        continue;
      }
      
      // Detectar fin de sección (siguiente sección con ## o ### que no sea "Carga y Roles")
      if (cargaSectionStartIndex >= 0 && cargaSectionEndIndex === -1) {
        // Si encontramos otra sección (## o ###) que no sea "Carga y Roles", es el fin
        if ((line.startsWith('##') || line.startsWith('###'))) {
          const isCargaSection = lineLower.includes('carga y roles') || lineLower.includes('carga y papeles');
          if (!isCargaSection) {
            cargaSectionEndIndex = i;
            break;
          }
        }
      }
    }
    
    // Si no encontramos fin, usar el final del texto
    if (cargaSectionStartIndex >= 0 && cargaSectionEndIndex === -1) {
      cargaSectionEndIndex = lines.length;
    }
    
    // Si encontramos la sección, intentar reemplazarla (usar datos de insights o extraer del markdown)
    if (cargaSectionStartIndex >= 0) {
      // Extraer métricas del markdown original (siempre intentar extraer)
      const markdownMetrics = extractMetricsFromMarkdown(text);
      
      // Usar participation_percent de insights si está disponible y tiene datos, sino usar el del markdown
      const finalParticipationPercent = (participationPercent && Object.keys(participationPercent).length > 0) 
        ? participationPercent 
        : markdownMetrics.participationPercent;
      
      // Solo procesar si tenemos datos de participación (de insights o del markdown)
      if (finalParticipationPercent && Object.keys(finalParticipationPercent).length > 0) {
        // Preparar participantes con porcentajes
        const participants = Object.entries(finalParticipationPercent).map(([name, percentage]) => ({
          name,
          percentage: typeof percentage === 'number' ? percentage : 0
        }));
        
        // Calcular roles
        const participantsWithRoles = assignMeetingRoles(participants);
        
        // Obtener métricas adicionales de insights o extraer del markdown
        // Nota: en insights los segundos vienen como talk_time_seconds, no talk_time
        const insightsTalkTime = insights.talk_time_seconds || insights.talk_time || {};
        const insightsTurns = insights.turns || {};
        const insightsResponsivity = insights.responsivity || {};
        
        // Combinar: priorizar insights (si tienen datos), luego markdown como fallback
        // Usar markdown primero, luego sobrescribir con insights si existen
        const talkTime = { ...markdownMetrics.talkTime };
        const turns = { ...markdownMetrics.turns };
        const responsivity = { ...markdownMetrics.responsivity };
        
        // Sobrescribir con datos de insights si existen (y tienen valores válidos)
        Object.keys(insightsTalkTime).forEach((name) => {
          const value = insightsTalkTime[name];
          if (value !== undefined && value !== null && value > 0) {
            talkTime[name] = value;
          }
        });
        
        Object.keys(insightsTurns).forEach((name) => {
          const value = insightsTurns[name];
          if (value !== undefined && value !== null && value > 0) {
            turns[name] = value;
          }
        });
        
        Object.keys(insightsResponsivity).forEach((name) => {
          const value = insightsResponsivity[name];
          if (value !== undefined && value !== null) {
            responsivity[name] = value;
          }
        });
        
        // Extraer la línea "Responsable principal" del markdown original si existe
        let responsableLine: string | null = null;
        for (let i = cargaSectionStartIndex; i < cargaSectionEndIndex; i++) {
          const line = lines[i];
          if (line.includes('Responsable principal') || line.includes('responsable principal')) {
            responsableLine = line.trim();
            break;
          }
        }
        
        // Generar markdown formateado
        const cargaSectionLines: string[] = ['## Carga y Roles', ''];
        
        // Añadir la línea del responsable principal si existe
        if (responsableLine) {
          cargaSectionLines.push(responsableLine);
          cargaSectionLines.push(''); // Línea en blanco de separación
        }
        
        // Añadir participantes con sus roles y métricas
        participantsWithRoles.forEach((participant) => {
          const pct = participant.percentage.toFixed(1);
          // Verificar si existen datos de tiempo (puede ser 0, pero debe existir)
          const hasTalkTime = participant.name in talkTime;
          const talkTimeSec = hasTalkTime ? Math.round(talkTime[participant.name]) : null;
          const hasTurns = participant.name in turns;
          const turnsCount = hasTurns ? turns[participant.name] : null;
          const respValue = participant.name in responsivity ? responsivity[participant.name] : null;
          
          // Mapear roles a nombres en inglés para mostrar
          const roleDisplayMap: Record<string, string> = {
            'driver': 'Driver',
            'co-driver': 'Co-driver',
            'contributor': 'Contributor',
            'assistant': 'Assistant'
          };
          const roleDisplay = roleDisplayMap[participant.role] || participant.role;
          
          // Formato: - **Nombre**: Role — XX.X% · YYYs · Z turnos · Responsividad AA/100
          // Los tooltips se aplican después en la pipeline de HTML (regex sobre el HTML final)
          let metricsLine = `- **${participant.name}**: ${roleDisplay} — ${pct}%`;
          
          // Añadir segundos si existen (incluso si son 0)
          if (talkTimeSec !== null && talkTimeSec !== undefined) {
            metricsLine += ` · ${talkTimeSec}s`;
          }
          
          // Añadir turnos si existen
          if (turnsCount !== null && turnsCount !== undefined) {
            metricsLine += ` · ${turnsCount} turnos`;
          }
          
          // Añadir responsividad si existe
          if (respValue !== null && respValue !== undefined) {
            metricsLine += ` · Responsividad ${Math.round(respValue)}/100`;
          }
          
          cargaSectionLines.push(metricsLine);
        });
        
        // Reemplazar la sección en el texto
        const beforeSection = lines.slice(0, cargaSectionStartIndex);
        const afterSection = lines.slice(cargaSectionEndIndex);
        
        return [...beforeSection, ...cargaSectionLines, '', ...afterSection].join('\n');
      }
    }
    
    // Si no hay datos de participación o no se encontró la sección, mantener el texto original
    return text;
  };

  // Aplicar formato a la sección Carga y Roles
  toonText = formatCargaYRolesSection(toonText);

  // Función para deduplicar participantes en la sección "Carga y Roles"
  // Mantiene solo una entrada por participante, priorizando el rol más relevante
  // Reordena para mostrar "Responsable principal" antes de la lista de participantes
  const deduplicateParticipants = (text: string): string => {
    const lines = text.split('\n');
    const result: string[] = [];
    let inCargaSection = false;
    const participantLines = new Map<string, { line: string; priority: number }>(); // nombre -> {línea, prioridad}
    let responsablePrincipalLine: string | null = null; // Línea del responsable principal

    // Prioridad de roles: Driver=3, Co-driver=2, Contributor=1
    const getRolePriority = (line: string): number => {
      if (line.includes(': Driver') || line.includes('Driver —')) return 3;
      if (line.includes(': Co-driver') || line.includes('Co-driver —')) return 2;
      if (line.includes(': Contributor') || line.includes('Contributor —')) return 1;
      return 0;
    };

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];

      // Detectar inicio de sección "Carga y Roles" o "Carga y papeles"
      if (line.includes('Carga y Roles') || line.includes('Carga y papeles')) {
        inCargaSection = true;
        result.push(line);
        participantLines.clear(); // Limpiar al empezar nueva sección
        responsablePrincipalLine = null; // Limpiar responsable principal
        continue;
      }

      // Detectar fin de sección (siguiente sección o título, o línea en blanco después de participantes)
      if (inCargaSection) {
        if (line.startsWith('##') || line.startsWith('###')) {
          // Antes de cerrar, añadir responsable principal (si existe), espacio, y luego participantes
          if (responsablePrincipalLine) {
            result.push(responsablePrincipalLine);
            result.push(''); // Espacio de separación
          }
          const sortedParticipants = Array.from(participantLines.entries())
            .sort((a, b) => b[1].priority - a[1].priority); // Ordenar por prioridad (mayor primero)
          sortedParticipants.forEach(([_, data]) => {
            result.push(data.line);
          });
          participantLines.clear();
          responsablePrincipalLine = null;
          inCargaSection = false;
          result.push(line);
          continue;
        }

        // Si hay una línea en blanco y ya tenemos participantes, podría ser el fin
        if (line.trim() === '' && participantLines.size > 0 && i < lines.length - 1) {
          const nextLine = lines[i + 1];
          // Si la siguiente línea no es un participante ni el responsable principal, cerrar sección
          if (!nextLine.match(/^[-*]\s*[^:—]+/) && !nextLine.includes('Responsable principal')) {
            // Añadir responsable principal (si existe), espacio, y luego participantes
            if (responsablePrincipalLine) {
              result.push(responsablePrincipalLine);
              result.push(''); // Espacio de separación
            }
            const sortedParticipants = Array.from(participantLines.entries())
              .sort((a, b) => b[1].priority - a[1].priority);
            sortedParticipants.forEach(([_, data]) => {
              result.push(data.line);
            });
            participantLines.clear();
            responsablePrincipalLine = null;
            inCargaSection = false;
            result.push(line);
            continue;
          }
        }
      }

      // Si estamos en la sección de Carga y Roles, detectar responsable principal
      if (inCargaSection && line.includes('Responsable principal')) {
        responsablePrincipalLine = line;
        continue; // No añadir todavía, lo añadiremos antes de los participantes
      }

      // Si estamos en la sección de Carga y Roles, procesar participantes
      if (inCargaSection && (line.startsWith('-') || line.startsWith('*'))) {
        // Extraer nombre del participante (antes de los dos puntos o guión)
        const match = line.match(/^[-*]\s*([^:—]+?)(?:\s*[:—]\s*)(.*)$/);
        if (match) {
          const participantName = match[1].trim();
          const metrics = match[2].trim(); // Todo lo que viene después de : o —
          const rolePriority = getRolePriority(line);

          // Reformatear la línea: nombre en una línea, métricas en la siguiente
          const formattedLine = `${line.charAt(0)} ${participantName}:\n${metrics}`;

          // Si ya vimos este participante, comparar prioridades
          if (participantLines.has(participantName)) {
            const existing = participantLines.get(participantName)!;
            // Mantener la línea con mayor prioridad (rol más relevante)
            if (rolePriority > existing.priority) {
              participantLines.set(participantName, { line: formattedLine, priority: rolePriority });
            }
            // No añadir esta línea al resultado todavía
            continue;
          } else {
            // Primera vez que vemos este participante
            participantLines.set(participantName, { line: formattedLine, priority: rolePriority });
            continue; // No añadir todavía, lo añadiremos al final de la sección
          }
        }
      }

      // Si no es un participante, responsable principal, o no estamos en la sección, añadir la línea normalmente
      if (!inCargaSection || (!line.includes('Responsable principal') && !line.startsWith('-') && !line.startsWith('*'))) {
        result.push(line);
      }
    }

    // Si terminamos y aún estamos en la sección, añadir responsable principal (si existe), espacio, y luego participantes
    if (inCargaSection) {
      if (responsablePrincipalLine) {
        result.push(responsablePrincipalLine);
        result.push(''); // Espacio de separación
      }
      if (participantLines.size > 0) {
        const sortedParticipants = Array.from(participantLines.entries())
          .sort((a, b) => b[1].priority - a[1].priority);
        sortedParticipants.forEach(([_, data]) => {
          result.push(data.line);
        });
      }
    }

    return result.join('\n');
  };

  // Aplicar deduplicación
  toonText = deduplicateParticipants(toonText);

  // Función para procesar HTML y convertir title a data-tooltip-text (evita tooltip nativo)
  const processHtmlForTooltips = (html: string): string => {
    return html.replace(
      /title\s*=\s*["']([^"']*)["']/gi,
      'data-tooltip-text="$1"'
    );
  };

  // Función auxiliar para convertir marcadores COLOR a HTML (ejecutar múltiples veces para asegurar conversión completa)
  const convertAllColorMarkers = (html: string): string => {
    let result = html;
    let previousResult = '';
    // Ejecutar hasta que no haya más cambios (para manejar casos anidados o complejos)
    while (result !== previousResult) {
      previousResult = result;
      result = result.replace(/\{\{COLOR:([^:]+):([^}]+)\}\}/g, '<span style="color: $1; font-weight: 500">$2</span>');
    }
    return result;
  };

  // Función para colorear nombres de participantes en texto markdown (antes de convertir a HTML)
  const colorizeSpeakerNamesInMarkdown = (text: string): string => {
    if (allSpeakers.length === 0) return text;

    let coloredText = text;
    // Ordenar speakers por longitud (más largos primero) para evitar reemplazos parciales
    const sortedSpeakers = [...allSpeakers].sort((a, b) => b.length - a.length);

    sortedSpeakers.forEach((speaker) => {
      const speakerColor = getSpeakerColor(speaker, allSpeakers);
      // Escapar caracteres especiales del nombre para regex
      const escapedName = speaker.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

      // Reemplazar nombres en negrita: **Nombre** -> **{{COLOR:Nombre}}**
      coloredText = coloredText.replace(
        new RegExp(`\\*\\*(${escapedName})\\*\\*`, 'gi'),
        `**{{COLOR:${speakerColor.main}:$1}}**`
      );

      // Reemplazar nombres sin formato (evitar si está después de : o —)
      coloredText = coloredText.replace(
        new RegExp(`(?<![:—]\\s)(${escapedName})(?!\\s*[—:])`, 'gi'),
        `{{COLOR:${speakerColor.main}:$1}}`
      );
    });

    return coloredText;
  };

  // Parsear puntos clave del toon (buscar líneas que empiezan con - o *)
  const keyPoints: string[] = [];
  const seenPoints = new Set<string>(); // Para evitar duplicados
  if (toonText) {
    const lines = toonText.split('\n');
    for (const line of lines) {
      const trimmed = line.trim();

      // Permitir líneas que empiezan con ** si contienen ":" (como "**Temas tratados:**")
      // Estas las procesaremos manteniendo el formato de negrita
      const isBoldLine = trimmed.startsWith('**') && trimmed.includes(':');

      // Excluir líneas que son solo nombres de participantes (terminan en : sin más texto)
      if (trimmed.endsWith(':') && !trimmed.startsWith('-') && !trimmed.startsWith('*')) {
        // Verificar si es un nombre de participante conocido
        const nameWithoutColon = trimmed.slice(0, -1).trim();
        if (allSpeakers.some(speaker => speaker.toLowerCase() === nameWithoutColon.toLowerCase())) {
          continue;
        }
      }

      // Capturar líneas que empiezan con - o * (incluyendo ** si tienen :)
      // Excluir lineas de la seccion "Carga y papeles/Roles"
      const isResponsableLine = trimmed.toLowerCase().includes('responsable principal');
      const isParticipantMetricsLine = trimmed.includes('Contributor') ||
        trimmed.includes('Driver') ||
        trimmed.includes('Co-driver') ||
        (trimmed.includes('%') && trimmed.includes('turnos'));

      // Verificar si es un nombre de participante con negrita (ej: "- **Elena Diaz**:" o "**Elena Diaz**:")
      const boldNameMatch = trimmed.match(/^\*\*([^*]+)\*\*\s*:?$/);
      const isParticipantNameOnly = boldNameMatch &&
        allSpeakers.some(speaker => speaker.toLowerCase() === boldNameMatch[1].trim().toLowerCase());

      // Verificar si la linea es un participante de la seccion Carga y Roles (empieza con - ** y tiene nombre conocido)
      const listBoldNameMatch = trimmed.match(/^[-*]\s*\*\*([^*]+)\*\*\s*:/);
      const isParticipantListItem = listBoldNameMatch &&
        allSpeakers.some(speaker => speaker.toLowerCase() === listBoldNameMatch[1].trim().toLowerCase());

      // Excluir lineas de metricas de senales de reunion
      const trimmedLower = trimmed.toLowerCase();
      const isSignalMetricLine = trimmedLower.includes('colaboraci') ||
        trimmedLower.includes('decisi') ||
        trimmedLower.includes('conflicto') ||
        trimmedLower.includes('clima:') ||
        trimmedLower.includes('valence') ||
        trimmedLower.includes('no se detectaron debates') ||
        trimmedLower.includes('no se detectaron decisiones');

      if (
        (trimmed.startsWith('-') || trimmed.startsWith('*') || isBoldLine) &&
        trimmed.length > 2 &&
        !trimmed.includes('_(evid:') &&
        !isResponsableLine &&
        !isParticipantMetricsLine &&
        !isParticipantNameOnly &&
        !isParticipantListItem &&
        !isSignalMetricLine &&
        !trimmed.includes('Quality flags')
      ) {
        let cleanedPoint: string;

        if (isBoldLine) {
          // Para líneas con **, mantener el formato de negrita pero quitar los asteriscos iniciales del marcador de lista
          // Si empieza con -**, quitar solo el -
          // Si empieza con **, mantenerlo como está (es un título/etiqueta)
          if (trimmed.startsWith('-**')) {
            cleanedPoint = trimmed.substring(1).trim(); // Quitar solo el -
          } else if (trimmed.startsWith('**')) {
            cleanedPoint = trimmed; // Mantener como está
          } else {
            cleanedPoint = trimmed.substring(1).trim();
          }
        } else {
          // Para líneas normales, quitar el marcador de lista (- o *)
          cleanedPoint = trimmed.substring(1).trim();
        }

        // Limpiar evidencia pero mantener el markdown de negrita
        cleanedPoint = cleanedPoint
          .replace(/\s*_\(evid:[^)]+\)\s*_/g, '')
          .replace(/\s*\(evid:[^)]+\)/g, '');

        // Excluir si es solo un nombre seguido de :
        if (cleanedPoint.endsWith(':') && !cleanedPoint.includes('**')) {
          const nameWithoutColon = cleanedPoint.slice(0, -1).trim();
          if (allSpeakers.some(speaker => speaker.toLowerCase() === nameWithoutColon.toLowerCase())) {
            continue;
          }
        }

        // Excluir la línea "Temas tratados:" con todos sus temas
        const normalizedForTemas = cleanedPoint
          .replace(/\*\*/g, '')
          .replace(/\*/g, '')
          .toLowerCase()
          .trim();
        if (normalizedForTemas.startsWith('temas tratados:')) {
          continue; // Saltar esta línea completamente
        }

        // Normalizar para comparación (minúsculas, sin espacios extra, sin markdown)
        const normalizedPoint = cleanedPoint
          .replace(/\*\*/g, '')
          .replace(/\*/g, '')
          .toLowerCase()
          .trim();

        if (cleanedPoint.length > 0 && !seenPoints.has(normalizedPoint)) {
          seenPoints.add(normalizedPoint);
          keyPoints.push(cleanedPoint);
        }
      }
    }
  }

  // Si no hay puntos clave extraídos, usar decisiones o topics
  if (keyPoints.length === 0 && insights.decisions) {
    insights.decisions.forEach((decision: string) => {
      if (decision && decision !== 'topic' && decision !== 'summary') {
        keyPoints.push(decision);
      }
    });
  }

  // Función para extraer texto limpio del contenedor preservando la estructura visual
  const extractCleanText = (element: HTMLElement): string => {
    // Usar innerText que preserva la estructura visual del navegador
    let text = element.innerText || '';

    // Limpiar y normalizar el texto
    text = text
      // Eliminar emojis de info que se usan para tooltips
      .replace(/ℹ️/g, '')
      // Unir bullets con su contenido (bullet seguido de salto de linea y texto)
      .replace(/•\s*\n+\s*/g, '• ')
      // Unir guiones de lista con su contenido
      .replace(/-\s*\n+\s*(?=[A-Z])/g, '- ')
      // Limpiar el formato del Clima (unir en una sola linea)
      .replace(/Clima:\s*\n+\s*/gi, 'Clima: ')
      .replace(/(neutro|positivo|negativo)\s*\n+\s*\(([^)]+)\)\s*\n+\s*/gi, '$1 ($2) ')
      // Limpiar multiples espacios en blanco
      .replace(/[ \t]+/g, ' ')
      // Limpiar lineas que solo tienen espacios
      .replace(/^\s+$/gm, '')
      // Añadir separacion antes de secciones principales
      .replace(/\n(Puntos clave)\n/g, '\n\n$1\n')
      .replace(/\n(Tareas)\n/g, '\n\n$1\n')
      // Reducir multiples saltos de linea a maximo 2
      .replace(/\n{3,}/g, '\n\n')
      // Limpiar espacios al inicio de cada linea
      .replace(/^[ \t]+/gm, '')
      .trim();

    return text;
  };

  // Función para copiar el resumen al portapapeles
  // Copia todo el texto visible del contenedor preservando el formato
  const copySummaryToClipboard = async () => {
    if (!summaryContainerRef.current) return;

    // Clonar el contenedor para no modificar el original
    const clone = summaryContainerRef.current.cloneNode(true) as HTMLElement;

    // Eliminar el botón "Copiar" del clon (y cualquier otro botón)
    const buttons = clone.querySelectorAll('button');
    buttons.forEach(btn => btn.remove());

    // Añadir temporalmente al DOM para que innerText funcione correctamente
    clone.style.position = 'absolute';
    clone.style.left = '-9999px';
    clone.style.top = '-9999px';
    document.body.appendChild(clone);

    // Extraer texto limpio
    const fullText = extractCleanText(clone);

    // Eliminar el clon del DOM
    document.body.removeChild(clone);

    if (!fullText.trim()) return;

    try {
      await navigator.clipboard.writeText(fullText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Error al copiar al portapapeles:', err);
      // Fallback para navegadores que no soportan clipboard API
      const textArea = document.createElement('textarea');
      textArea.value = fullText;
      textArea.style.position = 'fixed';
      textArea.style.left = '-999999px';
      document.body.appendChild(textArea);
      textArea.select();
      try {
        document.execCommand('copy');
        setCopied(true);
        setTimeout(() => setCopied(false), 2000);
      } catch (fallbackErr) {
        console.error('Error en fallback de copia:', fallbackErr);
      }
      document.body.removeChild(textArea);
    }
  };

  return (
    <div ref={summaryContainerRef} className="space-y-6">
      {/* Resumen ejecutivo */}
      {toonText && (
        <div>
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-slate-50">Resumen ejecutivo</h3>
            <button
              onClick={copySummaryToClipboard}
              className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-gray-700 dark:text-slate-300 bg-white dark:bg-slate-800 border border-gray-300 dark:border-slate-600 rounded-lg hover:bg-gray-50 dark:hover:bg-slate-700 transition-colors"
              title="Copiar resumen completo al portapapeles"
            >
              {copied ? (
                <>
                  <CheckIcon className="h-4 w-4 text-green-600 dark:text-green-400" />
                  <span className="text-green-600 dark:text-green-400">Copiado</span>
                </>
              ) : (
                <>
                  <ClipboardDocumentIcon className="h-4 w-4" />
                  <span>Copiar</span>
                </>
              )}
            </button>
          </div>
          <div
            className="text-gray-700 dark:text-slate-300 leading-relaxed prose prose-sm dark:prose-invert max-w-none"
            dangerouslySetInnerHTML={{
              __html: processHtmlForTooltips(
                convertAllColorMarkers(
                  (
                    colorizeSpeakerNamesInMarkdown(
                      toonText
                        // Limpiar evidencia entre paréntesis con guiones bajos (formato: _(evid: ...)_)
                        .replace(/_\s*\(evid:[^)]+\)\s*_/g, '')
                        // Limpiar evidencia entre paréntesis sin guiones bajos
                        .replace(/\s*\(evid:[^)]+\)/g, '')
                        // Reemplazar "resp" por "Responsividad" para mayor claridad
                        .replace(/\bresp\s+(\d+\/\d+)/g, 'Responsividad $1')
                        // Cambiar "Carga y papeles" por "Carga y Roles"
                        .replace(/Carga y papeles/gi, 'Carga y Roles')
                        // Añadir espacio entre el nombre del responsable principal y la lista de participantes
                        .replace(/(Responsable principal \(mayor carga\):\s*[^\n]+)\n(-\s)/g, '$1\n\n$2')
                    )
                      // Formatear markdown básico (convertir primero los que tienen COLOR en negrita)
                      .replace(/\*\*\{\{COLOR:([^:]+):([^}]+)\}\}\*\*/g, '<strong style="color: $1">$2</strong>')
                      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
                      .replace(/\*(.*?)\*/g, '<em>$1</em>')
                      .replace(/## (.*?)\n/g, '<h2 class="text-base font-semibold mt-4 mb-2">$1</h2>')
                      .replace(/### (.*?)\n/g, '<h3 class="text-sm font-semibold mt-3 mb-1">$1</h3>')
                      .replace(/\n/g, '<br />')
                  )
                    // Añadir tooltips a roles de participantes
                    // Después de la conversión a HTML, el formato es: </strong>: Driver — o simplemente : Driver —
                    // Buscar el patrón "Role —" (con espacio y guión largo)
                    .replace(/Driver\s*—/g, '<span class="tooltip-trigger" title="Driver: Lidera la conversacion. Es el participante con mayor porcentaje de participacion.">Driver</span> —')
                    .replace(/Co-driver\s*—/g, '<span class="tooltip-trigger" title="Co-driver: Co-lidera la conversacion. Segundo participante con alta participacion y diferencia pequena con el Driver.">Co-driver</span> —')
                    .replace(/Contributor\s*—/g, '<span class="tooltip-trigger" title="Contributor: Contribuye activamente a la conversacion. Participacion significativa.">Contributor</span> —')
                    .replace(/Assistant\s*—/g, '<span class="tooltip-trigger" title="Assistant: Participacion limitada en la conversacion. Menor participacion.">Assistant</span> —')
                    // Porcentajes (XX.X%) - usar tooltip-trigger para que se ajuste dinámicamente
                    .replace(/(\d+\.?\d*)%/g, '<span class="tooltip-trigger" title="Porcentaje del tiempo total de la reunión que habló este participante">$1%</span>')
                    // Segundos (YYYs) - usar tooltip-trigger para que se ajuste dinámicamente
                    .replace(/(\d+)s\b/g, '<span class="tooltip-trigger" title="Tiempo total de habla en segundos">$1s</span>')
                    // Turnos - alineado a derecha porque están al final
                    .replace(/(\d+)\s+turnos/g, '<span class="tooltip-trigger-right" title="Número de veces que este participante tomó la palabra durante la reunión">$1 turnos</span>')
                    // Responsividad - alineado a derecha porque está al final
                    .replace(/Responsividad\s+(\d+\/\d+)/g, '<span class="tooltip-trigger-right" title="Mide qué tan rápido y frecuentemente responde el participante. 100/100 = siempre responde, valores más bajos indican menor reactividad">Responsividad $1</span>')
                    // Colaboración
                    .replace(/Colaboración:\s*(\d+\/\d+)/g, '<span class="tooltip-trigger" title="Nivel de colaboración entre participantes. Valores altos indican trabajo en equipo efectivo">Colaboración: $1</span>')
                    // Decisión
                    .replace(/Decisión:\s*(\d+\/\d+)/g, '<span class="tooltip-trigger" title="Capacidad de la reunión para tomar decisiones. Valores altos indican reuniones más decisivas">Decisión: $1</span>')
                    // Conflicto
                    .replace(/Conflicto:\s*(\d+\/\d+)/g, '<span class="tooltip-trigger" title="Nivel de conflicto o tensión detectado. Valores altos indican más desacuerdos o tensiones">Conflicto: $1</span>')
                    // Limpiar metadatos de tareas: _(due: ...)_
                    .replace(/\s*_\(due:[^)]+\)\s*_/g, '')
                    // Añadir tooltip a conf (confidence) en Plan / próximos pasos
                    .replace(/\[conf\s+([\d.]+)\]/g, '<span class="tooltip-trigger" title="Confianza en la precisión de este punto. Valor entre 0 y 1, donde 1 = máxima confianza">[conf $1]</span>')
                    // Clima - reemplazar con componente visual (valores de -1 a 1, donde 0 es neutro)
                    .replace(/Clima:\s*valence\s*(-?[\d.]+);\s*labels\s*([^<\n]+)/g, (_match: string, valence: string, labels: string) => {
                      const valenceNum = parseFloat(valence);
                      let sentiment = 'neutro';
                      let colorClass = 'bg-yellow-500';
                      let textColor = 'text-yellow-700 dark:text-yellow-400';

                      if (valenceNum > 0.3) {
                        sentiment = 'positivo';
                        colorClass = 'bg-green-500';
                        textColor = 'text-green-700 dark:text-green-400';
                      } else if (valenceNum < -0.3) {
                        sentiment = 'negativo';
                        colorClass = 'bg-red-500';
                        textColor = 'text-red-700 dark:text-red-400';
                      }

                      const labelsList = labels.split(',').map((l: string) => l.trim()).filter((l: string) => l);
                      const labelsHtml = labelsList.map((label: string) =>
                        `<span class="px-2 py-1 text-xs rounded bg-gray-200 dark:bg-slate-700 text-gray-700 dark:text-slate-300 ml-2">${label}</span>`
                      ).join('');

                      return `<div class="flex items-center flex-wrap gap-2 mt-2">
                    <span class="tooltip-trigger ${textColor} font-semibold" title="Sentimiento general de la reunion. Valence: ${valence} (${sentiment}). Rango: -1 (muy negativo) a 1 (muy positivo), 0 = neutro.">
                      Clima: <span class="inline-flex items-center gap-1"><span class="inline-block w-3 h-3 rounded-full ${colorClass}"></span><span class="capitalize">${sentiment}</span> (${valence})</span>
                    </span>
                    ${labelsHtml}
                  </div>`;
                    })
                    // Limpiar Quality flags (solo informativo, no afecta métricas)
                    .replace(/### Quality flags:.*$/gm, '')
                )
              )
            }}
          />
        </div>
      )}

      {/* Puntos clave */}
      {keyPoints.length > 0 && (
        <div>
          <h4 className="text-md font-semibold text-gray-900 dark:text-slate-50 mb-3">
            Puntos clave
            <span
              className="tooltip-trigger ml-2 text-xs text-gray-500 dark:text-slate-400"
              data-tooltip-text="Temas, decisiones o puntos importantes identificados durante la reunión"
            >
              ℹ️
            </span>
          </h4>
          <ul className="space-y-2">
            {keyPoints.slice(0, 10).map((point, index) => {
              // Detectar si el punto menciona porcentajes, tiempos, o métricas
              const hasPercentage = /\d+\.?\d*%/.test(point);
              const hasTime = /\d+s\b/.test(point);
              const hasTurns = /\d+\s+turnos/.test(point);
              const hasConf = /\[conf\s+[\d.]+\]/.test(point);

              let tooltipText = '';
              if (hasPercentage) {
                tooltipText += 'Porcentaje de participación. ';
              }
              if (hasTime) {
                tooltipText += 'Tiempo de habla en segundos. ';
              }
              if (hasTurns) {
                tooltipText += 'Número de turnos de habla. ';
              }
              if (hasConf) {
                tooltipText += 'Confianza en la precisión de este punto. ';
              }

              // Convertir markdown a HTML primero (negritas **texto** -> <strong>texto</strong>)
              let formattedPoint = point
                .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

              // Luego colorear nombres de participantes (solo en texto que no esté dentro de <strong>)
              let coloredPoint = formattedPoint;
              allSpeakers.forEach((speaker) => {
                const speakerColor = getSpeakerColor(speaker, allSpeakers);
                const escapedName = speaker.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                // Colorear nombres que no estén dentro de tags <strong>
                // Dividir por <strong> y </strong> y solo colorear fuera de ellos
                const parts = coloredPoint.split(/(<strong>.*?<\/strong>)/g);
                coloredPoint = parts.map((part) => {
                  if (part.startsWith('<strong>') && part.endsWith('</strong>')) {
                    return part; // No colorear dentro de <strong>
                  }
                  return part.replace(
                    new RegExp(`(${escapedName})`, 'gi'),
                    `<span style="color: ${speakerColor.main}; font-weight: 500">$1</span>`
                  );
                }).join('');
              });

              return (
                <li key={index} className="flex items-start">
                  <span className="text-primary-600 dark:text-primary-400 mr-2">•</span>
                  <span
                    className={`text-gray-700 dark:text-slate-300 ${tooltipText ? 'tooltip-trigger' : ''}`}
                    data-tooltip-text={tooltipText || undefined}
                    dangerouslySetInnerHTML={{
                      __html: coloredPoint.replace(/\[conf\s+([\d.]+)\]/g, (_match: string, conf: string) =>
                        `<span class="tooltip-trigger" data-tooltip-text="Confianza en la precisión de este punto. Valor entre 0 y 1, donde 1 = máxima confianza">[conf ${conf}]</span>`
                      )
                    }}
                  />
                </li>
              );
            })}
          </ul>
        </div>
      )}

      {/* Tareas - siempre se muestra */}
      <div>
        <h4 className="text-md font-semibold text-gray-900 dark:text-slate-50 mb-3">Tareas</h4>
        <div className="space-y-2">
          {actionItems.length > 0 ? (
            actionItems.map((item: any, index: number) => (
              <div
                key={index}
                className="flex items-start p-3 rounded-lg bg-gray-50 dark:bg-slate-700"
              >
                <ClockIcon className="h-5 w-5 text-gray-400 dark:text-slate-500 mr-2 mt-0.5" />
                <div className="flex-1">
                  <p className="text-sm text-gray-900 dark:text-slate-50">{item.task}</p>
                  {item.owner && (
                    <p className="text-xs text-gray-500 dark:text-slate-400 mt-1">Asignado a: {item.owner}</p>
                  )}
                  {item.due_date && (
                    <p className="text-xs text-gray-500 dark:text-slate-400">Vence: {item.due_date}</p>
                  )}
                </div>
              </div>
            ))
          ) : (
            <div className="flex items-start p-3 rounded-lg bg-gray-50 dark:bg-slate-700">
              <ClockIcon className="h-5 w-5 text-gray-400 dark:text-slate-500 mr-2 mt-0.5" />
              <p className="text-sm text-gray-500 dark:text-slate-400 italic">Sin tareas asignadas</p>
            </div>
          )}
        </div>
      </div>

      {/* Mensaje si no hay datos */}
      {!toonText && keyPoints.length === 0 && actionItems.length === 0 && (
        <div className="text-center py-8">
          <p className="text-gray-600 dark:text-slate-400">No hay datos de resumen disponibles</p>
        </div>
      )}
    </div>
  );
};
