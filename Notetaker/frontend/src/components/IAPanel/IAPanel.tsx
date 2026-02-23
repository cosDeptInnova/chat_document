import { useState, useEffect, useCallback } from 'react';
import { SummaryTab } from './SummaryTab';
import { InsightsTab } from './InsightsTab';
import { AnalysisTab } from './AnalysisTab';
import { meetingsApi } from '../../services/api';

interface IAPanelProps {
  meetingId: string;
  userEmail?: string;
}

interface SummaryData {
  status: 'not_available' | 'pending' | 'processing' | 'completed' | 'failed';
  message?: string;
  error?: string;
  toon?: string;
  toon_transcript?: string;
  normalized?: any;
  insights?: {
    participation_percent?: Record<string, number>;
    talk_time_seconds?: Record<string, number>;
    turns?: Record<string, number>;
    collaboration?: { score_0_100: number; notes?: string[] };
    atmosphere?: { valence: number; labels?: string[]; notes?: string[] };
    decisiveness?: { score_0_100: number; notes?: string[] };
    conflict_level_0_100?: number;
    topics?: Array<{ name: string; weight?: number }> | string[];
    decisions?: string[];
    action_items?: Array<{ owner: string | null; task: string; due_date: string | null; confidence: number }>;
    summary?: string;
  };
  processing_time_seconds?: number;
  completed_at?: string;
  queue_estimation?: {
    queue_position: number;
    meetings_ahead: number;
    average_processing_time: number;
    estimated_wait_seconds: number;
    estimated_processing_seconds: number;
  };
}

export const IAPanel: React.FC<IAPanelProps> = ({ meetingId, userEmail }) => {
  const [activeTab, setActiveTab] = useState<'summary' | 'insights' | 'analysis'>('summary');
  const [summaryData, setSummaryData] = useState<SummaryData | null>(null);
  const [loading, setLoading] = useState(true);
  const [polling, setPolling] = useState(false);

  // Función para obtener datos de summary
  const fetchSummary = useCallback(async () => {
    try {
      const data = await meetingsApi.getSummary(meetingId, userEmail);
      setSummaryData(data);
      
      // Si está pendiente o procesando, activar polling
      if (data.status === 'pending' || data.status === 'processing') {
        setPolling(true);
      } else {
        setPolling(false);
      }
    } catch (error: any) {
      console.error('Error obteniendo summary:', error);
      // Si es 403, mostrar mensaje de acceso denegado
      if (error?.response?.status === 403) {
        setSummaryData({
          status: 'not_available',
          message: error?.response?.data?.detail || 'No tienes acceso a los datos de esta reunión',
        });
      } else {
        setSummaryData({
          status: 'not_available',
          message: 'Error al cargar el análisis',
        });
      }
    } finally {
      setLoading(false);
    }
  }, [meetingId, userEmail]);

  // Cargar datos al montar
  useEffect(() => {
    if (meetingId) {
      fetchSummary();
    }
  }, [meetingId, fetchSummary]);

  // Polling cada 10 segundos si está procesando
  useEffect(() => {
    if (!polling) return;

    const interval = setInterval(() => {
      fetchSummary();
    }, 10000); // 10 segundos

    return () => clearInterval(interval);
  }, [polling, fetchSummary]);

  // Ajustar tooltips dinámicamente para evitar que se corten
  useEffect(() => {
    const tooltipWidth = 220; // Ancho del tooltip
    const margin = 20; // Margen mínimo desde el borde
    
    const adjustTooltip = (element: HTMLElement) => {
      const rect = element.getBoundingClientRect();
      const spaceOnLeft = rect.left;
      const spaceOnRight = window.innerWidth - rect.right;
      const spaceOnTop = rect.top;
      const spaceOnBottom = window.innerHeight - rect.bottom;
      const isRight = element.classList.contains('tooltip-trigger-right');
      const tooltipHeight = 100; // Altura aproximada del tooltip (aumentada para tooltips largos)
      
      // Detectar si hay poco espacio arriba (reducir umbral para activarse antes)
      const needsBottomPosition = spaceOnTop < tooltipHeight + margin;
      const hasSpaceBelow = spaceOnBottom >= tooltipHeight + margin;
      
      // Si está cerca del borde superior, usar position fixed para evitar cortes por overflow
      // Usar un umbral más bajo (60px) para activarse antes y evitar cortes
      if (spaceOnTop < 60) {
        // Si hay poco espacio arriba, mostrar tooltip debajo si hay espacio
        if (needsBottomPosition && hasSpaceBelow) {
          element.classList.add('tooltip-bottom');
          // Calcular posición debajo del elemento
          const tooltipTop = rect.bottom + 10;
          element.classList.add('tooltip-fixed');
          element.style.setProperty('--tooltip-top', `${tooltipTop}px`);
        } else {
          // No hay espacio abajo, ponerlo arriba pero con fixed para evitar cortes
          element.classList.remove('tooltip-bottom');
          const tooltipTop = Math.max(margin, rect.top - tooltipHeight - 10);
          element.classList.add('tooltip-fixed');
          element.style.setProperty('--tooltip-top', `${tooltipTop}px`);
        }
        // Posicionar horizontalmente
        if (isRight) {
          const tooltipLeft = Math.max(margin, Math.min(rect.right - tooltipWidth, window.innerWidth - tooltipWidth - margin));
          element.style.setProperty('--tooltip-left', `${tooltipLeft}px`);
        } else {
          element.style.setProperty('--tooltip-left', `${Math.max(margin, rect.left)}px`);
        }
        element.style.removeProperty('--tooltip-offset');
        return; // Salir temprano ya que usamos fixed
      } else {
        // Hay suficiente espacio arriba, usar posición normal
        element.classList.remove('tooltip-bottom');
      }
      
      // Si es tooltip-right y hay poco espacio a la izquierda, cambiar a izquierdo
      if (isRight && spaceOnLeft < tooltipWidth + margin && spaceOnRight >= tooltipWidth + margin) {
        element.classList.remove('tooltip-trigger-right');
        element.classList.add('tooltip-trigger');
        // Recalcular después del cambio
        const newRect = element.getBoundingClientRect();
        const newSpaceOnLeft = newRect.left;
        if (newSpaceOnLeft < margin) {
          // Usar position fixed para evitar cortes
          const tooltipTop = newRect.top - tooltipHeight - 10;
          element.classList.add('tooltip-fixed');
          element.style.setProperty('--tooltip-top', `${Math.max(margin, tooltipTop)}px`);
          element.style.setProperty('--tooltip-left', `${margin}px`);
          element.style.removeProperty('--tooltip-offset');
        } else {
          element.classList.remove('tooltip-fixed');
          element.style.removeProperty('--tooltip-top');
          element.style.removeProperty('--tooltip-left');
          element.style.removeProperty('--tooltip-offset');
        }
      } else if (!isRight) {
        // Para tooltips izquierdos, calcular si se salen por la izquierda
        // El tooltip se posiciona con left: 0 relativo al elemento
        // Si el elemento está muy cerca del borde izquierdo de la ventana, el tooltip se saldrá
        if (spaceOnLeft < margin) {
          // Usar position fixed para evitar que se corte por overflow del contenedor
          // Calcular posición del tooltip: aparece arriba del elemento
          const tooltipTop = rect.top - tooltipHeight - 10;
          element.classList.add('tooltip-fixed');
          element.style.setProperty('--tooltip-top', `${Math.max(margin, tooltipTop)}px`);
          element.style.setProperty('--tooltip-left', `${margin}px`);
          element.style.removeProperty('--tooltip-offset');
        } else {
          // Si hay suficiente espacio, usar position absolute normal
          element.classList.remove('tooltip-fixed');
          element.style.removeProperty('--tooltip-top');
          element.style.removeProperty('--tooltip-left');
          // Verificar si necesita offset menor
          if (spaceOnLeft < tooltipWidth + margin) {
            const overflow = margin - spaceOnLeft;
            element.style.setProperty('--tooltip-offset', `${overflow}px`);
          } else {
            element.style.removeProperty('--tooltip-offset');
          }
        }
      }
    };
    
    const adjustTooltips = () => {
      // Ajustar todos los tooltips
      const allTooltips = document.querySelectorAll('.tooltip-trigger, .tooltip-trigger-right');
      allTooltips.forEach((element) => {
        adjustTooltip(element as HTMLElement);
      });
    };

    // Ajustar tooltips después de que el contenido se renderice
    const timeoutId = setTimeout(adjustTooltips, 300);
    
    // Ajustar tooltips cuando se hace scroll o se redimensiona la ventana
    const handleScroll = () => adjustTooltips();
    const handleResize = () => adjustTooltips();
    window.addEventListener('scroll', handleScroll, true);
    window.addEventListener('resize', handleResize);
    
    // Ajustar tooltips cuando se cambia de tab o se actualiza el contenido
    const observer = new MutationObserver(() => {
      setTimeout(adjustTooltips, 100);
    });
    const container = document.querySelector('.bg-white.dark\\:bg-slate-800');
    if (container) {
      observer.observe(container, { childList: true, subtree: true, attributes: true, characterData: true });
    }
    
    // Ajustar tooltip específico cuando se hace hover (para detectar en tiempo real)
    const handleMouseOver = (e: Event) => {
      const target = e.target as HTMLElement;
      if (!target) return;
      
      // Buscar el elemento tooltip (puede ser el target o un hijo)
      let tooltipElement: HTMLElement | null = null;
      if (target.classList.contains('tooltip-trigger') || target.classList.contains('tooltip-trigger-right')) {
        tooltipElement = target;
      } else {
        // Buscar el elemento tooltip más cercano
        tooltipElement = target.closest('.tooltip-trigger, .tooltip-trigger-right') as HTMLElement;
      }
      
      if (tooltipElement) {
        // Ajustar este tooltip específico inmediatamente
        requestAnimationFrame(() => {
          adjustTooltip(tooltipElement!);
        });
      }
    };
    
    // Usar delegación de eventos en el contenedor para capturar todos los tooltips
    if (container) {
      container.addEventListener('mouseover', handleMouseOver, true);
    }
    document.addEventListener('mouseover', handleMouseOver, true);

    return () => {
      clearTimeout(timeoutId);
      window.removeEventListener('scroll', handleScroll, true);
      window.removeEventListener('resize', handleResize);
      if (container) {
        container.removeEventListener('mouseover', handleMouseOver, true);
      }
      document.removeEventListener('mouseover', handleMouseOver, true);
      observer.disconnect();
    };
  }, [activeTab, summaryData]);

  return (
    <div className="bg-white dark:bg-slate-800 rounded-lg shadow transition-colors flex flex-col h-full">
      <div className="flex-shrink-0 border-b border-gray-200 dark:border-slate-700">
        <nav className="flex">
          <button
            onClick={() => setActiveTab('summary')}
            className={`flex-1 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'summary'
                ? 'border-primary-500 dark:border-primary-400 text-primary-600 dark:text-primary-400'
                : 'border-transparent text-gray-500 dark:text-slate-400 hover:text-gray-700 dark:hover:text-slate-300'
            }`}
          >
            Resumen
          </button>
          <button
            onClick={() => setActiveTab('insights')}
            className={`flex-1 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'insights'
                ? 'border-primary-500 dark:border-primary-400 text-primary-600 dark:text-primary-400'
                : 'border-transparent text-gray-500 dark:text-slate-400 hover:text-gray-700 dark:hover:text-slate-300'
            }`}
          >
            Insights
          </button>
          <button
            onClick={() => setActiveTab('analysis')}
            className={`flex-1 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
              activeTab === 'analysis'
                ? 'border-primary-500 dark:border-primary-400 text-primary-600 dark:text-primary-400'
                : 'border-transparent text-gray-500 dark:text-slate-400 hover:text-gray-700 dark:hover:text-slate-300'
            }`}
          >
            Análisis
          </button>
        </nav>
      </div>

      <div className="flex-1 min-h-0 overflow-y-auto p-6">
        {activeTab === 'summary' && <SummaryTab summaryData={summaryData} loading={loading} />}
        {activeTab === 'insights' && <InsightsTab summaryData={summaryData} loading={loading} />}
        {activeTab === 'analysis' && <AnalysisTab summaryData={summaryData} loading={loading} />}
      </div>
      
      {/* Estilos globales para tooltips - aparecen ARRIBA para evitar cortes */}
      <style>{`
        .tooltip-trigger {
          position: relative;
          cursor: help;
          border-bottom: 1px dotted currentColor;
          text-decoration: none;
        }
        .tooltip-trigger:hover::after,
        .tooltip-trigger-right:hover::after {
          content: attr(data-tooltip-text);
          position: absolute;
          bottom: calc(100% + 10px);
          padding: 8px 12px;
          background-color: #1f2937;
          color: white;
          border-radius: 6px;
          font-size: 0.75rem;
          white-space: normal;
          width: 220px;
          max-width: min(220px, calc(100vw - 40px));
          z-index: 50000;
          box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
          pointer-events: none;
          word-wrap: break-word;
          box-sizing: border-box;
        }
        /* Tooltip que aparece debajo cuando no hay espacio arriba */
        .tooltip-trigger.tooltip-bottom:hover::after,
        .tooltip-trigger-right.tooltip-bottom:hover::after {
          bottom: auto;
          top: calc(100% + 10px);
        }
        /* Tooltip alineado a la izquierda (para elementos a la izquierda) */
        .tooltip-trigger:hover::after {
          left: 0;
          right: auto;
          transform: translateX(var(--tooltip-offset, 0px));
        }
        /* Tooltip con position fixed cuando está cerca del borde (para evitar cortes por overflow) */
        .tooltip-trigger.tooltip-fixed:hover::after,
        .tooltip-trigger-right.tooltip-fixed:hover::after {
          position: fixed !important;
          left: var(--tooltip-left, 0) !important;
          top: var(--tooltip-top, auto) !important;
          bottom: auto !important;
          transform: translateX(0) !important;
        }
        /* Tooltip alineado a la derecha (para elementos a la derecha) */
        .tooltip-trigger-right {
          position: relative;
          cursor: help;
          border-bottom: 1px dotted currentColor;
          text-decoration: none;
        }
        .tooltip-trigger-right:hover::after {
          left: auto;
          right: 0;
          transform: translateX(0);
        }
        .dark .tooltip-trigger:hover::after,
        .dark .tooltip-trigger-right:hover::after {
          background-color: #374151;
        }
        /* Flechita para tooltip izquierdo */
        .tooltip-trigger:hover::before {
          content: '';
          position: absolute;
          bottom: calc(100% + 5px);
          left: calc(10px + var(--tooltip-offset, 0px));
          right: auto;
          border: 5px solid transparent;
          border-top-color: #1f2937;
          z-index: 50001;
        }
        /* Flechita para tooltip derecho */
        .tooltip-trigger-right:hover::before {
          content: '';
          position: absolute;
          bottom: calc(100% + 5px);
          left: auto;
          right: 10px;
          border: 5px solid transparent;
          border-top-color: #1f2937;
          z-index: 50001;
        }
        /* Flechita para tooltips que aparecen debajo */
        .tooltip-trigger.tooltip-bottom:hover::before,
        .tooltip-trigger-right.tooltip-bottom:hover::before {
          bottom: auto;
          top: calc(100% + 5px);
          border-top-color: transparent;
          border-bottom-color: #1f2937;
        }
        .dark .tooltip-trigger:hover::before,
        .dark .tooltip-trigger-right:hover::before {
          border-top-color: #374151;
        }
        .dark .tooltip-trigger.tooltip-bottom:hover::before,
        .dark .tooltip-trigger-right.tooltip-bottom:hover::before {
          border-top-color: transparent;
          border-bottom-color: #374151;
        }
        /* Ocultar tooltip nativo del navegador */
        .tooltip-trigger[title]:hover::after {
          content: attr(title);
        }
        .tooltip-trigger-right[title]:hover::after {
          content: attr(title);
        }
      `}</style>
      
    </div>
  );
};

