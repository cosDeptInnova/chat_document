import type { IASummary, IAInsights, IAAnalysis } from '../types';

export const mockSummary: IASummary = {
  summary: `Esta reunión abordó temas importantes relacionados con el desarrollo del proyecto Notetaker. 
  Los participantes discutieron las funcionalidades principales, la arquitectura del sistema y los próximos pasos. 
  Se llegó a varios acuerdos sobre la implementación de nuevas características y mejoras en la experiencia del usuario.`,
  key_points: [
    'Implementación de sistema de transcripción en tiempo real',
    'Integración con Microsoft Teams mediante Recall.ai',
    'Desarrollo de panel de análisis con IA',
    'Mejoras en la gestión de permisos y accesos',
    'Optimización del rendimiento del sistema',
  ],
  action_items: [
    {
      id: '1',
      text: 'Revisar documentación de Recall.ai API',
      assignee: 'Juan Pérez',
      due_date: '2025-12-20',
      completed: false,
    },
    {
      id: '2',
      text: 'Implementar sistema de notificaciones',
      assignee: 'María García',
      due_date: '2025-12-22',
      completed: false,
    },
    {
      id: '3',
      text: 'Configurar entorno de pruebas',
      assignee: 'Carlos López',
      due_date: '2025-12-21',
      completed: true,
    },
  ],
};

export const mockInsights: IAInsights = {
  sentiment: 'positive',
  participation: [
    {
      speaker: 'Juan Pérez',
      percentage: 35,
      speaking_time: 420,
    },
    {
      speaker: 'María García',
      percentage: 28,
      speaking_time: 336,
    },
    {
      speaker: 'Carlos López',
      percentage: 22,
      speaking_time: 264,
    },
    {
      speaker: 'Ana Martínez',
      percentage: 15,
      speaking_time: 180,
    },
  ],
  topics: [
    { topic: 'Desarrollo de software', mentions: 12 },
    { topic: 'Arquitectura del sistema', mentions: 8 },
    { topic: 'Mejoras de rendimiento', mentions: 6 },
    { topic: 'Experiencia de usuario', mentions: 5 },
    { topic: 'Integración con APIs', mentions: 4 },
  ],
};

export const mockAnalysis: IAAnalysis = {
  total_duration: 1200,
  speakers_count: 4,
  average_speaking_time: 300,
  engagement_score: 85,
  decision_points: [
    {
      timestamp: 180,
      description: 'Decisión sobre arquitectura del sistema',
    },
    {
      timestamp: 450,
      description: 'Aprobación de nuevas funcionalidades',
    },
    {
      timestamp: 780,
      description: 'Asignación de tareas y responsabilidades',
    },
    {
      timestamp: 1050,
      description: 'Definición de próximos pasos',
    },
  ],
};

