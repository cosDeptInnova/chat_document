import {
  ChatBubbleBottomCenterTextIcon,
  BoltIcon,
  DocumentMagnifyingGlassIcon,
  CloudArrowUpIcon,
  MapIcon,
  GlobeAltIcon,
  ChartBarIcon,
  BookOpenIcon,
} from "@heroicons/react/24/solid";

// Imports de las imagenes de los AGEN-TICS
import agenTicSoporteImg from "../images/agenTicSoporte.PNG";
import agenTicPredictivoImg from "../images/agenTicPredictivo.PNG";
import agenTicNotificadorImg from "../images/agenTicNotificador.PNG";
import agenTicSupervisorImg from "../images/agenTicSupervisor.PNG";
import agenTicAnalistaImg from "../images/agenTicAnalista.PNG";
import agenTicLogisticoImg from "../images/agenTicLogistico.PNG";
import agenTicFormadorImg from "../images/agenTicFormador.PNG";
import { PencilSquareIcon } from "@heroicons/react/24/solid";

// Aquí van los datos de los casos de uso

export const useCases = [
  {
    title: "Consultas IA Cosmos",
    description:
      "Utiliza un equipo de agentes de Cosmos avanzado para hablar con el conocimiento que le cargues",
    departments: ["Más usados", "RRHH", "Administración", "Comunicación", "Comercial"],
    icon: ChatBubbleBottomCenterTextIcon,
    tab: "nuevo", // para que al abrir la pestaña se seleccione Nuevo Chat
    popularity: 10,
    // engine: "modelo", // implícito por defecto
    // initialMessage: "Hola, ¿en qué puedo ayudarte?", // opcional
  },
  /*
  {
    title: "Consultas IA Cosmos sobre conocimiento activo",
    description:
      "Consulta información sobre todos los documentos cargados en la plataforma",
    departments: ["Más usados", "RRHH", "Administración", "Comunicación", "Comercial"],
    icon: BoltIcon,
    tab: "nuevo",
    popularity: 9,
    initialMessage: "Hola, ¿sobre qué documento quieres consultar información?",
    // engine: "modelo",
  },
  {
    title: "Carga de archivos",
    description: "Sube archivos a tu directorio personal o departamental",
    departments: ["Más usados", "RRHH", "Administración", "Comunicación", "Comercial"],
    icon: CloudArrowUpIcon,
    tab: "conocimientos",
    popularity: 8,
  },
  {
    title: "Planificador de rutas GPS",
    description: "Consulta la mejor ruta en el momento",
    departments: ["Más usados", "Administración"],
    icon: MapIcon,
    tab: "nuevo",
    popularity: 5,
    initialMessage: "Hola, ¿con qué ruta GPS puedo ayudarte?",
  },
  */
  {
    title: "Comparador de documentos",
    description: "Perfecto para comparar archivos de texto. Carga dos archivos y deja que Cosmos descubra modificaciones entre ambos textos por ti",
    departments: ["Más usados", "RRHH", "Administración", "Comunicación", "Comercial"],
    icon: DocumentMagnifyingGlassIcon,
    tab: "nuevo",
    href: "/main/text-compare",
    popularity: 7,
    initialMessage: "Hola, ¿qué documentos quieres que compare?",
  },
    {
    title: "Notetaker",
    description:
      "Abre Notetaker con SSO (se crea sesión automáticamente) y ve directo al dashboard.",
    departments: ["Más usados", "RRHH", "Administración", "Comunicación", "Comercial"],
    icon: PencilSquareIcon,
    popularity: 8,

    // NUEVO: acción especial (no navega por router)
    action: "notetaker",
  },

  /*
  {
    title: "Monitoreo y gráficas en tiempo real de la herramienta",
    description: "Consulta estadísticas en tiempo real",
    departments: ["Más usados", "Comercial"],
    icon: ChartBarIcon,
    tab: "nuevo",
    popularity: 5,
    initialMessage: "Hola, ¿qué gráficas deseas consultar?",
  },
  */
  {
    title: "Búsqueda web",
    description: "Utiliza al equipo agentes virtuales de Cosmos para realizar búsquedas web avanzadas",
    departments: ["Más usados", "RRHH", "Administración", "Comunicación", "Comercial"],
    icon: GlobeAltIcon,
    tab: "nuevo",
    popularity: 7,
    engine: "web",
    initialMessage: "Hola, ¿qué información quieres que consulte en la web por ti?",
  },
  {
    title: "Exploración legal avanzada",
    description: "Investiga normativa, jurisprudencia y criterios con un flujo legal especializado y fuentes trazables",
    departments: ["Más usados", "RRHH", "Administración", "Comunicación", "Comercial"],
    icon: GlobeAltIcon,
    tab: "nuevo",
    popularity: 7,
    engine: "legal_explorer",
    initialMessage: "Hola, ¿qué cuestión legal quieres explorar en profundidad?",
  },
  {
    title: "Hablar con documentos",
    description: "Sube cualquier archivo de texto grande e interactúa con ello mediante una de las crews de Cosmos",
    departments: ["Más usados", "RRHH", "Administración", "Comunicación", "Comercial"],
    icon: BookOpenIcon,
    tab: "nuevo",
    popularity: 6,
    initialMessage: "Hola, ¿de qué documento quieres que hablemos?",
    // Aquí marcamos explícitamente que este caso de uso usa el motor chatdoc
    engine: "chatdoc",
  },

  // AGEN-TICS con imagen.
  /*
  {
    title: "AGEN-TIC Soporte",
    description: "Ofrece servicios de soporte",
    departments: ["Más usados", "Agen-tic"],
    image: agenTicSoporteImg,
    tab: "nuevo",
    popularity: 10,
    initialMessage:
      "Hola, soy tu agente de soporte ¿con qué puedo ayudarte?",
  },
  {
    title: "AGEN-TIC Predictivo",
    description: "Ofrece servicios de predicción",
    departments: ["Más usados", "Agen-tic"],
    image: agenTicPredictivoImg,
    tab: "nuevo",
    popularity: 6,
    initialMessage:
      "Hola, soy tu agente predictivo ¿quieres que revise algún sistema en busca de posibles fallos?",
  },
  {
    title: "AGEN-TIC Notificador",
    description: "Ofrece servicios de notificaciones",
    departments: ["Más usados", "Agen-tic"],
    image: agenTicNotificadorImg,
    tab: "nuevo",
    popularity: 7,
    initialMessage:
      "Hola, soy tu agente notificador ¿quieres que notifique algún evento?",
  },
  {
    title: "AGEN-TIC Supervisor",
    description: "Ofrece servicios de supervisión",
    departments: ["Más usados", "Agen-tic"],
    image: agenTicSupervisorImg,
    tab: "nuevo",
    popularity: 6,
    initialMessage:
      "Hola, soy tu agente supervisor ¿quieres que supervise alguna tarea o proceso?",
  },
  {
    title: "AGEN-TIC Analista",
    description: "Ofrece servicios de análisis",
    departments: ["Más usados", "Agen-tic"],
    image: agenTicAnalistaImg,
    tab: "nuevo",
    popularity: 5,
    initialMessage:
      "Hola, soy tu agente analista ¿qué quieres que analice por ti?",
  },
  {
    title: "AGEN-TIC Logístico",
    description: "Ofrece servicios de logística",
    departments: ["Más usados", "Agen-tic"],
    image: agenTicLogisticoImg,
    tab: "nuevo",
    popularity: 7,
    initialMessage:
      "Hola, soy tu agente logístico ¿quieres que gestione alguna operación?",
  },
  {
    title: "AGEN-TIC Formador",
    description: "Ofrece servicios de formación",
    departments: ["Más usados", "Agen-tic"],
    image: agenTicFormadorImg,
    tab: "nuevo",
    popularity: 4,
    initialMessage:
      "Hola, soy tu agente formador ¿en qué quieres que te forme?",
  },
  */
];
