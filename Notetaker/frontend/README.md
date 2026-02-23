# Frontend Notetaker

Frontend React + TypeScript para el sistema Notetaker de transcripción y análisis de reuniones Teams.

## 🚀 Inicio rápido

### Prerrequisitos

- Node.js 18+ y npm
- Backend FastAPI ejecutándose en `http://localhost:7000`

### Instalación

```bash
# Instalar dependencias
npm install

# Crear archivo de configuración
cp .env.example .env

# Editar .env y configurar la URL del backend si es necesario
# VITE_API_BASE_URL=http://localhost:7000
```

### Desarrollo

```bash
# Iniciar servidor de desarrollo
npm run dev
```

El frontend estará disponible en `http://localhost:5173` (puerto por defecto de Vite).

### Build para producción

```bash
npm run build
```

Los archivos compilados estarán en la carpeta `dist/`.

## 📁 Estructura del proyecto

```
src/
├── components/          # Componentes reutilizables
│   ├── Layout/        # Layout principal (Sidebar, Topbar)
│   ├── MeetingDetail/  # Componentes de detalle de reunión
│   └── IAPanel/       # Panel de análisis IA
├── context/           # Contextos de React (Auth)
├── hooks/             # Custom hooks
├── mocks/             # Datos mock para IA
├── pages/             # Páginas principales
├── services/          # Cliente API
├── types/             # Tipos TypeScript
└── App.tsx            # Componente principal con rutas
```

## 🔐 Autenticación

El frontend usa autenticación simple por email mediante el endpoint `/api/auth/simple-login`. El token y la información del usuario se almacenan en `localStorage`.

## 📋 Funcionalidades

### Usuarios normales

- **Basic**: Ver transcripciones y análisis IA básico
- **Advanced**: Todo lo anterior + reproductor de audio
- **Pro**: Todo lo anterior + reproductor de vídeo

### Administradores

- Acceso completo a todas las funcionalidades
- Gestión de usuarios y licencias
- Vista global de todas las reuniones

## 🎨 Tecnologías

- **React 19** + **TypeScript**
- **Vite** - Build tool
- **React Router** - Routing
- **Tailwind CSS** - Estilos
- **Axios** - Cliente HTTP
- **Recharts** - Gráficos
- **Heroicons** - Iconos
- **date-fns** - Manejo de fechas

## 🔌 Endpoints del backend utilizados

- `POST /api/auth/simple-login` - Login por email
- `GET /api/meetings/list` - Listar reuniones
- `GET /api/meetings/{id}` - Detalle de reunión
- `GET /api/meetings/{id}/transcription` - Transcripción
- `GET /api/meetings/{id}/audio` - Archivo de audio
- `GET /api/meetings/{id}/video` - Archivo de vídeo

## 📝 Notas

- Los datos de análisis IA están mockeados por ahora (`src/mocks/iaData.ts`)
- Las licencias se calculan en el frontend basándose en `is_admin` e `is_premium` del backend
- El diseño es responsive y está inspirado en MeetGeek
