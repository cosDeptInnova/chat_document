# Problema: Scrolls Independientes en Página de Detalle de Reuniones

## CONTEXTO DEL PROYECTO

Estamos trabajando en una aplicación React con TypeScript y Tailwind CSS llamada "notetaker". Es una aplicación para gestionar reuniones que incluye:
- Visualización de transcripciones de reuniones
- Reproductores de audio y video
- Panel de análisis con IA (resumen, insights, análisis)

**Estructura de archivos relevantes:**
- `frontend/src/pages/MeetingDetail.tsx` - Página principal que contiene todo el layout
- `frontend/src/components/MeetingDetail/TranscriptionView.tsx` - Componente que muestra transcripciones
- `frontend/src/components/MeetingDetail/AudioPlayer.tsx` - Reproductor de audio con transcripción
- `frontend/src/components/MeetingDetail/VideoPlayer.tsx` - Reproductor de video con transcripción
- `frontend/src/components/IAPanel/IAPanel.tsx` - Panel lateral derecho con pestañas de IA

## ESTRUCTURA ACTUAL DEL LAYOUT

```
┌─────────────────────────────────────────────────────────┐
│ Header: "Volver a reuniones" + Info del usuario         │
├─────────────────────────────────────────────────────────┤
│ Información de la reunión (título, fecha, estado)       │
├─────────────────────────────┬───────────────────────────┤
│ COLUMNA IZQUIERDA (2/3)     │ COLUMNA DERECHA (1/3)     │
│                             │                           │
│ ┌─────────────────────────┐ │ ┌──────────────────────┐ │
│ │ Tabs: Trans/Audio/Video │ │ │ IAPanel Tabs:        │ │
│ ├─────────────────────────┤ │ │ Resumen/Insights/    │ │
│ │                         │ │ │ Análisis             │ │
│ │ CONTENIDO TAB ACTIVA    │ │ ├──────────────────────┤ │
│ │                         │ │ │                      │ │
│ │                         │ │ │ CONTENIDO PESTAÑA    │ │
│ │                         │ │ │                      │ │
│ └─────────────────────────┘ │ └──────────────────────┘ │
└─────────────────────────────┴───────────────────────────┘
```

## OBJETIVO QUE QUEREMOS CONSEGUIR

Queremos implementar **scrolls independientes** para tres secciones diferentes:

### 1. Pestaña "Transcripción" (columna izquierda)
- **Sección FIJA arriba**: Estadísticas (segmentos, duración, participantes) + Lista de participantes
- **Sección con SCROLL**: Transcripción completa (debe hacer scroll independiente, sin mover la sección fija)

### 2. Pestaña "Audio" (columna izquierda)
- **Sección FIJA arriba**: Reproductor de audio
- **Sección con SCROLL**: Solo la transcripción (sin estadísticas/participantes, ya que se ven en pestaña Transcripción)

### 3. Pestaña "Video" (columna izquierda)
- **Sección FIJA arriba**: Reproductor de video
- **Sección con SCROLL**: Solo la transcripción (sin estadísticas/participantes)

### 4. Panel IA (columna derecha)
- **Sección FIJA arriba**: Nav de pestañas (Resumen, Insights, Análisis)
- **Sección con SCROLL**: Contenido de la pestaña activa (SummaryTab, InsightsTab, AnalysisTab)

## CAMBIOS YA IMPLEMENTADOS

1. **TranscriptionView.tsx**:
   - Añadida prop `showHeaderInfo?: boolean` (default: `true`)
   - Cuando `showHeaderInfo=true`: Muestra estadísticas y participantes fijos + transcripción con scroll
   - Cuando `showHeaderInfo=false`: Solo muestra transcripción con scroll
   - Estructura: `<div className="flex flex-col h-full">` → Sección fija `flex-shrink-0` → Scroll `flex-1 min-h-0 overflow-y-auto`

2. **AudioPlayer.tsx y VideoPlayer.tsx**:
   - Estructura: `<div className="flex flex-col h-full">` → Reproductor fijo `flex-shrink-0` → Transcripción con scroll usando `<TranscriptionView showHeaderInfo={false} />`

3. **IAPanel.tsx**:
   - Estructura: `<div className="flex flex-col h-full">` → Nav fijo `flex-shrink-0` → Contenido con scroll `flex-1 overflow-y-auto`

4. **MeetingDetail.tsx**:
   - Contenedor de tabs: `max-h-[calc(100vh-200px)]` con `flex flex-col`
   - Contenedor de contenido: `flex-1 min-h-0 overflow-hidden p-6`

## PROBLEMAS ACTUALES

### Problema Principal
**NO aparece el scroll en ninguna de las secciones que deberían tenerlo:**
- ❌ La transcripción completa NO muestra scroll (contenido cortado, no se puede hacer scroll)
- ❌ El panel de IA (Resumen/Insights/Análisis) NO muestra scroll (contenido cortado)

### Síntomas Observados
- Solo aparece un scroll general a la derecha de toda la página
- El contenido de la transcripción está cortado y no se puede acceder al resto
- El contenido del panel de IA está cortado y no se puede hacer scroll

### Hipótesis del Problema
Creemos que el problema puede estar relacionado con:
1. **Cadena de altura en flexbox**: Los contenedores padre no están pasando correctamente la altura (`h-full`) a los hijos
2. **El `max-h-[calc(100vh-200px)]`** en MeetingDetail puede no estar funcionando correctamente porque el grid padre no tiene altura definida
3. **El `min-h-0`** puede no ser suficiente si algún contenedor padre intermedio no tiene altura limitada
4. **El padding `p-6`** en el contenedor de contenido puede estar afectando el cálculo de altura

## CÓDIGO ACTUAL RELEVANTE

### MeetingDetail.tsx (líneas 177-255)
```tsx
<div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
  <div className="lg:col-span-2 space-y-6">
    <div className="bg-white dark:bg-slate-800 rounded-lg shadow transition-colors flex flex-col max-h-[calc(100vh-200px)]">
      <div className="flex-shrink-0 border-b ..."> {/* Nav tabs */} </div>
      <div className="flex-1 min-h-0 overflow-hidden p-6">
        {/* TranscriptionView o AudioPlayer o VideoPlayer */}
      </div>
    </div>
  </div>
  <div className="lg:col-span-1 flex flex-col">
    <IAPanel meetingId={meeting.id} />
  </div>
</div>
```

### TranscriptionView.tsx (modo completo, líneas 181-229)
```tsx
<div className="flex flex-col h-full">
  <div className="flex-shrink-0 space-y-3 pb-4">
    {/* Estadísticas y participantes fijos */}
  </div>
  <div className="flex-1 min-h-0 overflow-y-auto">
    {/* Transcripción con scroll */}
  </div>
</div>
```

### IAPanel.tsx (líneas 83-125)
```tsx
<div className="bg-white dark:bg-slate-800 ... flex flex-col h-full">
  <div className="flex-shrink-0 border-b ..."> {/* Nav tabs */} </div>
  <div className="flex flex-col flex-1 overflow-hidden">
    <div className="flex-1 overflow-y-auto p-6">
      {/* SummaryTab/InsightsTab/AnalysisTab */}
    </div>
  </div>
</div>
```

## QUÉ NECESITAMOS

Una solución que:
1. ✅ Haga que el scroll aparezca y funcione en la sección de transcripción (en las 3 pestañas)
2. ✅ Haga que el scroll aparezca y funcione en el panel de IA
3. ✅ Mantenga las secciones fijas visibles (stats/participantes/reproductores/nav)
4. ✅ Funcione de manera responsive
5. ✅ No rompa el diseño existente

## RESTRICCIONES

- Usamos Tailwind CSS (no CSS personalizado)
- React con TypeScript
- El código debe mantener la estructura actual en la mayor medida posible
- La solución debe ser compatible con navegadores modernos

---

**¿Puedes ayudarnos a identificar por qué los scrolls no aparecen y proponer una solución que funcione?**
