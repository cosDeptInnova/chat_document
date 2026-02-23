# Prompt para IA de Análisis de Transcripciones

## Estructura del JSON de Transcripción

La transcripción de las reuniones se guarda en la base de datos como un campo JSON (`raw_transcript_json`) que contiene el formato original de Recall.ai/Deepgram. El JSON puede tener diferentes estructuras, pero todas contienen información sobre los segmentos de habla de cada participante.

### Estructuras posibles del JSON:

#### Formato 1: Array directo de utterances
```json
[
  {
    "text": "Hola, ¿cómo estás?",
    "speaker": "Speaker:0",
    "speaker_name": "Juan Pérez",
    "start_time": 0.5,
    "end_time": 2.3,
    "words": [
      {
        "word": "Hola",
        "start_timestamp": { "relative": 0.5 },
        "end_timestamp": { "relative": 0.8 }
      },
      {
        "word": "¿cómo",
        "start_timestamp": { "relative": 0.9 },
        "end_timestamp": { "relative": 1.2 }
      }
    ],
    "confidence": 0.95
  },
  {
    "text": "Bien, gracias. ¿Y tú?",
    "speaker": "Speaker:1",
    "speaker_name": "María García",
    "start_time": 2.5,
    "end_time": 4.1
  }
]
```

#### Formato 2: Objeto con clave "utterances"
```json
{
  "utterances": [
    {
      "text": "Hola, ¿cómo estás?",
      "speaker": "Speaker:0",
      "start": 0.5,
      "end": 2.3
    }
  ]
}
```

#### Formato 3: Objeto anidado "transcript.utterances"
```json
{
  "transcript": {
    "utterances": [
      {
        "text": "Hola, ¿cómo estás?",
        "speaker": "Speaker:0",
        "start_time": 0.5,
        "end_time": 2.3
      }
    ]
  }
}
```

#### Formato 4: Array con objetos "words" + "participant" (formato Recall.ai detallado)
Este formato es común en transcripciones de Recall.ai y contiene información detallada palabra por palabra:

```json
[
  {
    "words": [
      {
        "text": "Hola.",
        "end_timestamp": {
          "absolute": "2026-01-16T14:30:07.435Z",
          "relative": 15.948799
        },
        "start_timestamp": {
          "absolute": "2026-01-16T14:30:07.035Z",
          "relative": 15.5487995
        }
      }
    ],
    "participant": {
      "id": 300,
      "name": "Ruben Carro Sanchez",
      "email": null,
      "is_host": true,
      "platform": "unknown",
      "extra_data": {
        "microsoft_teams": {
          "role": "8:orgid:...",
          "user_id": "8:orgid:...",
          "meeting_role": "organizer",
          "tenant_id": "...",
          "participant_type": "inTenant"
        }
      }
    }
  },
  {
    "words": [
      {
        "text": "Epa.",
        "end_timestamp": { "relative": 18.008797 },
        "start_timestamp": { "relative": 17.208797 }
      },
      {
        "text": "Oye,",
        "end_timestamp": { "relative": 18.648798 },
        "start_timestamp": { "relative": 18.328798 }
      }
    ],
    "participant": {
      "id": 100,
      "name": "Fernando Descartin Naves",
      "is_host": false,
      "platform": "unknown"
    }
  }
]
```

**Características del Formato 4:**
- No tiene campos directos `text`, `speaker`, `start_time`, `end_time`
- El **texto** se construye concatenando `words[].text`
- El **speaker** viene de `participant.name` o `participant.id`
- Los **timestamps** se toman de `words[0].start_timestamp.relative` (inicio) y `words[last].end_timestamp.relative` (fin)
- Cada objeto del array es un segmento de habla de un participante
- Los `words` pueden ser múltiples palabras en un mismo segmento

**Ejemplo de extracción del Formato 4:**
```javascript
// Para cada objeto en el array:
const segment = rawJson[0];

// Extraer texto: concatenar todas las palabras
const text = segment.words.map(w => w.text).join(' ');

// Extraer speaker: usar participant.name o participant.id
const speaker = segment.participant.name || `Participant:${segment.participant.id}`;

// Extraer timestamps: primera y última palabra
const startTime = segment.words[0].start_timestamp.relative;
const endTime = segment.words[segment.words.length - 1].end_timestamp.relative;
```

### Campos importantes en cada utterance:

**Formatos 1-3:**
- **`text`** o **`transcript`**: El texto transcrito del segmento
- **`speaker`** o **`speaker_id`**: Identificador del hablante (ej: "Speaker:0", "Speaker:1")
- **`speaker_name`**: Nombre real del participante si está disponible (opcional)
- **`start`**, **`start_time`** o **`timestamp`**: Tiempo de inicio en segundos desde el comienzo de la reunión
- **`end`** o **`end_time`**: Tiempo de fin en segundos
- **`words`**: Array opcional de palabras individuales con timestamps precisos
- **`confidence`**: Nivel de confianza de la transcripción (0-1)

**Formato 4 (words + participant):**
- **`words`**: Array obligatorio de palabras, cada una con:
  - `text`: Texto de la palabra
  - `start_timestamp.relative`: Tiempo de inicio en segundos
  - `end_timestamp.relative`: Tiempo de fin en segundos
  - `start_timestamp.absolute`: Timestamp absoluto (opcional, formato ISO)
  - `end_timestamp.absolute`: Timestamp absoluto (opcional, formato ISO)
- **`participant`**: Objeto con información del participante:
  - `id`: ID numérico del participante
  - `name`: Nombre completo del participante (campo principal para speaker)
  - `email`: Email (opcional, puede ser null)
  - `is_host`: Boolean indicando si es el organizador
  - `platform`: Plataforma usada (ej: "unknown", "teams")
  - `extra_data.microsoft_teams`: Metadata adicional de Teams (opcional)

## Cómo se procesa y presenta en el frontend

### 1. Normalización del formato

El sistema normaliza todos los formatos posibles extrayendo siempre una lista de "utterances" (segmentos de habla) con esta estructura mínima:

```typescript
{
  speaker: string;        // Ej: "Speaker:0" o "Juan Pérez"
  text: string;           // Texto completo del segmento
  start_time: number;     // Segundos desde inicio (0.5)
  end_time: number;       // Segundos desde inicio (2.3)
  duration: number | null // Duración calculada (end_time - start_time)
}
```

### 2. Agrupación de segmentos consecutivos

Los segmentos se agrupan cuando:
- **Mismo speaker**: El hablante es el mismo
- **Pausa corta**: La diferencia entre el `end_time` del segmento anterior y el `start_time` del siguiente es menor a **1.2 segundos**

**Ejemplo de agrupación:**

Segmentos originales:
```
Speaker:0 [0.5 - 2.3]: "Hola,"
Speaker:0 [2.4 - 3.1]: "¿cómo estás?"
Speaker:0 [3.2 - 4.0]: "Bien gracias."
Speaker:1 [5.0 - 6.2]: "Perfecto."
```

Agrupados (pausa entre 4.0 y 5.0 es > 1.2s):
```
Speaker:0 [0.5 - 4.0]: "Hola, ¿cómo estás? Bien gracias."
Speaker:1 [5.0 - 6.2]: "Perfecto."
```

### 3. Presentación visual en el frontend

Cada grupo se presenta así:

```
┌─────────────────────────────────────────┐
│ Speaker:0              [00:00]          │ ← Header con speaker y timestamp
│─────────────────────────────────────────│
│ Hola, ¿cómo estás? Bien gracias.        │ ← Texto agrupado
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ Speaker:1              [00:05]          │
│─────────────────────────────────────────│
│ Perfecto.                               │
└─────────────────────────────────────────┘
```

## Instrucciones para la IA de Análisis

### Contexto

Eres una IA especializada en analizar transcripciones de reuniones de Teams. Recibirás el JSON raw (`raw_transcript_json`) de una reunión y debes:

1. **Normalizar el formato** (extraer utterances independientemente de la estructura)
2. **Agrupar segmentos consecutivos** del mismo speaker (pausa < 1.2s)
3. **Generar análisis** (resumen, puntos clave, acciones, etc.)

### Pasos de procesamiento:

#### Paso 1: Extraer utterances del JSON

Busca las utterances en este orden de prioridad:
1. Si `raw_json` es un array → usa directamente
2. Si tiene `raw_json.utterances` → usa ese array
3. Si tiene `raw_json.transcript.utterances` → usa ese array
4. Busca cualquier clave que contenga un array de objetos con campos `text`, `speaker`, `start_time`, o `words`

**Detectar formato:**
- Si el array contiene objetos con `words` y `participant` → **Formato 4** (procesar especial)
- Si el array contiene objetos con `text` y `speaker` → **Formatos 1-3** (procesar normal)

#### Paso 2: Normalizar cada utterance

Para cada utterance, extrae según el formato detectado:

**Formatos 1-3 (con campos directos):**
```javascript
const text = utterance.text || utterance.transcript || '';
const speaker = utterance.speaker_name || utterance.speaker || utterance.speaker_id || 'Speaker:unknown';
const startTime = utterance.start || utterance.start_time || utterance.timestamp || 0;
const endTime = utterance.end || utterance.end_time || startTime;
```

**Si no hay texto pero hay words (Formatos 1-3):**
```javascript
if (!text && utterance.words && Array.isArray(utterance.words)) {
  text = utterance.words.map(w => w.word || w.text || '').join(' ').trim();
}
```

**Formato 4 (words + participant):**
```javascript
// Extraer texto concatenando todas las palabras
const text = utterance.words
  .map(w => w.text || w.word || '')
  .filter(Boolean)
  .join(' ')
  .trim();

// Extraer speaker del participant
const speaker = utterance.participant?.name || 
                `Participant:${utterance.participant?.id}` || 
                'Speaker:unknown';

// Extraer timestamps de primera y última palabra
const startTime = utterance.words[0]?.start_timestamp?.relative || 
                  utterance.words[0]?.start_timestamp || 0;
const endTime = utterance.words[utterance.words.length - 1]?.end_timestamp?.relative || 
                utterance.words[utterance.words.length - 1]?.end_timestamp || 
                startTime;
```

#### Paso 3: Ordenar por tiempo

Ordena todos los segmentos normalizados por `start_time` ascendente.

#### Paso 4: Agrupar segmentos consecutivos

```python
# Pseudocódigo de agrupación
grouped = []
current_group = None
PAUSE_THRESHOLD = 1.2  # segundos

for segment in sorted_segments:
    if current_group is None:
        current_group = {
            'speaker': segment.speaker,
            'start_time': segment.start_time,
            'texts': [segment.text],
            'end_time': segment.end_time
        }
    elif (current_group['speaker'] == segment.speaker and 
          segment.start_time - current_group['end_time'] < PAUSE_THRESHOLD):
        # Mismo speaker y pausa corta: agregar al grupo
        current_group['texts'].append(segment.text)
        current_group['end_time'] = segment.end_time
    else:
        # Nuevo speaker o pausa larga: guardar grupo anterior
        grouped.append(current_group)
        current_group = {
            'speaker': segment.speaker,
            'start_time': segment.start_time,
            'texts': [segment.text],
            'end_time': segment.end_time
        }

if current_group:
    grouped.append(current_group)
```

#### Paso 5: Generar formato legible para análisis

Para cada grupo, genera:

```
[SPEAKER] Nombre del Speaker
[TIMESTAMP] MM:SS (tiempo de inicio)
[TEXT] Texto completo del grupo
```

**Ejemplo:**

```
[SPEAKER] Juan Pérez
[TIMESTAMP] 00:00
[TEXT] Hola, ¿cómo estás? Bien gracias.

[SPEAKER] María García
[TIMESTAMP] 00:05
[TEXT] Perfecto. Vamos a revisar el proyecto entonces.
```

### Análisis esperado

Con el texto agrupado y formateado, genera:

1. **Resumen ejecutivo**: 2-3 párrafos con los temas principales
2. **Puntos clave**: Lista de 5-10 puntos importantes discutidos
3. **Participación**: Distribución de tiempo de habla por speaker
4. **Acciones**: Tareas identificadas con asignado y fecha si se menciona
5. **Decisiones**: Decisiones tomadas durante la reunión
6. **Temas**: Temas principales y sub-temas identificados

### Consideraciones importantes:

- Los **speakers** pueden ser "Speaker:0", "Speaker:1" (anonimizados) o nombres reales
- Los **timestamps** están en segundos desde el inicio (ej: 125.5 = 2 minutos y 5.5 segundos)
- El **idioma** es generalmente español, pero puede variar
- La **confianza** puede variar por segmento (verificar `confidence` si está disponible)
- Los **segmentos están ordenados cronológicamente** después de normalizar

### Ejemplo completo de prompt para la IA:

```
Analiza esta transcripción de reunión de Teams. El JSON contiene utterances de múltiples participantes.

Estructura del JSON (puede tener 4 formatos diferentes):

**Formatos 1-3 (campos directos):**
- Puede ser un array directo, o tener claves "utterances" o "transcript.utterances"
- Cada utterance tiene: text, speaker (o speaker_id/name), start_time, end_time
- Los timestamps están en segundos desde el inicio de la reunión

**Formato 4 (words + participant):**
- Es un array directo de objetos
- Cada objeto tiene:
  - words: Array de palabras con text, start_timestamp.relative, end_timestamp.relative
  - participant: Objeto con name, id, is_host
- Para extraer: concatenar words[].text para el texto, usar participant.name para speaker
- Timestamps: words[0].start_timestamp.relative (inicio) y words[last].end_timestamp.relative (fin)

Tarea:
1. Detecta el formato del JSON (verificar si tiene words+participant o campos directos)
2. Normaliza y extrae todos los segmentos de habla según el formato detectado
3. Agrupa segmentos consecutivos del mismo speaker (pausa < 1.2s)
4. Genera un formato legible con [SPEAKER], [TIMESTAMP] y [TEXT]
5. Analiza y proporciona:
   - Resumen ejecutivo
   - Puntos clave
   - Participación por speaker
   - Acciones identificadas
   - Decisiones tomadas

JSON a analizar:
{raw_transcript_json}
```

---

**Nota técnica**: Este formato se guarda en PostgreSQL como tipo JSON y se recupera desde el endpoint `GET /api/meetings/{meeting_id}/transcription` en el campo `raw_transcript_json`.
