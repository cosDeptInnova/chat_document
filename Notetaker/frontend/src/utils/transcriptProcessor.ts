/**
 * Utilidades para procesar el JSON raw de Recall.ai y convertirlo al formato
 * que necesita el frontend para mostrar la transcripción.
 */

import type { TranscriptionSegment } from '../types';

interface RecallUtterance {
  text?: string;
  transcript?: string;
  speaker?: string;
  speaker_name?: string;
  speaker_id?: string;
  participant?: {
    name?: string;
    id?: string;
  };
  start?: number;
  start_time?: number;
  timestamp?: number;
  end?: number;
  end_time?: number;
  words?: Array<{
    word?: string;
    text?: string;
    start_timestamp?: {
      relative?: number;
    };
    end_timestamp?: {
      relative?: number;
    };
  }>;
  confidence?: number;
}

/**
 * Procesa el JSON raw de Recall.ai y lo convierte al formato del frontend
 */
export function processRecallTranscript(rawJson: any): {
  conversation: TranscriptionSegment[];
  speakers: Array<{
    name: string;
    segments_count: number;
    total_duration: number;
  }>;
  total_segments: number;
  total_duration_seconds: number;
  full_text: string;
} {
  // Normalizar a una lista de utterances
  let utterances: RecallUtterance[] = [];

  if (Array.isArray(rawJson)) {
    utterances = rawJson;
  } else if (typeof rawJson === 'object' && rawJson !== null) {
    // Formato 1: {"utterances": [...]}
    if (Array.isArray(rawJson.utterances)) {
      utterances = rawJson.utterances;
    }
    // Formato 2: {"transcript": {"utterances": [...]}}
    else if (rawJson.transcript && Array.isArray(rawJson.transcript.utterances)) {
      utterances = rawJson.transcript.utterances;
    }
    // Formato 3: Buscar cualquier clave que contenga una lista de utterances
    else {
      for (const key in rawJson) {
        if (Array.isArray(rawJson[key]) && rawJson[key].length > 0) {
          const first = rawJson[key][0];
          if (
            typeof first === 'object' &&
            (first.text || first.words || first.speaker || first.start || first.start_time)
          ) {
            utterances = rawJson[key];
            break;
          }
        }
      }
    }
  }

  if (utterances.length === 0) {
    return {
      conversation: [],
      speakers: [],
      total_segments: 0,
      total_duration_seconds: 0,
      full_text: '',
    };
  }

  // Procesar cada utterance
  const conversation: TranscriptionSegment[] = [];
  const speakersMap = new Map<string, { name: string; segments_count: number; total_duration: number }>();
  let maxEndTime = 0;

  for (const utterance of utterances) {
    if (typeof utterance !== 'object' || utterance === null) {
      continue;
    }

    // Extraer texto
    let text = utterance.text || utterance.transcript || '';
    if (!text && utterance.words && Array.isArray(utterance.words)) {
      text = utterance.words
        .map((w: any) => w.word || w.text || '')
        .filter(Boolean)
        .join(' ')
        .trim();
    }

    if (!text) {
      continue;
    }

    // Extraer speaker
    const speakerName =
      utterance.speaker_name ||
      utterance.speaker ||
      utterance.participant?.name ||
      utterance.speaker_id ||
      'Speaker:unknown';

    // Extraer timestamps
    let startTime = utterance.start || utterance.start_time || utterance.timestamp || 0;
    let endTime = utterance.end || utterance.end_time || startTime;

    // Si no hay timestamps pero hay words, usar los de las palabras
    if ((!startTime || !endTime) && utterance.words && Array.isArray(utterance.words) && utterance.words.length > 0) {
      const firstWord = utterance.words[0];
      const lastWord = utterance.words[utterance.words.length - 1];
      if (firstWord?.start_timestamp?.relative !== undefined) {
        startTime = firstWord.start_timestamp.relative;
      }
      if (lastWord?.end_timestamp?.relative !== undefined) {
        endTime = lastWord.end_timestamp.relative;
      }
    }

    startTime = Number(startTime) || 0;
    endTime = Number(endTime) || startTime;
    const duration = Math.max(0, endTime - startTime);

    // Crear segmento
    const segment: TranscriptionSegment = {
      speaker: speakerName,
      text: text.trim(),
      start_time: startTime,
      end_time: endTime,
      duration: duration > 0 ? duration : null,
    };

    conversation.push(segment);

    // Actualizar estadísticas de speakers
    if (!speakersMap.has(speakerName)) {
      speakersMap.set(speakerName, {
        name: speakerName,
        segments_count: 0,
        total_duration: 0,
      });
    }
    const speakerStats = speakersMap.get(speakerName)!;
    speakerStats.segments_count += 1;
    speakerStats.total_duration += duration;

    maxEndTime = Math.max(maxEndTime, endTime);
  }

  // Ordenar por tiempo de inicio
  conversation.sort((a, b) => a.start_time - b.start_time);

  // Construir texto completo
  const fullText = conversation.map((seg) => seg.text).join(' ').trim();

  return {
    conversation,
    speakers: Array.from(speakersMap.values()),
    total_segments: conversation.length,
    total_duration_seconds: maxEndTime,
    full_text: fullText,
  };
}

