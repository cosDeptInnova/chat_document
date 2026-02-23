// Paleta de colores pastel para participantes (misma que TranscriptionView)
export const PASTEL_COLORS = [
  { main: '#A8D5E2', bg: '#A8D5E220' }, // Azul pastel
  { main: '#F4A5AE', bg: '#F4A5AE20' }, // Rosa pastel
  { main: '#B5E5CF', bg: '#B5E5CF20' }, // Verde pastel
  { main: '#F9E79F', bg: '#F9E79F20' }, // Amarillo pastel
  { main: '#D4B5E8', bg: '#D4B5E820' }, // Morado pastel
  { main: '#F5C29F', bg: '#F5C29F20' }, // Naranja pastel
  { main: '#A8E6CF', bg: '#A8E6CF20' }, // Turquesa pastel
  { main: '#FFD4B3', bg: '#FFD4B320' }, // Melocotón pastel
  { main: '#C7CEEA', bg: '#C7CEEA20' }, // Lavanda pastel
  { main: '#FFB3BA', bg: '#FFB3BA20' }, // Rosa claro pastel
];

// Función para asignar color único a cada participante.
// Usa el índice del nombre en allSpeakers (orden alfabético recomendado) para que
// cada participante distinto tenga un color distinto. Si el nombre no está en la lista,
// se usa un hash; para evitar colisiones con pocos colores, se usa un hash más disperso.
export const getSpeakerColor = (speakerName: string, allSpeakers: string[]): { main: string; bg: string } => {
  const speakerIndex = allSpeakers.indexOf(speakerName);

  if (speakerIndex >= 0) {
    return PASTEL_COLORS[speakerIndex % PASTEL_COLORS.length];
  }

  // Fallback: nombre no estaba en la lista (hash más disperso para reducir colisiones)
  let hash = 0;
  for (let i = 0; i < speakerName.length; i++) {
    hash = speakerName.charCodeAt(i) + ((hash << 5) - hash);
  }
  const index = Math.abs(hash) % PASTEL_COLORS.length;
  return PASTEL_COLORS[index];
};

// Obtener solo el color main (para gráficos)
export const getSpeakerColorMain = (speakerName: string, allSpeakers: string[]): string => {
  return getSpeakerColor(speakerName, allSpeakers).main;
};
