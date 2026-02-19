// Este archivo va a contener el fondo animado del login (el de pulso orbital) y de centrar todo en la pantalla
// Es un Layout para el login/registro

import React, { useEffect, useRef } from "react";

export default function AuthLayout({ children }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const container = canvasRef.current;
    if (!container) return;

    // Creamos el canvas
    const canvas = document.createElement("canvas");
    canvas.style.position = "absolute";
    canvas.style.top = "0";
    canvas.style.left = "0";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    canvas.style.zIndex = "0"; // Ponemos el fondo al final/abajo del todo
    container.appendChild(canvas);

    const ctx = canvas.getContext("2d");
    let animationId;

    // Configuración base
    const baseSpacing = 275; 
    let spacing = baseSpacing;
    let cols, rows;

    // Función para ajustar tamaño según pantalla
    const resizeCanvas = () => {
      canvas.width = window.innerWidth;
      canvas.height = window.innerHeight;
      
      if (window.innerWidth < 768) {
        // Para móvil
        spacing = 140; 
      } else if (window.innerWidth < 1536) { 
        // Para portátil
        spacing = 180; 
      } else {
        // Pantallas de escritorio
        spacing = baseSpacing;
      }
      
      cols = Math.floor(canvas.width / spacing) + 4;
      rows = Math.floor(canvas.height / spacing) + 4;
    };

    // Inicializar tamaño
    resizeCanvas();

    const amplitude = 35;
    const waveSpeed = 0.0003;

    const baseColors = [
      [172, 217, 54], [255, 153, 0], [78, 61, 142],
      [40, 178, 150], [210, 50, 80], [121, 181, 230],
      [235, 235, 145], [255, 218, 172], [216, 216, 243], [255, 197, 210]
    ];

    const colorChangeCooldown = 3000;
    const opacity = 0.7;

    function lerpColor(c1, c2, t) {
      return [
        Math.round(c1[0] + (c2[0] - c1[0]) * t),
        Math.round(c1[1] + (c2[1] - c1[1]) * t),
        Math.round(c1[2] + (c2[2] - c1[2]) * t)
      ];
    }

    function lightenColor(rgbaStr, percent) {
      const rgba = rgbaStr.match(/rgba?\((\d+), (\d+), (\d+),? ?([\d.]*)?\)/);
      if (!rgba) return rgbaStr;
      const r = Math.min(255, parseInt(rgba[1]) + (255 * percent / 100));
      const g = Math.min(255, parseInt(rgba[2]) + (255 * percent / 100));
      const b = Math.min(255, parseInt(rgba[3]) + (255 * percent / 100));
      const a = rgba[4] || 1;
      return `rgba(${r}, ${g}, ${b}, ${a})`;
    }

    function draw(time) {
      if (!canvas.parentElement) return; // Con esto evitamos errores si se desmonta

      const t = time * waveSpeed;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2 + 100;

      ctx.shadowColor = "rgba(0,0,0,0.4)";
      ctx.shadowBlur = 8;
      ctx.shadowOffsetX = 0;
      ctx.shadowOffsetY = 4;

      const transitionProgress = (time % colorChangeCooldown) / colorChangeCooldown;

      for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
          // En lugar de calcular el ancho total y dividir, restamos la mitad de las columnas al índice actual.
          // Esto fuerza matemáticamente a que el "centro del bucle" sea el "centro de la pantalla".
          const x = (col - cols / 2) * spacing; 
          const z = (row - rows / 2) * spacing;

          const dx = col - cols / 2;
          const dz = row - rows / 2;
          const dist = Math.sqrt(dx * dx + dz * dz);
          const offset = dist * 0.4;
          const height = Math.sin(t * 3 - offset) * amplitude;

          const angleX = Math.PI / 4;
          const angleZ = Math.PI / 3;

          const isoX = centerX + x * Math.cos(angleX) * 1.5;
          const isoY = centerY - height + z * Math.sin(angleZ) * 1.5;

          const squareSize = spacing * 0.8;

          const baseIndex = (col + row + Math.floor(time / colorChangeCooldown)) % baseColors.length;
          const nextIndex = (baseIndex + 1) % baseColors.length;

          const colorFrom = baseColors[baseIndex];
          const colorTo = baseColors[nextIndex];

          const interpolated = lerpColor(colorFrom, colorTo, transitionProgress);
          const rgba = `rgba(${interpolated[0]}, ${interpolated[1]}, ${interpolated[2]}, ${opacity})`;

          const gradient = ctx.createLinearGradient(isoX, isoY, isoX, isoY + squareSize);
          gradient.addColorStop(0, lightenColor(rgba, 20));
          gradient.addColorStop(1, rgba);

          ctx.fillStyle = gradient;
          ctx.beginPath();
          ctx.moveTo(isoX, isoY);
          ctx.lineTo(isoX + squareSize, isoY);
          ctx.lineTo(isoX + squareSize, isoY + squareSize);
          ctx.lineTo(isoX, isoY + squareSize);
          ctx.closePath();
          ctx.fill();
        }
      }

      ctx.shadowColor = "transparent";
      ctx.shadowBlur = 0;

      animationId = requestAnimationFrame(draw);
    }

    animationId = requestAnimationFrame(draw);

    window.addEventListener("resize", resizeCanvas);

    return () => {
      window.removeEventListener("resize", resizeCanvas);
      cancelAnimationFrame(animationId);
      if (container.contains(canvas)) {
        container.removeChild(canvas);
      }
    };
  }, []);

  return (
    <div 
      ref={canvasRef} 
      className="relative min-h-screen w-full overflow-hidden bg-white flex items-center justify-center"
    >
      {/* Contenedor para el formulario (children) centrado y por encima del canvas */}
      <div className="relative z-10 w-full h-full flex items-center justify-center p-2">
        {children}
      </div>
    </div>
  );
}