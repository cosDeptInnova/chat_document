import React, { useEffect, useRef, useState } from "react";

export default function VoiceAvatar({ 
    isAgentSpeaking, 
    videosPreloaded = false,
    className = "" 
}) {
    const videoRef1 = useRef(null);
    const videoRef2 = useRef(null);
    const [activeVideoIndex, setActiveVideoIndex] = useState(1); // 1 para que empiece escuchando y 2 para que empiece hablando

    const [currentVideo, setCurrentVideo] = useState("");
    const [isLoading, setIsLoading] = useState(true);
    const [hasError, setHasError] = useState(false);

    // Flag para saber si el primer video ya está listo
    const [firstVideoReady, setFirstVideoReady] = useState(false);

    // Rutas de los videos
    const speakingVideo = "/attached_assets/cosmos-speaking.mp4";
    const listeningVideo = "/attached_assets/cosmos-listening.mp4";

    // Manejo del preload
    useEffect(() => {
        if (videosPreloaded) {
            console.log("VoiceAvatar: Videos ya precargados - la transición será instantánea");
            setIsLoading(false);
        } else {
            console.log("VoiceAvatar: Videos aún no precargados - usando fallback");
            //setIsLoading(true);
        }
    }, [videosPreloaded]);

    // Para la transición entre vídeos. Inicializamos el primer video al montar la pagina, el de listening
    useEffect(() => {
        // SIEMPRE empezamos con listening (escuchando)
        const initialVideo = listeningVideo;
        setCurrentVideo(initialVideo);

        // Cargamos el primer video en videoRef1 (el del elefante escuchando)
        if (videoRef1.current) {
            videoRef1.current.src = initialVideo;
            
            // Listener para saber cuándo está listo
            videoRef1.current.onloadeddata = () => {
                console.log("Video inicial listo para reproducir");
                setFirstVideoReady(true);
                setIsLoading(false);
                
                // Iniciar reproducción
                videoRef1.current.play().catch(err => {
                    console.warn("Autoplay bloqueado (normal en algunos navegadores):", err);
                    // Aún así, marcamos como listo
                    setFirstVideoReady(true);
                    setIsLoading(false);
                });
            };

            // Para manejar errores de carga
            videoRef1.current.onerror = (e) => {
                console.error("Error cargando video inicial:", e);
                setHasError(true);
                setIsLoading(false);
            };
            
            videoRef1.current.load();
        }
    }, []); // Solo se hace al montar

    // Cambiar video según estado y con una transición
    useEffect(() => {
        // Esperar a que el primer video esté listo antes de permitir cambios
        if (!firstVideoReady) return;

        const newVideo = isAgentSpeaking ? speakingVideo : listeningVideo;
        
        // Solo cambiar si es diferente
        if (newVideo === currentVideo) return;

        console.log(`Cambiando video: ${currentVideo} → ${newVideo}`);
        
        // Determinar cuál video usar para el crossfade
        const nextVideoRef = activeVideoIndex === 1 ? videoRef2 : videoRef1;
        const nextIndex = activeVideoIndex === 1 ? 2 : 1;

        if (nextVideoRef.current) {
            // Precargar el nuevo video en el ref inactivo
            nextVideoRef.current.src = newVideo;
            
            // Cuando esté listo, hacer el crossfade
            nextVideoRef.current.onloadeddata = () => {
                nextVideoRef.current.play().then(() => {
                    console.log(`Video ${nextIndex} listo y reproduciendo`);
                    // Cambiar el video activo (esto activa la transición CSS)
                    setActiveVideoIndex(nextIndex);
                    setCurrentVideo(newVideo);
                }).catch(err => {
                    console.warn("Autoplay bloqueado:", err);
                    // Cambiar de todas formas
                    setActiveVideoIndex(nextIndex);
                    setCurrentVideo(newVideo);
                });
            };

            nextVideoRef.current.onerror = (e) => {
                console.error(`Error cargando video ${nextIndex}:`, e);
            };
            
            nextVideoRef.current.load();
        }

    }, [isAgentSpeaking, currentVideo, activeVideoIndex, speakingVideo, listeningVideo, firstVideoReady]);	

    // Manejar error en video (Fallback si no hay videos disponibles)
    const handleVideoError = (error, videoNumber) => {
        console.error(`Error cargando video ${videoNumber}:`, error);
        setHasError(true);
        setIsLoading(false)
    };

    return (
        <div className={`relative ${className}`}>
            {/* Contenedor del avatar */}
            <div className="w-full aspect-square bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg overflow-hidden relative">
                
                {/* Videos principales con transición */}
                <div className="absolute inset-0">
                    {/* Video 1 - Siempre renderizado */}
                    <video 
                        ref={videoRef1}
                        autoPlay 
                        loop 
                        muted 
                        playsInline 
                        onError={(e) => handleVideoError(e, 1)}
                        className="absolute inset-0 w-full h-full object-cover transition-opacity duration-700 ease-in-out"
                        style={{ 
                            opacity: (activeVideoIndex === 1 && firstVideoReady) ? 1 : 0,
                            zIndex: activeVideoIndex === 1 ? 2 : 1
                        }}
                    />
                    
                    {/* Video 2 - Siempre renderizado */}
                    <video 
                        ref={videoRef2}
                        autoPlay 
                        loop 
                        muted 
                        playsInline 
                        onError={(e) => handleVideoError(e, 2)}
                        className="absolute inset-0 w-full h-full object-cover transition-opacity duration-700 ease-in-out"
                        style={{ 
                            opacity: (activeVideoIndex === 2 && firstVideoReady) ? 1 : 0,
                            zIndex: activeVideoIndex === 2 ? 2 : 1
                        }}
                    />
                </div>

                {/* Fallback solo cuando no hay video listo */}
                {!firstVideoReady && (
                    <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-blue-500 to-purple-600 z-10">
                        {isLoading && !hasError ? (
                            // Estado: Cargando el primer video
                            <div className="text-center">
                                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-white mb-2 mx-auto"></div>
                                <div className="text-xs text-white opacity-80">Cargando avatar...</div>
                            </div>
                        ) : hasError ? (
                            // Estado: Error en videos
                            <div className="text-center">
                                <div className="w-16 h-16 bg-white bg-opacity-90 rounded-full flex items-center justify-center shadow-lg mb-2 mx-auto">
                                    <span className="text-2xl font-bold text-blue-600">C</span>
                                </div>
                                <div className="text-xs text-white opacity-80">Modo estático</div>
                            </div>
                        ) : (
                            // Estado: Fallback genérico
                            <div className="w-20 h-20 bg-white rounded-full flex items-center justify-center shadow-lg">
                                <span className="text-3xl font-bold text-blue-600">C</span>
                            </div>
                        )}
                    </div>
                )}

                {/* Indicador de estado - solo visible cuando el video está listo */}
                {firstVideoReady && (
                    <div className="absolute bottom-2 right-2 z-10">
                        <div className={`w-3 h-3 rounded-full transition-colors duration-300 ${
                            isAgentSpeaking ? "bg-green-500 animate-pulse" : "bg-blue-500"
                        }`} />
                    </div>
                )}

                {/* Animación de ondas cuando habla - solo visible cuando el video está listo */}
                {isAgentSpeaking && firstVideoReady && (
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none z-10">
                        <div className="absolute w-24 h-24 border-2 border-green-400 rounded-full animate-ping opacity-30"></div>
                        <div className="absolute w-32 h-32 border-2 border-green-300 rounded-full animate-ping opacity-20" style={{ animationDelay: "0.5s" }}></div>
                    </div>
                )}
            </div>

            {/* Estado textual */}
            <div className="mt-2 text-center">
                {firstVideoReady ? (
                    <span className={`text-xs font-medium transition-colors duration-300 ${
                        isAgentSpeaking ? "text-green-600" : "text-blue-600"
                    }`}>
                        {isAgentSpeaking ? "Hablando..." : "Escuchando..."}
                    </span>
                ) : (
                    <span className="text-xs font-medium text-gray-500">
                        {isLoading ? "Inicializando..." : "Error"}
                    </span>
                )}
                
                {/* Indicador de optimización solo mientras carga */}
                {!firstVideoReady && isLoading && (
                    <div className="text-xs text-amber-600 mt-1">
                        Preparando avatar
                    </div>
                )}
            </div>
        </div>
    );
}