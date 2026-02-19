// src/pages/ConversationHistoryMainPanel.jsx
import React, { useState, useEffect, useRef } from 'react';
import { FiMessageSquare, FiStar, FiTrash2, FiClock, FiArrowRightCircle } from "react-icons/fi";
import DeleteChatModal from '../components/utils/DeleteChatModal';
import { fetchConversations, deleteConversation, toggleFavoriteConversation } from '../lib/api';

export default function HistoryAndFavorites ({ isDarkMode, onOpenConversation }){
  const [activeTab, setActiveTab] = useState("historial");
  const [indicatorStyle, setIndicatorStyle] = useState({});
  const tabsRef = useRef({});
  const [chats, setChats] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState(null);

  // Estado para eliminación "visual" (no borra de BBDD)
  const [chatToDelete, setChatToDelete] = useState(null);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [isClosing, setIsClosing] = useState(false);

  const openModal = () => {
    setShowDeleteModal(true);
  };

  const closeModal = () => {
    setIsClosing(true);
    setTimeout(() => {
      setShowDeleteModal(false);
      setIsClosing(false);
      setChatToDelete(null); // Buena práctica: limpiar el ID al cerrar
    }, 300);
  };

  // Cargar conversaciones reales desde el backend
  useEffect(() => {
    let cancelled = false;

    const load = async () => {
      setLoading(true);
      setLoadError(null);
      try {
        const data = await fetchConversations();
        if (cancelled) return;

        // data = [{ id, title, created_at }]
        const mapped = (data || []).map(c => ({
          id: c.id,
          title: c.title || `Conversación ${c.id}`,
          createdAt: c.created_at,
          favorite: c.is_favorite
        }));

        setChats(mapped);
      } catch (err) {
        console.error("Error cargando conversaciones:", err);
        if (!cancelled) {
          setLoadError("No se pudo cargar el historial. Intenta de nuevo más tarde.");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    load();
    return () => { cancelled = true; };
  }, []);

  // Actualizar la posición del indicador cuando cambia la pestaña activa
  useEffect(() => {
    if (tabsRef.current[activeTab]) {
      const tabElement = tabsRef.current[activeTab];
      const { offsetLeft, offsetWidth } = tabElement;
      
      setIndicatorStyle({
        left: offsetLeft,
        width: offsetWidth,
        transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)'
      });
    }
  }, [activeTab]);

  // Función asíncrona para guardar en Backend
  const toggleFavorite = async (id) => {
    // a) Encontramos el estado actual
    const chatToUpdate = chats.find(c => c.id === id);
    if (!chatToUpdate) return;

    const newStatus = !chatToUpdate.favorite;

    // b) Actualización visual inmediata
    setChats(prev =>
      prev.map(chat => 
        chat.id === id ? { ...chat, favorite: newStatus } : chat
      )
    );

    try {
      // c) Llamada al Backend para guardar
      await toggleFavoriteConversation(id, newStatus);
    } catch (error) {
      console.error("Error al guardar favorito:", error);
      // d) Si falla, revertimos el cambio visual
      setChats(prev =>
        prev.map(chat => 
          chat.id === id ? { ...chat, favorite: !newStatus } : chat
        )
      );
    }
  };

  const filteredChats = activeTab === "favoritos" 
    ? chats.filter(chat => chat.favorite) 
    : chats;

  const formatDate = (iso) => {
    if (!iso) return "";
    try {
      const d = new Date(iso);
      return d.toLocaleString("es-ES", {
        day: "2-digit",
        month: "2-digit",
        year: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      });
    } catch {
      return iso;
    }
  };

  const handleConfirmDelete = async () => {
    if (!chatToDelete) return;

    try {
      // 1. Llamada al Backend (BBDD)
      await deleteConversation(chatToDelete);

      // 2. Actualización del frontend
      // Solo si la API no da error, lo quitamos de la lista visual
      setChats(prev => prev.filter(c => c.id !== chatToDelete));
      
      // 3. Cerrar el modal
      closeModal();
      
    } catch (error) {
      console.error("Error al eliminar la conversación:", error);
      alert("No se pudo eliminar la conversación. Inténtalo de nuevo.");
      closeModal() // Cerramos aunque falle
    }
  };

  return (
    <div className={`min-h-screen p-4 2xl:p-6 ${isDarkMode ? 'bg-gray-900 text-white' : 'bg-white text-gray-800'}`}>
      <div className="max-w-3xl 2xl:max-w-4xl mx-auto flex flex-col h-[calc(100vh-6rem)]">
        {/* Header */}
        <header className="mb-4 2xl:mb-6">
          <h1 className={`text-xl md:text-2xl 2xl:text-3xl font-extrabold ${isDarkMode ? 'text-blue-400' : 'text-blue-700'}`}>
            Historial de Conversaciones
          </h1>
          <p className={`mt-1 2xl:mt-2 text-xs md:text-sm 2xl:text-base ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
            Revisa tus conversaciones anteriores y tus favoritos
          </p>
        </header>

        {/* Tabs con indicador animado */}
        <div className="relative mb-4 2xl:mb-6">
          <div className="flex border-b border-gray-300 dark:border-gray-700">
            <button 
              ref={el => tabsRef.current.historial = el}
              onClick={() => setActiveTab("historial")} 
              className={`flex items-center justify-center flex-1 py-2 2xl:py-3 text-center text-sm 2xl:text-base font-medium transition-colors duration-300 gap-2 ${
                activeTab === "historial" 
                  ? (isDarkMode ? "text-white" : "text-blue-600") 
                  : (isDarkMode ? "text-gray-400 hover:text-white" : "text-gray-500 hover:text-blue-600")
              }`}
            >
              <FiClock className="w-4 h-4 2xl:w-5 2xl:h-5" />
              Historial de chats
            </button>
            
            <button 
              ref={el => tabsRef.current.favoritos = el}
              onClick={() => setActiveTab("favoritos")} 
              className={`flex items-center justify-center flex-1 py-2 2xl:py-3 text-center text-sm 2xl:text-base font-medium transition-colors duration-300 gap-2 ${
                activeTab === "favoritos" 
                  ? (isDarkMode ? "text-yellow-400" : "text-yellow-500") 
                  : (isDarkMode ? "text-gray-400 hover:text-yellow-400" : "text-gray-500 hover:text-yellow-500")
              }`}
            >
              {/* Lógica de estrella rellena en el Tab */}
              <FiStar className={`w-4 h-4 2xl:w-5 2xl:h-5 ${activeTab === "favoritos" ? "fill-current" : ""}`} />
              Favoritos
            </button>
          </div>
          
          {/* Indicador animado */}
          <div 
            className="absolute bottom-0 h-[2px] bg-gradient-to-r from-blue-500 to-purple-500 transition-all duration-300"
            style={indicatorStyle}
          />
        </div>

        {/* Contenido de la pestaña activa */}
        <div className="flex-1 overflow-y-auto pb-12 scrollbar-hide space-y-2 2xl:space-y-3">
          {loading && (
            <div className={`text-center py-12 rounded-2xl ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
              <p className="text-lg font-medium">Cargando historial...</p>
            </div>
          )}

          {!loading && loadError && (
            <div className={`text-center py-12 rounded-2xl ${isDarkMode ? 'bg-red-900/30' : 'bg-red-50'}`}>
              <p className="text-lg font-medium text-red-500">{loadError}</p>
            </div>
          )}

          {!loading && !loadError && (
            <>
              {filteredChats.length === 0 ? (
                <div className={`text-center py-8 2xl:py-10 rounded-xl ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
                  <FiMessageSquare className="w-12 h-12 2xl:w-16 2xl:h-16 mx-auto text-gray-400 mb-2" />
                  <p className="text-base 2xl:text-lg font-medium">
                    {activeTab === "favoritos" 
                      ? "No tienes conversaciones favoritas" 
                      : "No hay historial de conversaciones"}
                  </p>
                  <p className={`mt-1 2xl:mt-2 text-sm 2xl:text-base ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                    {activeTab === "favoritos"
                      ? "Marca algunas conversaciones como favoritas para verlas aquí"
                      : "Tus conversaciones aparecerán aquí cuando hables con COSMOS"}
                  </p>
                </div>
              ) : (
                filteredChats.map(chat => (
                  <div 
                    key={chat.id} 
                    className={`p-3 2xl:p-4 rounded-xl 2xl:rounded-2xl transition-all duration-300 ${
                      isDarkMode 
                        ? 'bg-gray-800 hover:bg-gray-700' 
                        : 'bg-white hover:bg-blue-50'
                    }`}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0 pr-2">
                        <h3 className="font-semibold text-sm sm:text-base 2xl:text-lg truncate">{chat.title}</h3>
                        <p className={`text-xs 2xl:text-sm mt-0.5 2xl:mt-1 ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                          {formatDate(chat.createdAt)}
                        </p>
                      </div>

                      {/* Botón Favorito */}
                      <button 
                        onClick={() => toggleFavorite(chat.id)}
                        className={`p-1.5 2xl:p-2 rounded-full transition-colors duration-300 ${
                          chat.favorite 
                            ? 'text-yellow-400 bg-yellow-400 bg-opacity-10' 
                            : (isDarkMode ? 'text-gray-500 hover:text-yellow-400' : 'text-gray-400 hover:text-yellow-500')
                        }`}
                        aria-label={chat.favorite ? "Quitar de favoritos" : "Añadir a favoritos"}
                      >
                        {/* Estrella de favoritos que se rellena */}
                        <FiStar className={`w-5 h-5 2xl:w-6 2xl:h-6 ${chat.favorite ? "fill-current" : ""}`} />
                      </button>
                    </div>

                    {/* Botones de acción inferiores */}
                    <div className="mt-2 2xl:mt-3 flex space-x-2 2xl:space-x-3">
                      {/* Botón Abrir */}
                      <button
                        onClick={() => onOpenConversation(chat.id)}
                        className={`flex items-center gap-1.5 2xl:gap-2 px-3 py-1.5 2xl:px-4 2xl:py-2 text-xs sm:text-sm 2xl:text-base rounded-full shadow-md 
                          transition-all duration-300 hover:scale-105 
                          ${
                            isDarkMode
                              ? "bg-blue-400 hover:bg-blue-300 text-gray-900"
                              : "bg-blue-500 hover:bg-blue-600 text-white"
                          }`}
                      >
                        <FiMessageSquare className="w-4 h-4 2xl:w-5 2xl:h-5" />
                        Abrir
                      </button>

                      {/* Botón Eliminar (solo visual, no BBDD) */}
                      <button
                        onClick={() => {
                          setChatToDelete(chat.id);
                          openModal();
                        }}
                        className={`flex items-center gap-1.5 2xl:gap-2 px-3 py-1.5 2xl:px-4 2xl:py-2 text-xs sm:text-sm 2xl:text-base rounded-full shadow-md 
                          transition-all duration-300 hover:scale-105
                          ${isDarkMode 
                            ? "bg-red-500 hover:bg-red-400 text-white" 
                            : "bg-red-600 hover:bg-red-500 text-white"
                          }`}
                      >
                        <FiTrash2 className="w-4 h-4 2xl:w-5 2xl:h-5" />
                        Eliminar
                      </button>
                    </div>
                  </div>
                ))
              )}
            </>
          )}

          {/* MODAL DE ELIMINACIÓN */}
          <DeleteChatModal
            isDarkMode={isDarkMode}
            isOpen={showDeleteModal}
            isClosing={isClosing}
            onClose={closeModal}
            onConfirm={handleConfirmDelete}
          />
        </div>
      </div>
    </div>
  );
};
