import React, { useState, useEffect, useRef } from "react";
import { FiMic, FiStar, FiTrash2, FiClock, FiArrowRightCircle } from "react-icons/fi";
import DeleteChatModal from "../../components/utils/DeleteChatModal";

export default function VoiceConversationHistoryPage({ isDarkMode, onOpenConversation }) {
  const [activeTab, setActiveTab] = useState("historial");
  const [indicatorStyle, setIndicatorStyle] = useState({});
  const tabsRef = useRef({});
  const [chats, setChats] = useState([
    { id: 1, title: "Asistente de voz - Tarea pendiente", date: "Hace 2 horas", favorite: true },
    { id: 2, title: "Recordatorio de eventos", date: "Ayer", favorite: false },
    { id: 3, title: "Búsqueda por voz de información", date: "Hace 3 días", favorite: true },
    { id: 4, title: "Conversación con COSMOS", date: "Hace 1 semana", favorite: false },
  ]);

  const [chatToDelete, setChatToDelete] = useState(null);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [isClosing, setIsClosing] = useState(false);

  const openModal = () => setShowDeleteModal(true);
  const closeModal = () => {
    setIsClosing(true);
    setTimeout(() => {
      setShowDeleteModal(false);
      setIsClosing(false);
    }, 300);
  };

  // Animación del indicador de pestaña
  useEffect(() => {
    if (tabsRef.current[activeTab]) {
        const el = tabsRef.current[activeTab];
        setIndicatorStyle({
        left: el.offsetLeft,
        width: el.offsetWidth,
        transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
        });
    }
  }, [activeTab, isDarkMode]);

  const toggleFavorite = (id) => {
    setChats(chats.map((c) => (c.id === id ? { ...c, favorite: !c.favorite } : c)));
  };

  const filteredChats =
    activeTab === "favoritos" ? chats.filter((c) => c.favorite) : chats;

  return (
    <div
      key={isDarkMode ? "dark" : "light"} // Fuerza para que la pagina y el Header cambien a la vez a modo oscuro
      className={`min-h-screen p-4 2xl:p-6 transition-colors duration-300 ${
        isDarkMode ? "bg-gray-900 text-white" : "bg-white text-gray-800"
      }`}
    >
      <div className="max-w-3xl 2xl:max-w-4xl mx-auto h-[calc(100vh-2rem)] flex flex-col">
        {/* Header */}
        <header className="mb-4 2xl:mb-6 text-center">
          <div className="flex justify-center mb-2 2xl:mb-3">
            <div
                className={`p-2 2xl:p-3 rounded-full shadow-lg ${
                  isDarkMode ? "bg-blue-800/30 text-blue-400" : "bg-blue-100 text-blue-700"
                }`}
              >
              <FiMic className="w-5 h-5 2xl:w-6 2xl:h-6" />
            </div>
          </div>
          <h1
              className={`text-xl md:text-2xl 2xl:text-3xl font-extrabold ${isDarkMode ? 'text-blue-400' : 'text-blue-700'}`}
          >
              Conversaciones por voz
          </h1>
          <p
              className={`mt-1 2xl:mt-2 text-xs md:text-sm 2xl:text-base ${
                isDarkMode ? "text-gray-400" : "text-gray-600"
              }`}
          >
              Historial de interacciones con el Agente de Voz COSMOS
          </p>
        </header>

        {/* Tabs */}
        <div className="relative mb-4 2xl:mb-6">
          <div
            className={`flex border-b ${
              isDarkMode ? "border-gray-700" : "border-gray-300"
            }`}
          >
            <button
              ref={(el) => (tabsRef.current.historial = el)}
              onClick={() => setActiveTab("historial")}
              className={`flex-1 py-2 2xl:py-3 text-center text-sm 2xl:text-base font-medium flex items-center justify-center gap-2 transition-colors ${
                activeTab === "historial"
                  ? isDarkMode ? "text-white" : "text-blue-700"
                  : isDarkMode ? "text-gray-400 hover:text-white" : "text-gray-500 hover:text-blue-700"
              }`}
            >
              <FiClock className="w-4 h-4 2xl:w-5 2xl:h-5" /> Historial
            </button>

            <button
              ref={(el) => (tabsRef.current.favoritos = el)}
              onClick={() => setActiveTab("favoritos")}
              className={`flex-1 py-2 2xl:py-3 text-center text-sm 2xl:text-base font-medium flex items-center justify-center gap-2 transition-colors ${
                activeTab === "favoritos"
                  ? isDarkMode ? "text-yellow-400" : "text-yellow-500"
                  : isDarkMode ? "text-gray-400 hover:text-yellow-400" : "text-gray-500 hover:text-yellow-500"
              }`}
            >
                <FiStar 
                  className={`w-4 h-4 2xl:w-5 2xl:h-5 ${
                    activeTab === "favoritos" ? "fill-current" : ""
                  }`} 
                />
                Favoritos
            </button>
          </div>
          <div
            className="absolute bottom-0 h-[2px] bg-gradient-to-r from-blue-500 to-purple-500"
            style={indicatorStyle}
          />
        </div>

        {/* Lista de conversaciones */}
        <div className="flex-1 space-y-2 2xl:space-y-3 pb-4 overflow-y-auto scrollbar-hide">
          {filteredChats.length === 0 ? (
            <div
              className={`text-center py-8 2xl:py-10 rounded-xl ${
              isDarkMode ? "bg-gray-800" : "bg-white"
              }`}
            >
              <FiMic className="w-8 h-8 2xl:w-10 2xl:h-10 mx-auto text-gray-400 mb-2"/>
              <p className="text-base 2xl:text-lg font-medium">
                {activeTab === "favoritos"
                  ? "No hay conversaciones favoritas"
                  : "No hay conversaciones registradas en el historial"}
              </p>
              <p
                className={`mt-1 2xl:mt-2 text-sm 2xl:text-base ${
                  isDarkMode ? "text-gray-400" : "text-gray-600"
                }`}
              >
                {activeTab === "favoritos"
                  ? "Marca tus conversaciones destacadas para acceder rápido a ellas"
                  : "Tus interacciones por voz con el agente de voz COSMOS aparecerán aquí"}
              </p>
            </div>
          ) : (
            filteredChats.map((chat) => (
              <div
                key={chat.id}
                className={`p-3 2xl:p-4 rounded-xl 2xl:rounded-2xl flex flex-col sm:flex-row sm:items-center justify-between transition-all duration-300 gap-2 sm:gap-4 ${
                  isDarkMode
                    ? "bg-gray-800 hover:bg-gray-700"
                    : "bg-white hover:bg-blue-50"
                }`}
              >
                <div className="min-w-0">
                  <h3 className="font-semibold text-sm sm:text-base 2xl:text-lg truncate">{chat.title}</h3>
                  <p
                    className={`text-xs 2xl:text-sm mt-0.5 2xl:mt-1 ${
                      isDarkMode ? "text-gray-400" : "text-gray-600"
                    }`}
                  >
                    {chat.date}
                  </p>
                </div>

                {/* Botones de acción */}
                <div className="flex gap-2 mt-3 sm:mt-0">
                  {/* Abrir conversación */}
                  <button
                    onClick={() => onOpenConversation(chat.id)}
                    className={`p-1.5 2xl:p-2 rounded-full transition ${
                      isDarkMode
                        ? "text-blue-400 hover:text-blue-300"
                        : "text-blue-600 hover:text-blue-800"
                    }`}
                    title="Abrir"
                  >
                    <FiArrowRightCircle className="w-4 h-4 2xl:w-6 2xl:h-6" />
                  </button>

                  {/* Favorito */}
                  <button
                    onClick={() => toggleFavorite(chat.id)}
                    className={`p-1.5 2xl:p-2 rounded-full transition ${
                      chat.favorite
                        ? "text-yellow-400 bg-yellow-400/10"
                        : isDarkMode
                        ? "text-gray-400 hover:text-yellow-400"
                        : "text-gray-500 hover:text-yellow-500"
                    }`}
                    title="Favorito"
                  >
                    <FiStar 
                      className={`w-4 h-4 2xl:w-6 2xl:h-6 ${
                        chat.favorite ? "fill-current" : ""
                      }`} 
                    />
                  </button>

                  {/* Eliminar */}
                  <button
                    onClick={() => {
                      setChatToDelete(chat.id);
                      openModal();
                    }}
                    className={`p-1.5 2xl:p-2 rounded-full transition ${
                      isDarkMode
                        ? "text-red-400 hover:text-red-300"
                        : "text-red-500 hover:text-red-600"
                    }`}
                    title="Eliminar"
                  >
                    <FiTrash2 className="w-4 h-4 2xl:w-6 2xl:h-6" />
                  </button>
                </div>
              </div>
            ))
          )}
        </div>

        {/* Modal de eliminación */}
        <DeleteChatModal
          isDarkMode={isDarkMode}
          isOpen={showDeleteModal}
          isClosing={isClosing}
          onClose={closeModal}
          onConfirm={() => {
            setChats(chats.filter((c) => c.id !== chatToDelete));
            closeModal();
          }}
        />
      </div>
    </div>
  );
}