import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App.jsx';
import { BrowserRouter } from "react-router-dom";
import { SettingsProvider } from './hooks/useSettings';

import '@fortawesome/fontawesome-free/css/all.min.css'; // Librería para importar los íconos de FontAwesome

// Este código se encargará de renderizar el App.jsx en el contenedor del index.html para poder visualizar la interfaz
const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <SettingsProvider>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </SettingsProvider>
  </React.StrictMode>
);