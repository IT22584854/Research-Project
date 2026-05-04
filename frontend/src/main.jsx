<<<<<<< HEAD
import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.jsx';
import './styles.css';

createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
=======
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.jsx'
import { ThemeProvider } from './contexts/ThemeContext'
import { ChatProvider } from './contexts/ChatContext'
import { ToastProvider } from './components/shared/Toast'

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <ThemeProvider>
      <ToastProvider>
        <ChatProvider>
          <App />
        </ChatProvider>
      </ToastProvider>
    </ThemeProvider>
  </StrictMode>,
)
>>>>>>> ce9f875a5dd78bca583d8d1fdbe619d2185e2dd1
