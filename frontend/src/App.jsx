import { useState, useEffect } from 'react';
import { Settings } from 'lucide-react';
import Sidebar from './components/Sidebar';
import ChatWindow from './components/ChatWindow';
import ChatInput from './components/ChatInput';
import { SettingsPanel } from './components/layout/SettingsPanel';
import { useChat } from './contexts/ChatContext';

export default function App() {
  const [showSettings, setShowSettings] = useState(false);
  const {
    conversations,
    activeId,
    messages,
    isLoading,
    error,
    sendMessage,
    startNewChat,
    openConversation,
    deleteConversation,
  } = useChat();
  // Global keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Ctrl+N or Cmd+N: New conversation
      if ((e.ctrlKey || e.metaKey) && e.key === 'n') {
        e.preventDefault();
        startNewChat();
      }
      
      // Ctrl+, or Cmd+,: Open settings
      if ((e.ctrlKey || e.metaKey) && e.key === ',') {
        e.preventDefault();
        setShowSettings(true);
      }
      
      // Escape: Close settings
      if (e.key === 'Escape' && showSettings) {
        setShowSettings(false);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [showSettings, startNewChat]);

  return (
    <div className="app-layout relative">
      {/* Screen reader announcements for new messages */}
      <div
        role="status"
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
      >
        {messages.length > 0 && messages[messages.length - 1]?.role === 'assistant' && (
          `New message from assistant`
        )}
      </div>

      {/* Sidebar */}
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        onNewChat={startNewChat}
        onOpen={openConversation}
        onDelete={deleteConversation}
      />

      {/* Main Chat Area */}
      <main className="chat-area" role="main">
        {/* Top Bar */}
        <header className="chat-topbar">
          <div className="topbar-copy">
            <div className="topbar-favicon" aria-hidden="true">🩺</div>
            <div>
              <h1 className="topbar-title">Sri lankan Health Assistant</h1>
              <p className="topbar-subtitle">Evidence-based health guidance with citations and safety framing</p>
            </div>
          </div>
          <div className="topbar-actions">
            <div className="topbar-pill" role="status">
              <span className="topbar-pill-dot" aria-hidden="true" />
              Source-backed
            </div>
            <button
              onClick={() => setShowSettings(true)}
              className="topbar-settings-btn"
              title="Settings (Ctrl+,)"
              aria-label="Open settings"
            >
              <Settings size={18} />
            </button>
          </div>
        </header>

        {/* Messages */}
        <ChatWindow
          messages={messages}
          isLoading={isLoading}
          error={error}
          onSuggestion={sendMessage}
        />

        {/* Input */}
        <ChatInput onSend={sendMessage} disabled={isLoading} />
      </main>

      {/* Settings Panel */}
      <SettingsPanel isOpen={showSettings} onClose={() => setShowSettings(false)} />
    </div>
  );
}
