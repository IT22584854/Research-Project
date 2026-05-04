import { Sun, Moon, Monitor, Download, Trash2, Keyboard } from 'lucide-react';
import { useState } from 'react';
import { useTheme, THEME_MODES } from '../../contexts/ThemeContext';
import { useChat } from '../../contexts/ChatContext';
import { useToast } from '../shared/Toast';
import { Modal } from '../shared/Modal';
import { KeyboardShortcuts } from '../shared/KeyboardShortcuts';

/**
 * Settings panel component
 */
export function SettingsPanel({ isOpen, onClose }) {
  const { themeMode, setTheme, actualTheme } = useTheme();
  const { activeConversation, exportConversationJSON, exportConversationMarkdown, deleteConversation } = useChat();
  const { success, error: showError } = useToast();
  const [showShortcuts, setShowShortcuts] = useState(false);

  const themeOptions = [
    { value: THEME_MODES.LIGHT, label: 'Light', icon: Sun },
    { value: THEME_MODES.DARK, label: 'Dark', icon: Moon },
    { value: THEME_MODES.SYSTEM, label: 'System', icon: Monitor },
  ];

  const handleExportJSON = () => {
    const data = exportConversationJSON();
    if (!data) {
      showError('No active conversation to export');
      return;
    }

    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-${activeConversation.id}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
    success('Conversation exported as JSON');
  };

  const handleExportMarkdown = () => {
    const data = exportConversationMarkdown();
    if (!data) {
      showError('No active conversation to export');
      return;
    }

    const blob = new Blob([data], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-${activeConversation.id}-${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
    success('Conversation exported as Markdown');
  };

  const handleClearHistory = () => {
    if (window.confirm('Are you sure you want to clear all conversation history? This cannot be undone.')) {
      // Get all conversation IDs
      const allConversations = JSON.parse(localStorage.getItem('mta_conversations') || '[]');
      allConversations.forEach(conv => {
        deleteConversation(conv.id);
      });
      success('All conversations cleared');
      onClose();
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      title="Settings"
      size="md"
    >
      <div className="settings-panel">
        {/* Theme Settings */}
        <section className="settings-section">
          <h3 className="settings-section-title">
            Appearance
          </h3>
          <div className="settings-theme-grid">
            {themeOptions.map(({ value, label, icon: Icon }) => (
              <button
                key={value}
                onClick={() => setTheme(value)}
                className={`settings-theme-option ${themeMode === value ? 'active' : ''}`}
                aria-label={`Set theme to ${label}`}
              >
                <Icon size={18} />
                <span>
                  {label}
                </span>
              </button>
            ))}
          </div>
          <p className="settings-muted">
            Current: {actualTheme === THEME_MODES.DARK ? 'Dark' : 'Light'}
            {themeMode === THEME_MODES.SYSTEM && ' (auto)'}
          </p>
        </section>

        {/* Export Options */}
        <section className="settings-section">
          <h3 className="settings-section-title">
            Export Current Conversation
          </h3>
          <div className="settings-action-row">
            <button
              onClick={handleExportJSON}
              disabled={!activeConversation}
              className="ui-btn ui-btn-secondary ui-btn-sm settings-action-btn"
            >
              <Download size={15} />
              JSON
            </button>
            <button
              onClick={handleExportMarkdown}
              disabled={!activeConversation}
              className="ui-btn ui-btn-secondary ui-btn-sm settings-action-btn"
            >
              <Download size={15} />
              Markdown
            </button>
          </div>
          {!activeConversation && (
            <p className="settings-muted">
              Start a conversation to enable export
            </p>
          )}
        </section>

        {/* Data Management */}
        <section className="settings-section">
          <h3 className="settings-section-title">
            Data Management
          </h3>
          <button
            onClick={handleClearHistory}
            className="ui-btn ui-btn-danger ui-btn-sm settings-danger-btn"
          >
            <Trash2 size={15} />
            Clear All Conversations
          </button>
          <p className="settings-muted">
            This will permanently delete all your conversation history
          </p>
        </section>

        {/* App Info */}
        <section className="settings-section settings-section-separated">
          <div className="settings-section-header-row">
            <h3 className="settings-section-title">
              About
            </h3>
            <button
              onClick={() => setShowShortcuts(true)}
              className="settings-link-btn"
            >
              <Keyboard size={13} />
              Shortcuts
            </button>
          </div>
          <div className="settings-app-meta">
            <p className="settings-app-name">Sri lankan Health Assistant</p>
            <p>Version 2.0.0</p>
            <p>Built with React + Vite</p>
          </div>
        </section>
      </div>

      {/* Keyboard Shortcuts Modal */}
      <Modal
        isOpen={showShortcuts}
        onClose={() => setShowShortcuts(false)}
        title="Keyboard Shortcuts"
        size="md"
      >
        <KeyboardShortcuts />
      </Modal>
    </Modal>
  );
}
