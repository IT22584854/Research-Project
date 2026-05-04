import { useState } from 'react';
import { Copy, Edit2, RefreshCw, Check } from 'lucide-react';
import { useChat } from '../../contexts/ChatContext';
import { useToast } from '../shared/Toast';

/**
 * Message action buttons component (copy, edit, regenerate)
 */
export function MessageActions({ message, isLastAssistantMessage }) {
  const [copied, setCopied] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editContent, setEditContent] = useState(message.content);
  const { copyMessage, editMessage, regenerateResponse, isLoading } = useChat();
  const { success, error: showError } = useToast();

  const handleCopy = async () => {
    const result = await copyMessage(message.id);
    if (result) {
      setCopied(true);
      success('Copied to clipboard');
      setTimeout(() => setCopied(false), 2000);
    } else {
      showError('Failed to copy message');
    }
  };

  const handleEdit = () => {
    setIsEditing(true);
    setEditContent(message.content);
  };

  const handleSaveEdit = async () => {
    if (editContent.trim() && editContent !== message.content) {
      await editMessage(message.id, editContent);
      setIsEditing(false);
    }
  };

  const handleCancelEdit = () => {
    setIsEditing(false);
    setEditContent(message.content);
  };

  const handleRegenerate = async () => {
    await regenerateResponse();
  };

  if (isEditing) {
    return (
      <div className="message-edit-shell">
        <textarea
          value={editContent}
          onChange={(e) => setEditContent(e.target.value)}
          className="message-edit-textarea"
          autoFocus
          onKeyDown={(e) => {
            if (e.key === 'Escape') handleCancelEdit();
            if (e.key === 'Enter' && e.ctrlKey) handleSaveEdit();
          }}
        />
        <div className="message-edit-buttons">
          <button
            onClick={handleSaveEdit}
            disabled={!editContent.trim() || editContent === message.content}
            className="ui-btn ui-btn-primary ui-btn-sm"
          >
            Save (Ctrl+Enter)
          </button>
          <button
            onClick={handleCancelEdit}
            className="ui-btn ui-btn-secondary ui-btn-sm"
          >
            Cancel (Esc)
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="message-actions">
      {/* Copy button */}
      <button
        onClick={handleCopy}
        className="message-action-btn"
        title="Copy message"
        aria-label="Copy message to clipboard"
      >
        {copied ? (
          <Check size={15} className="message-action-icon message-action-icon-success" />
        ) : (
          <Copy size={15} className="message-action-icon" />
        )}
      </button>

      {/* Edit button (only for user messages) */}
      {message.role === 'user' && (
        <button
          onClick={handleEdit}
          disabled={isLoading}
          className="message-action-btn"
          title="Edit message"
          aria-label="Edit and resend message"
        >
          <Edit2 size={15} className="message-action-icon" />
        </button>
      )}

      {/* Regenerate button (only for last assistant message) */}
      {message.role === 'assistant' && isLastAssistantMessage && (
        <button
          onClick={handleRegenerate}
          disabled={isLoading}
          className="message-action-btn"
          title="Regenerate response"
          aria-label="Regenerate assistant response"
        >
          <RefreshCw size={15} className={`message-action-icon ${isLoading ? 'message-action-spin' : ''}`} />
        </button>
      )}

      {/* Edited indicator */}
      {message.edited && (
        <span className="message-action-label">
          (edited)
        </span>
      )}

      {/* Regenerated indicator */}
      {message.regenerated && (
        <span className="message-action-label">
          (regenerated)
        </span>
      )}
    </div>
  );
}
