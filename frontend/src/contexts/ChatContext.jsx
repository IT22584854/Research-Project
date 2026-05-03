import { createContext, useContext, useState, useCallback, useRef, useEffect } from 'react';
import { v4 as uuidv4 } from 'uuid';
import api, { ApiError } from '../services/api';
import { useLocalStorage } from '../hooks/useLocalStorage';

const ChatContext = createContext();

const STORAGE_KEY = 'mta_conversations';

function canonicalizeText(value) {
  return typeof value === 'string'
    ? value
      .trim()
      .toLowerCase()
      .replace(/[\W_]+/g, ' ')
      .replace(/\s+/g, ' ')
    : '';
}

function canonicalizeUrl(value) {
  if (typeof value !== 'string' || !value.trim()) return '';

  try {
    const parsed = new URL(value.trim());
    parsed.search = '';
    parsed.hash = '';
    const normalized = `${parsed.protocol}//${parsed.hostname}${parsed.pathname}`;
    return normalized.replace(/\/$/, '').toLowerCase();
  } catch {
    return '';
  }
}

function dedupeSources(sources) {
  if (!Array.isArray(sources) || sources.length === 0) return [];

  const seen = new Set();
  const unique = [];

  for (const source of sources) {
    if (!source || typeof source !== 'object') continue;

    const derivedUrl = source.url || (typeof source.source === 'string' && /^https?:\/\//i.test(source.source) ? source.source : '');
    const urlKey = canonicalizeUrl(derivedUrl);
    const textKey = canonicalizeText(source.title || source.source);
    const dedupeKey = urlKey || textKey;

    if (!dedupeKey || seen.has(dedupeKey)) continue;

    seen.add(dedupeKey);
    unique.push({
      ...source,
      ...(derivedUrl && !source.url ? { url: derivedUrl } : {}),
    });
  }

  return unique;
}

/**
 * ChatProvider manages all chat state and operations
 */
export function ChatProvider({ children }) {
  const [conversations, setConversations] = useLocalStorage(STORAGE_KEY, []);
  const [activeId, setActiveId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const sessionIdRef = useRef(null);

  const getConversationTimestamp = () => Date.now();

  // Get active conversation
  const activeConversation = conversations.find(c => c.id === activeId);

  // Load messages when active conversation changes
  useEffect(() => {
    if (activeConversation) {
      setMessages(activeConversation.messages || []);
      sessionIdRef.current = activeConversation.sessionId;
    } else {
      setMessages([]);
      sessionIdRef.current = null;
    }
  }, [activeId, activeConversation]);

  // Helper: Update conversations in state and localStorage
  const updateConversations = useCallback((updater) => {
    setConversations(prev => {
      const updated = typeof updater === 'function' ? updater(prev) : updater;
      return updated;
    });
  }, [setConversations]);

  // Helper: Update active conversation
  const updateActiveConversation = useCallback((updater) => {
    if (!activeId) return;

    updateConversations(prev =>
      prev.map(c => {
        if (c.id === activeId) {
          return typeof updater === 'function' ? updater(c) : { ...c, ...updater };
        }
        return c;
      })
    );
  }, [activeId, updateConversations]);

  /**
   * Start a new chat
   */
  const startNewChat = useCallback(() => {
    const id = uuidv4();
    const sessionId = uuidv4();
    const now = getConversationTimestamp();
    sessionIdRef.current = sessionId;

    const newConv = {
      id,
      sessionId,
      title: 'New Conversation',
      createdAt: now,
      updatedAt: now,
      messages: [],
    };

    updateConversations(prev => [newConv, ...prev]);
    setActiveId(id);
    setMessages([]);
    setError(null);
  }, [updateConversations]);

  /**
   * Open an existing conversation
   */
  const openConversation = useCallback((convId) => {
    setActiveId(convId);
    setError(null);
  }, []);

  /**
   * Delete a conversation
   */
  const deleteConversation = useCallback((id) => {
    updateConversations(prev => prev.filter(c => c.id !== id));
    
    if (id === activeId) {
      setActiveId(null);
      setMessages([]);
      sessionIdRef.current = null;
    }
  }, [activeId, updateConversations]);

  /**
   * Send a message
   */
  const sendMessage = useCallback(async (text) => {
    if (!text.trim() || isLoading) return;

    // Ensure there's an active conversation
    let convId = activeId;
    let sessionId = sessionIdRef.current;

    if (!convId) {
      convId = uuidv4();
      sessionId = uuidv4();
      const now = getConversationTimestamp();
      sessionIdRef.current = sessionId;
      
      const newConv = {
        id: convId,
        sessionId,
        title: text.slice(0, 45),
        createdAt: now,
        updatedAt: now,
        messages: [],
      };
      
      updateConversations(prev => [newConv, ...prev]);
      setActiveId(convId);
    }

    const userMsg = {
      id: uuidv4(),
      role: 'user',
      content: text,
      timestamp: Date.now(),
    };

    const nextMessages = [...messages, userMsg];
    setMessages(nextMessages);
    setIsLoading(true);
    setError(null);

    // Persist user message immediately
    updateActiveConversation(c => ({
      ...c,
      messages: nextMessages,
      title: c.title === 'New Conversation' ? text.slice(0, 45) : c.title,
      updatedAt: getConversationTimestamp(),
    }));

    try {
      const data = await api.sendChatMessage(text, sessionId);
      const aiMsg = {
        id: uuidv4(),
        role: 'assistant',
        content: data.response,
        sources: dedupeSources(data.sources),
        timestamp: Date.now(),
      };
      
      const finalMessages = [...nextMessages, aiMsg];
      setMessages(finalMessages);
      updateActiveConversation(c => ({ ...c, messages: finalMessages, updatedAt: getConversationTimestamp() }));
    } catch (err) {
      const errorMessage = err instanceof ApiError
        ? err.message
        : 'Something went wrong. Please try again.';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [activeId, isLoading, messages, updateConversations, updateActiveConversation]);

  /**
   * Edit a message and resend
   */
  const editMessage = useCallback(async (messageId, newContent) => {
    if (!activeId || !newContent.trim() || isLoading) return;

    // Find the message index
    const messageIndex = messages.findIndex(m => m.id === messageId);
    if (messageIndex === -1 || messages[messageIndex].role !== 'user') return;

    // Remove messages after the edited one
    const updatedMessages = messages.slice(0, messageIndex);
    
    // Add edited message
    const editedMsg = {
      ...messages[messageIndex],
      content: newContent,
      edited: true,
      editedAt: Date.now(),
    };

    const nextMessages = [...updatedMessages, editedMsg];
    setMessages(nextMessages);
    setIsLoading(true);
    setError(null);

    // Persist immediately
    updateActiveConversation(c => ({ ...c, messages: nextMessages, updatedAt: getConversationTimestamp() }));

    try {
      const data = await api.sendChatMessage(newContent, sessionIdRef.current);
      const aiMsg = {
        id: uuidv4(),
        role: 'assistant',
        content: data.response,
        sources: dedupeSources(data.sources),
        timestamp: Date.now(),
      };
      
      const finalMessages = [...nextMessages, aiMsg];
      setMessages(finalMessages);
      updateActiveConversation(c => ({ ...c, messages: finalMessages, updatedAt: getConversationTimestamp() }));
    } catch (err) {
      const errorMessage = err instanceof ApiError
        ? err.message
        : 'Failed to send edited message.';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [activeId, isLoading, messages, updateActiveConversation]);

  /**
   * Regenerate the last assistant response
   */
  const regenerateResponse = useCallback(async () => {
    if (!activeId || isLoading || messages.length < 2) return;

    // Find the last user message
    let lastUserMsgIndex = -1;
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === 'user') {
        lastUserMsgIndex = i;
        break;
      }
    }

    if (lastUserMsgIndex === -1) return;

    const lastUserMsg = messages[lastUserMsgIndex];
    
    // Remove messages after the last user message
    const updatedMessages = messages.slice(0, lastUserMsgIndex + 1);
    setMessages(updatedMessages);
    setIsLoading(true);
    setError(null);

    try {
      const data = await api.sendChatMessage(lastUserMsg.content, sessionIdRef.current);
      const aiMsg = {
        id: uuidv4(),
        role: 'assistant',
        content: data.response,
        sources: dedupeSources(data.sources),
        timestamp: Date.now(),
        regenerated: true,
      };
      
      const finalMessages = [...updatedMessages, aiMsg];
      setMessages(finalMessages);
      updateActiveConversation(c => ({ ...c, messages: finalMessages, updatedAt: getConversationTimestamp() }));
    } catch (err) {
      const errorMessage = err instanceof ApiError
        ? err.message
        : 'Failed to regenerate response.';
      setError(errorMessage);
    } finally {
      setIsLoading(false);
    }
  }, [activeId, isLoading, messages, updateActiveConversation]);

  /**
   * Copy message to clipboard
   */
  const copyMessage = useCallback(async (messageId) => {
    const message = messages.find(m => m.id === messageId);
    if (!message) return false;

    try {
      await navigator.clipboard.writeText(message.content);
      return true;
    } catch (err) {
      console.error('Failed to copy message:', err);
      return false;
    }
  }, [messages]);

  /**
   * Export conversation as JSON
   */
  const exportConversationJSON = useCallback(() => {
    if (!activeConversation) return null;
    
    const exportData = {
      ...activeConversation,
      exportedAt: new Date().toISOString(),
    };
    
    return JSON.stringify(exportData, null, 2);
  }, [activeConversation]);

  /**
   * Export conversation as Markdown
   */
  const exportConversationMarkdown = useCallback(() => {
    if (!activeConversation) return null;
    
    let markdown = `# ${activeConversation.title}\n\n`;
    markdown += `Exported: ${new Date().toLocaleString()}\n\n`;
    markdown += `---\n\n`;
    
    activeConversation.messages.forEach(msg => {
      const role = msg.role === 'user' ? 'User' : 'Assistant';
      markdown += `## ${role}\n\n`;
      markdown += `${msg.content}\n\n`;
      
      if (msg.sources && msg.sources.length > 0) {
        const uniqueSources = dedupeSources(msg.sources);

        if (uniqueSources.length === 0) {
          markdown += `---\n\n`;
          return;
        }

        markdown += `### Sources\n\n`;
        uniqueSources.forEach((source, idx) => {
          markdown += `${idx + 1}. ${source.source}`;
          if (source.url) markdown += ` - [Link](${source.url})`;
          markdown += `\n`;
        });
        markdown += `\n`;
      }
      
      markdown += `---\n\n`;
    });
    
    return markdown;
  }, [activeConversation]);

  const value = {
    // State
    conversations,
    activeId,
    messages,
    isLoading,
    error,
    activeConversation,
    
    // Actions
    startNewChat,
    openConversation,
    deleteConversation,
    sendMessage,
    editMessage,
    regenerateResponse,
    copyMessage,
    exportConversationJSON,
    exportConversationMarkdown,
  };

  return (
    <ChatContext.Provider value={value}>
      {children}
    </ChatContext.Provider>
  );
}

/**
 * Hook to use the chat context
 */
export function useChat() {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChat must be used within ChatProvider');
  }
  return context;
}
