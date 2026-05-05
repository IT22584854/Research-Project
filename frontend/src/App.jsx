import { useEffect, useMemo, useRef, useState } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  Activity,
  AlertCircle,
  Bot,
  Copy,
  ExternalLink,
  Loader2,
  Plus,
  Send,
  Shield,
  Stethoscope,
  Trash2,
  UserRound,
} from 'lucide-react';
import { v4 as uuidv4 } from 'uuid';
import AdminLogin from './components/admin/AdminLogin';
import AdminLayout from './components/admin/AdminLayout';

const STORAGE_KEY = 'medtriage_conversations';

function getApiBase() {
  const runtimeBase = window.__APP_CONFIG__?.VITE_API_BASE;
  const envBase = import.meta.env.VITE_API_BASE;
  const fallbackBase = import.meta.env.DEV ? 'http://127.0.0.1:8000' : '';
  return (runtimeBase || envBase || fallbackBase).replace(/\/$/, '');
}

async function apiRequest(path, options = {}) {
  const base = getApiBase();
  if (!base) {
    throw new Error('API base is not configured. Set VITE_API_BASE or runtime-config.js.');
  }

  const response = await fetch(`${base}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
    ...options,
  });

  const contentType = response.headers.get('content-type') || '';
  const data = contentType.includes('application/json') ? await response.json() : await response.text();

  if (!response.ok) {
    const detail = typeof data === 'object' ? data.detail || data.message : data;
    throw new Error(detail || `Request failed with HTTP ${response.status}`);
  }

  return data;
}

function loadConversations() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
  } catch {
    return [];
  }
}

function saveConversations(conversations) {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(conversations));
}

function useHealth() {
  const [health, setHealth] = useState({ status: 'checking', detail: 'Checking backend' });

  useEffect(() => {
    let cancelled = false;

    apiRequest('/api/health')
      .then((data) => {
        if (!cancelled) {
          setHealth({ status: data.status === 'ok' ? 'ok' : 'warn', detail: data.service || 'Backend reachable' });
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setHealth({ status: 'error', detail: error.message });
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return health;
}

function SourceList({ sources }) {
  if (!sources?.length) return null;

  return (
    <section className="sources" aria-label="Sources">
      <div className="sourcesHeader">
        <span>Sources</span>
        <strong>{sources.length}</strong>
      </div>
      <div className="sourcesGrid">
        {sources.map((source, index) => {
          const label = source.title || source.source || `Source ${index + 1}`;
          const href = source.url || (/^https?:\/\//i.test(source.source || '') ? source.source : '');

          if (href) {
            return (
              <a className="sourceItem" href={href} target="_blank" rel="noreferrer" key={`${label}-${index}`}>
                <span>{label}</span>
                <ExternalLink size={14} />
              </a>
            );
          }

          return (
            <span className="sourceItem sourceStatic" key={`${label}-${index}`}>
              {label}
            </span>
          );
        })}
      </div>
    </section>
  );
}

function Message({ message, onCopy }) {
  const isUser = message.role === 'user';

  return (
    <article className={`message ${isUser ? 'messageUser' : 'messageAssistant'}`}>
      <div className="avatar" aria-hidden="true">
        {isUser ? <UserRound size={18} /> : <Bot size={18} />}
      </div>
      <div className="messageBody">
        <div className="messageMeta">
          <span>{isUser ? 'You' : 'MedTriage AI'}</span>
          <button className="iconButton" type="button" onClick={() => onCopy(message.content)} aria-label="Copy message">
            <Copy size={14} />
          </button>
        </div>
        <div className="bubble">
          {isUser ? (
            <p>{message.content}</p>
          ) : (
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown>
          )}
        </div>
        {!isUser && <SourceList sources={message.sources} />}
      </div>
    </article>
  );
}

function EmptyState({ onPrompt }) {
  const prompts = [
    'What are common dengue symptoms?',
    'When should someone seek medical care for fever?',
    'Latest dengue outbreak advisory in Sri Lanka',
  ];

  return (
    <div className="emptyState">
      <div className="emptyIcon">
        <Stethoscope size={34} />
      </div>
      <h2>Medical information assistant</h2>
      <p>Ask a public-health question and get a source-backed response from the backend agent.</p>
      <div className="promptGrid">
        {prompts.map((prompt) => (
          <button key={prompt} type="button" onClick={() => onPrompt(prompt)}>
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [conversations, setConversations] = useState(() => loadConversations());
  const [activeId, setActiveId] = useState(() => loadConversations()[0]?.id || null);
  const [draft, setDraft] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const health = useHealth();

  /*
   * ADMIN PANEL ROUTING
   * -------------------
   * We use simple React state ('currentView') instead of react-router-dom
   * to keep the dependency footprint small. The admin panel is a secondary
   * view embedded inside the existing chat SPA. If multi-tab deep-linking
   * or URL-based routing is ever needed, replace this with a proper router
   * (e.g. react-router-dom v6+).
   *
   * Views: 'chat' | 'admin-login' | 'admin-panel'
   */
  const [currentView, setCurrentView] = useState('chat');
  const bottomRef = useRef(null);

  const activeConversation = useMemo(
    () => conversations.find((conversation) => conversation.id === activeId) || null,
    [activeId, conversations],
  );
  const messages = activeConversation?.messages || [];

  useEffect(() => {
    saveConversations(conversations);
  }, [conversations]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  function upsertConversation(updater) {
    setConversations((current) => {
      const next = updater(current);
      saveConversations(next);
      return next;
    });
  }

  function startConversation(initialMessage = '') {
    const id = uuidv4();
    const sessionId = uuidv4();
    const conversation = {
      id,
      sessionId,
      title: initialMessage ? initialMessage.slice(0, 48) : 'New conversation',
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messages: [],
    };
    upsertConversation((current) => [conversation, ...current]);
    setActiveId(id);
    return conversation;
  }

  function ensureConversation(initialMessage) {
    if (activeConversation) {
      return activeConversation;
    }

    const id = uuidv4();
    const sessionId = uuidv4();
    const conversation = {
      id,
      sessionId,
      title: initialMessage ? initialMessage.slice(0, 48) : 'New conversation',
      createdAt: Date.now(),
      updatedAt: Date.now(),
      messages: [],
    };
    setActiveId(id);
    return conversation;
  }

  function deleteConversation(id) {
    upsertConversation((current) => current.filter((conversation) => conversation.id !== id));
    if (id === activeId) {
      const remaining = conversations.filter((conversation) => conversation.id !== id);
      setActiveId(remaining[0]?.id || null);
    }
  }

  async function sendMessage(text = draft) {
    const content = text.trim();
    if (!content || isLoading) return;

    setError('');
    setDraft('');

    const conversation = ensureConversation(content);

    const userMessage = {
      id: uuidv4(),
      role: 'user',
      content,
      timestamp: Date.now(),
    };

    upsertConversation((current) => {
      const existingConversation = current.some((item) => item.id === conversation.id);

      if (!existingConversation) {
        return [{ ...conversation, messages: [userMessage] }, ...current];
      }

      return current.map((item) =>
        item.id === conversation.id
          ? {
              ...item,
              title: item.title === 'New conversation' ? content.slice(0, 48) : item.title,
              updatedAt: Date.now(),
              messages: [...item.messages, userMessage],
            }
          : item,
      );
    });

    setIsLoading(true);
    try {
      const data = await apiRequest('/api/chat', {
        method: 'POST',
        body: JSON.stringify({ message: content, session_id: conversation.sessionId }),
      });

      const assistantMessage = {
        id: uuidv4(),
        role: 'assistant',
        content: data.response,
        sources: data.sources || [],
        timestamp: Date.now(),
      };

      upsertConversation((current) =>
        current.map((item) =>
          item.id === conversation.id
            ? { ...item, updatedAt: Date.now(), messages: [...item.messages, assistantMessage] }
            : item,
        ),
      );
    } catch (requestError) {
      setError(requestError.message || 'Unable to contact the backend.');
    } finally {
      setIsLoading(false);
    }
  }

  async function copyText(text) {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      setError('Could not copy to clipboard.');
    }
  }

  // ── Admin view rendering ──────────────────────────────────────────
  if (currentView === 'admin-login') {
    return (
      <AdminLogin
        onLogin={() => setCurrentView('admin-panel')}
        onCancel={() => setCurrentView('chat')}
      />
    );
  }

  if (currentView === 'admin-panel') {
    return <AdminLayout onLogout={() => setCurrentView('chat')} />;
  }

  // ── Chat view (default) ───────────────────────────────────────────
  return (
    <div className="appShell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brandIcon">
            <Stethoscope size={22} />
          </div>
          <div>
            <h1>MedTriage AI</h1>
            <p>Source-backed health guidance</p>
          </div>
        </div>

        <button className="newChatButton" type="button" onClick={() => startConversation()}>
          <Plus size={16} />
          New chat
        </button>

        <nav className="conversationList" aria-label="Conversations">
          {conversations.map((conversation) => (
            <button
              className={`conversationItem ${conversation.id === activeId ? 'active' : ''}`}
              type="button"
              key={conversation.id}
              onClick={() => setActiveId(conversation.id)}
            >
              <span>{conversation.title}</span>
              <Trash2
                size={14}
                onClick={(event) => {
                  event.stopPropagation();
                  deleteConversation(conversation.id);
                }}
              />
            </button>
          ))}
        </nav>
      </aside>

      <main className="chatPane">
        <header className="topbar">
          <div>
            <h2>Medical Information Assistant</h2>
            <p>Educational guidance only. Seek professional care for medical concerns.</p>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
            <div className={`healthBadge health-${health.status}`} title={health.detail}>
              {health.status === 'checking' ? <Loader2 size={15} className="spin" /> : <Activity size={15} />}
              <span>{health.status === 'ok' ? 'Backend online' : health.status}</span>
            </div>
            <button
              className="adminTrigger"
              type="button"
              onClick={() => setCurrentView('admin-login')}
              title="Admin Panel"
              aria-label="Open admin panel"
            >
              <Shield size={16} />
            </button>
          </div>
        </header>

        <section className="messages" aria-live="polite">
          {messages.length === 0 && !isLoading ? <EmptyState onPrompt={sendMessage} /> : null}
          {messages.map((message) => (
            <Message key={message.id} message={message} onCopy={copyText} />
          ))}
          {isLoading && (
            <div className="loadingRow">
              <Loader2 size={18} className="spin" />
              <span>Agent is checking sources...</span>
            </div>
          )}
          <div ref={bottomRef} />
        </section>

        {error && (
          <div className="errorBox" role="alert">
            <AlertCircle size={17} />
            <span>{error}</span>
          </div>
        )}

        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault();
            sendMessage();
          }}
        >
          <textarea
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            placeholder="Ask about symptoms, prevention, public health guidance..."
            rows={1}
            onKeyDown={(event) => {
              if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
              }
            }}
          />
          <button type="submit" disabled={!draft.trim() || isLoading} aria-label="Send message">
            <Send size={17} />
          </button>
        </form>
      </main>
    </div>
  );
}
