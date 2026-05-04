import { useRef, useEffect } from 'react';
import PropTypes from 'prop-types';
import MessageBubble from './MessageBubble';
import TypingIndicator from './TypingIndicator';

const SUGGESTIONS = [
    { icon: '🤒', text: 'What are common symptoms of the flu vs COVID-19?' },
    { icon: '💊', text: 'What is the difference between ibuprofen and paracetamol?' },
    { icon: '🩺', text: 'When should I see a doctor for chest pain?' },
    { icon: '🧠', text: 'What are early warning signs of diabetes?' },
];

function WelcomeState({ onSuggestion }) {
    return (
        <div className="welcome-state">
            <div className="welcome-icon">🩺</div>
            <h1 className="welcome-title">Nenagov</h1>
            <p className="welcome-subtitle">
                Ask me anything about symptoms, conditions, medications, or when to seek medical care.
                I'll provide evidence-based information to guide your health decisions.
            </p>
            <div className="suggestion-grid">
                {SUGGESTIONS.map((s, i) => (
                    <button key={i} className="suggestion-card" onClick={() => onSuggestion(s.text)}>
                        <span className="suggestion-card-icon">{s.icon}</span>
                        <span className="suggestion-card-text">{s.text}</span>
                    </button>
                ))}
            </div>
        </div>
    );
}

export default function ChatWindow({ messages, isLoading, error, onSuggestion }) {
    const bottomRef = useRef(null);

    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages, isLoading]);

    const isEmpty = messages.length === 0;

    // Find the last assistant message index
    const lastAssistantMessageIndex = messages.reduceRight((acc, msg, index) => {
        if (acc === -1 && msg.role === 'assistant') {
            return index;
        }
        return acc;
    }, -1);

    return (
        <div className="chat-window">
            {isEmpty && !isLoading ? (
                <WelcomeState onSuggestion={onSuggestion} />
            ) : (
                <>
                    {messages.map((msg, index) => (
                        <MessageBubble 
                            key={msg.id} 
                            msg={msg} 
                            isLastAssistantMessage={index === lastAssistantMessageIndex}
                        />
                    ))}
                    {isLoading && <TypingIndicator />}
                    {error && (
                        <div className="error-message" role="alert">
                            ⚠️ {error}
                        </div>
                    )}
                </>
            )}
            <div ref={bottomRef} />
        </div>
    );
}

ChatWindow.propTypes = {
    messages: PropTypes.arrayOf(PropTypes.shape({
        id: PropTypes.string.isRequired,
        role: PropTypes.string.isRequired,
        content: PropTypes.string.isRequired,
        sources: PropTypes.array,
    })).isRequired,
    isLoading: PropTypes.bool,
    error: PropTypes.string,
    onSuggestion: PropTypes.func,
};
