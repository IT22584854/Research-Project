import { useRef, useState, useEffect } from 'react';
import PropTypes from 'prop-types';
import { Send } from 'lucide-react';

const MAX_CHARS = 2000;

export default function ChatInput({ onSend, disabled }) {
    const [text, setText] = useState('');
    const textareaRef = useRef(null);

    // auto-grow textarea
    useEffect(() => {
        const el = textareaRef.current;
        if (!el) return;
        el.style.height = 'auto';
        el.style.height = Math.min(el.scrollHeight, 200) + 'px';
    }, [text]);

    const handleSend = () => {
        const trimmed = text.trim();
        if (!trimmed || disabled) return;
        onSend(trimmed);
        setText('');
        if (textareaRef.current) {
            textareaRef.current.style.height = 'auto';
        }
    };

    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    };

    const handleChange = (e) => {
        const newText = e.target.value;
        if (newText.length <= MAX_CHARS) {
            setText(newText);
        }
    };

    const charCount = text.length;
    const isNearLimit = charCount > MAX_CHARS * 0.8;
    const isAtLimit = charCount >= MAX_CHARS;

    return (
        <div className="chat-input-wrapper">
            <div className="chat-input-box">
                <textarea
                    ref={textareaRef}
                    className="chat-textarea"
                    rows={1}
                    placeholder="Ask about symptoms, medications, conditions…"
                    value={text}
                    onChange={handleChange}
                    onKeyDown={handleKeyDown}
                    disabled={disabled}
                    aria-label="Message input"
                />
                <div className="chat-input-toolbar">
                    {/* Character Counter (only show when typing or near limit) */}
                    {(charCount > 0 || isNearLimit) && (
                        <span
                            className={`chat-char-count ${
                                isAtLimit
                                    ? 'chat-char-count-limit'
                                    : isNearLimit
                                    ? 'chat-char-count-warning'
                                    : ''
                            }`}
                        >
                            {charCount}/{MAX_CHARS}
                        </span>
                    )}

                    {/* Send button */}
                    <button
                        className="send-btn"
                        onClick={handleSend}
                        disabled={disabled || !text.trim()}
                        title="Send message"
                        aria-label="Send message"
                    >
                        <Send size={16} />
                    </button>
                </div>
            </div>
            <p className="input-hint">
                Press <strong>Enter</strong> to send · <strong>Shift+Enter</strong> for new line
                <span className="input-hint-separator">•</span>
                For emergencies call <strong>999 / 911</strong>
            </p>
        </div>
    );
}

ChatInput.propTypes = {
    onSend: PropTypes.func.isRequired,
    disabled: PropTypes.bool,
};
