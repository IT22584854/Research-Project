export default function TypingIndicator() {
    return (
        <div className="message-group">
            <div className="message-row">
                <div className="message-avatar assistant-avatar">🩺</div>
                <div className="message-body">
                    <div className="message-sender">Nenagov</div>
                    <div className="typing-indicator">
                        <span className="typing-dot" />
                        <span className="typing-dot" />
                        <span className="typing-dot" />
                    </div>
                </div>
            </div>
        </div>
    );
}
