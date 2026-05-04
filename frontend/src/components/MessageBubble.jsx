import PropTypes from 'prop-types';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import Citations from './Citations';
import { MessageActions } from './chat/MessageActions';

function stripInlineSources(content, sources) {
    if (!content || !Array.isArray(sources) || sources.length === 0) return content;

    return content
        .replace(/\n{0,2}□?Sources:\s*[\s\S]*$/i, '')
        .trim();
}

function UserMessage({ msg, showActions }) {
    return (
        <div className="message-group group">
            <div className="message-row message-row-user">
                <div className="message-body message-body-user">
                    <div className="message-sender message-sender-user">You</div>
                    <div className="user-bubble">{msg.content}</div>
                    {showActions && <MessageActions message={msg} isLastAssistantMessage={false} />}
                </div>
                <div className="message-avatar user-avatar">U</div>
            </div>
        </div>
    );
}

function AssistantMessage({ msg, isLastAssistantMessage, showActions }) {
    const renderedContent = stripInlineSources(msg.content, msg.sources);

    return (
        <div className="message-group group">
            <div className="message-row">
                <div className="message-avatar assistant-avatar">🩺</div>
                <div className="message-body">
                    <div className="message-sender">Nenagov</div>
                    <div className="message-content">
                        <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={{
                                code({ node, inline, className, children, ...props }) {
                                    const match = /language-(\w+)/.exec(className || '');
                                    return !inline && match ? (
                                        <SyntaxHighlighter
                                            style={oneDark}
                                            language={match[1]}
                                            PreTag="div"
                                            customStyle={{ borderRadius: '10px', fontSize: '13px' }}
                                            {...props}
                                        >
                                            {String(children).replace(/\n$/, '')}
                                        </SyntaxHighlighter>
                                    ) : (
                                        <code className={className} {...props}>{children}</code>
                                    );
                                },
                            }}
                        >
                            {renderedContent}
                        </ReactMarkdown>
                    </div>
                    <Citations sources={msg.sources} />
                    {showActions && <MessageActions message={msg} isLastAssistantMessage={isLastAssistantMessage} />}
                </div>
            </div>
        </div>
    );
}

export default function MessageBubble({ msg, isLastAssistantMessage = false, showActions = true }) {
    if (msg.role === 'user') return <UserMessage msg={msg} showActions={showActions} />;
    return <AssistantMessage msg={msg} isLastAssistantMessage={isLastAssistantMessage} showActions={showActions} />;
}

MessageBubble.propTypes = {
    msg: PropTypes.shape({
        id: PropTypes.string.isRequired,
        role: PropTypes.oneOf(['user', 'assistant']).isRequired,
        content: PropTypes.string.isRequired,
        sources: PropTypes.array,
    }).isRequired,
    isLastAssistantMessage: PropTypes.bool,
    showActions: PropTypes.bool,
};
