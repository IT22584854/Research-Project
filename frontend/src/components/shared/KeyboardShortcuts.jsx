import { Keyboard } from 'lucide-react';

/**
 * Keyboard shortcuts reference component
 */
export function KeyboardShortcuts() {
  const shortcuts = [
    {
      category: 'General',
      items: [
        { keys: ['Ctrl', 'N'], description: 'Start new conversation' },
        { keys: ['Ctrl', ','], description: 'Open settings' },
        { keys: ['Esc'], description: 'Close modal/dialog' },
      ],
    },
    {
      category: 'Chat',
      items: [
        { keys: ['Enter'], description: 'Send message' },
        { keys: ['Shift', 'Enter'], description: 'New line in message' },
        { keys: ['Ctrl', 'C'], description: 'Copy selected message' },
      ],
    },
    {
      category: 'Editing',
      items: [
        { keys: ['Ctrl', 'Enter'], description: 'Save edited message' },
        { keys: ['Esc'], description: 'Cancel editing' },
      ],
    },
    {
      category: 'Navigation',
      items: [
        { keys: ['Tab'], description: 'Move focus forward' },
        { keys: ['Shift', 'Tab'], description: 'Move focus backward' },
      ],
    },
  ];

  return (
    <div className="keyboard-shortcuts">
      <div className="keyboard-shortcuts-header">
        <Keyboard size={18} />
        <h3>Keyboard Shortcuts</h3>
      </div>

      {shortcuts.map((section) => (
        <div key={section.category} className="keyboard-shortcuts-section">
          <h4 className="keyboard-shortcuts-title">
            {section.category}
          </h4>
          <div className="keyboard-shortcuts-list">
            {section.items.map((shortcut, idx) => (
              <div
                key={idx}
                className="keyboard-shortcut-row"
              >
                <span className="keyboard-shortcut-description">
                  {shortcut.description}
                </span>
                <div className="keyboard-shortcut-keys">
                  {shortcut.keys.map((key, keyIdx) => (
                    <span key={keyIdx} className="keyboard-shortcut-key-group">
                      <kbd className="keyboard-shortcut-key">
                        {key}
                      </kbd>
                      {keyIdx < shortcut.keys.length - 1 && (
                        <span className="keyboard-shortcut-plus">+</span>
                      )}
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}

      <div className="keyboard-shortcuts-footer">
        <p>💡 Tip: Some shortcuts may vary by browser or operating system.</p>
      </div>
    </div>
  );
}
