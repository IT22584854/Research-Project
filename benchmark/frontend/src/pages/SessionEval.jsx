import { useState } from "react";
import { evalHighRiskSession, evalGroundTruthSession } from "../api";

export default function SessionEval() {
  const [session, setSession] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const run = async (type) => {
    setLoading(true);
    try {
      const res =
        type === "high"
          ? await evalHighRiskSession(session)
          : await evalGroundTruthSession(session);
      setResult(res.data);
    } catch (error) {
      console.error("Evaluation error:", error);
      setResult({ error: "Failed to evaluate session" });
    } finally {
      setLoading(false);
    }
  };

  const styles = {
    container: {
      maxWidth: '800px',
      margin: '0 auto',
      padding: '20px',
      backgroundColor: '#fafafa',
      minHeight: '100vh',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif',
    },
    
    header: {
      textAlign: 'center',
      marginBottom: '30px',
      paddingTop: '20px',
    },
    
    title: {
      fontSize: '24px',
      fontWeight: '600',
      color: '#262626',
      margin: '0 0 10px 0',
      letterSpacing: '-0.5px',
    },
    
    headerUnderline: {
      width: '60px',
      height: '3px',
      background: 'linear-gradient(90deg, #0095f6, #00c3ff)',
      margin: '0 auto',
      borderRadius: '2px',
    },
    
    card: {
      backgroundColor: '#ffffff',
      borderRadius: '16px',
      boxShadow: '0 1px 3px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04)',
      overflow: 'hidden',
      marginBottom: '24px',
    },
    
    inputSection: {
      padding: '24px',
    },
    
    label: {
      display: 'block',
      fontSize: '14px',
      fontWeight: '600',
      color: '#262626',
      marginBottom: '8px',
    },
    
    labelIcon: {
      marginRight: '8px',
    },
    
    inputWrapper: {
      marginBottom: '24px',
    },
    
    input: {
      width: '100%',
      padding: '12px 16px',
      fontSize: '15px',
      border: '1px solid #dbdbdb',
      borderRadius: '12px',
      backgroundColor: '#ffffff',
      transition: 'all 0.2s ease',
      fontFamily: 'inherit',
      boxSizing: 'border-box',
    },
    
    buttonGroup: {
      display: 'flex',
      gap: '12px',
    },
    
    riskButton: {
      flex: '1',
      padding: '12px 20px',
      backgroundColor: '#363636',
      color: '#ffffff',
      border: 'none',
      borderRadius: '10px',
      fontSize: '14px',
      fontWeight: '600',
      cursor: 'pointer',
      transition: 'all 0.2s ease',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '8px',
      fontFamily: 'inherit',
    },
    
    groundButton: {
      flex: '1',
      padding: '12px 20px',
      backgroundColor: '#0095f6',
      color: '#ffffff',
      border: 'none',
      borderRadius: '10px',
      fontSize: '14px',
      fontWeight: '600',
      cursor: 'pointer',
      transition: 'all 0.2s ease',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '8px',
      fontFamily: 'inherit',
    },
    
    resultSection: {
      padding: '24px',
      borderTop: '1px solid #efefef',
    },
    
    resultTitle: {
      fontSize: '16px',
      fontWeight: '600',
      color: '#262626',
      marginBottom: '16px',
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
    },
    
    resultContent: {
      backgroundColor: '#f8f9fa',
      borderRadius: '12px',
      padding: '16px',
      overflow: 'auto',
      maxHeight: '400px',
    },
    
    pre: {
      margin: 0,
      fontSize: '13px',
      color: '#262626',
      fontFamily: 'monospace',
      whiteSpace: 'pre-wrap',
      wordWrap: 'break-word',
    },
    
    emptyState: {
      textAlign: 'center',
      padding: '40px 24px',
      color: '#8e8e8e',
    },
    
    loadingOverlay: {
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '12px',
      padding: '16px',
      backgroundColor: '#f8f9fa',
      borderRadius: '12px',
      color: '#0095f6',
    },
    
    spinner: {
      width: '20px',
      height: '20px',
      border: '2px solid #efefef',
      borderTop: '2px solid #0095f6',
      borderRadius: '50%',
      animation: 'spin 0.8s linear infinite',
    },
  };

  // Button hover handlers
  const handleRiskButtonHover = (e, isHovering) => {
    if (isHovering) {
      e.currentTarget.style.backgroundColor = '#262626';
      e.currentTarget.style.transform = 'translateY(-1px)';
    } else {
      e.currentTarget.style.backgroundColor = '#363636';
      e.currentTarget.style.transform = 'translateY(0)';
    }
  };

  const handleGroundButtonHover = (e, isHovering) => {
    if (isHovering) {
      e.currentTarget.style.backgroundColor = '#005c8a';
      e.currentTarget.style.transform = 'translateY(-1px)';
    } else {
      e.currentTarget.style.backgroundColor = '#0095f6';
      e.currentTarget.style.transform = 'translateY(0)';
    }
  };

  const handleInputHover = (e, isHovering) => {
    if (isHovering) {
      e.currentTarget.style.borderColor = '#0095f6';
      e.currentTarget.style.boxShadow = '0 0 0 3px rgba(0,149,246,0.1)';
    } else {
      e.currentTarget.style.borderColor = '#dbdbdb';
      e.currentTarget.style.boxShadow = 'none';
    }
  };

  const handleInputFocus = (e) => {
    e.currentTarget.style.borderColor = '#0095f6';
    e.currentTarget.style.boxShadow = '0 0 0 3px rgba(0,149,246,0.1)';
    e.currentTarget.style.outline = 'none';
  };

  const handleInputBlur = (e) => {
    e.currentTarget.style.borderColor = '#dbdbdb';
    e.currentTarget.style.boxShadow = 'none';
  };

  // Add keyframes animation for spinner
  if (!document.querySelector('#spinner-styles')) {
    const style = document.createElement('style');
    style.id = 'spinner-styles';
    style.textContent = `
      @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
      }
    `;
    document.head.appendChild(style);
  }

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>Session Evaluation</h3>
        <div style={styles.headerUnderline}></div>
      </div>

      <div style={styles.card}>
        <div style={styles.inputSection}>
          <div style={styles.inputWrapper}>
            <label style={styles.label}>
              <span style={styles.labelIcon}>🔑</span>
              Session ID
            </label>
            <input
              style={styles.input}
              placeholder="Enter session ID (e.g., session_123)"
              value={session}
              onChange={(e) => setSession(e.target.value)}
              onMouseEnter={(e) => handleInputHover(e, true)}
              onMouseLeave={(e) => handleInputHover(e, false)}
              onFocus={handleInputFocus}
              onBlur={handleInputBlur}
            />
          </div>

          <div style={styles.buttonGroup}>
            <button
              onClick={() => run("high")}
              style={styles.riskButton}
              onMouseEnter={(e) => handleRiskButtonHover(e, true)}
              onMouseLeave={(e) => handleRiskButtonHover(e, false)}
              disabled={!session.trim() || loading}
            >
              ⚠️ High Risk Session
            </button>

            <button
              onClick={() => run("gt")}
              style={styles.groundButton}
              onMouseEnter={(e) => handleGroundButtonHover(e, true)}
              onMouseLeave={(e) => handleGroundButtonHover(e, false)}
              disabled={!session.trim() || loading}
            >
              ✓ Ground Truth Session
            </button>
          </div>
        </div>

        <div style={styles.resultSection}>
          <div style={styles.resultTitle}>
            <span>📊</span>
            <span>Evaluation Result</span>
          </div>
          
          {loading ? (
            <div style={styles.loadingOverlay}>
              <div style={styles.spinner}></div>
              <span>Evaluating session...</span>
            </div>
          ) : result ? (
            <div style={styles.resultContent}>
              <pre style={styles.pre}>
                {JSON.stringify(result, null, 2)}
              </pre>
            </div>
          ) : (
            <div style={styles.emptyState}>
              <span>💡</span>
              <p style={{ margin: '8px 0 0 0', fontSize: '14px' }}>
                Enter a session ID and click a button to see evaluation results
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}