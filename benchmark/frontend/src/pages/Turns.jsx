import { useEffect, useState } from "react";
import {
  getTurns,
  evalHighRiskTurn,
  evalGroundTruthTurn,
} from "../api";

export default function Turns() {
  const [turns, setTurns] = useState([]);
  const [results, setResults] = useState({});
  const [loading, setLoading] = useState({});
  const [errors, setErrors] = useState({});

  const load = async () => {
    try {
      const res = await getTurns();
      setTurns(res.data.turns);
    } catch (err) {
      console.error("Failed to load turns:", err);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const evaluate = async (id, type) => {
    try {
      setLoading((prev) => ({ ...prev, [id]: true }));
      setErrors((prev) => ({ ...prev, [id]: null }));

      let res;
      if (type === "high") {
        res = await evalHighRiskTurn(id);
      } else {
        res = await evalGroundTruthTurn(id);
      }

      setResults((prev) => ({
        ...prev,
        [id]: res.data,
      }));
    } catch (err) {
      setErrors((prev) => ({
        ...prev,
        [id]: err.response?.data || err.message,
      }));
    } finally {
      setLoading((prev) => ({ ...prev, [id]: false }));
    }
  };

  const styles = {
    container: {
      maxWidth: '900px',
      margin: '0 auto',
    },
    header: {
      textAlign: 'center',
      marginBottom: '30px',
      paddingTop: '10px',
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
    turnsContainer: {
      display: 'flex',
      flexDirection: 'column',
      gap: '20px',
    },
    turnCard: {
      backgroundColor: '#ffffff',
      borderRadius: '16px',
      boxShadow: '0 1px 3px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04)',
      overflow: 'hidden',
      transition: 'transform 0.2s ease, box-shadow 0.2s ease',
    },
    turnHeader: {
      padding: '16px 20px',
      backgroundColor: '#f8f9fa',
      borderBottom: '1px solid #efefef',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
    },
    turnNumber: {
      fontSize: '12px',
      fontWeight: '600',
      color: '#0095f6',
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
    },
    turnId: {
      fontSize: '12px',
      color: '#8e8e8e',
      fontFamily: 'monospace',
    },
    contentSection: {
      padding: '20px',
    },
    messageBlock: {
      marginBottom: '20px',
    },
    messageLabel: {
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
      marginBottom: '10px',
    },
    labelIcon: {
      fontSize: '16px',
    },
    labelText: {
      fontSize: '12px',
      fontWeight: '600',
      color: '#8e8e8e',
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
    },
    questionText: {
      fontSize: '15px',
      color: '#262626',
      lineHeight: '1.5',
      margin: '0',
      fontWeight: '500',
      backgroundColor: '#f8f9fa',
      padding: '12px',
      borderRadius: '12px',
    },
    answerText: {
      fontSize: '14px',
      color: '#4a4a4a',
      lineHeight: '1.6',
      margin: '0',
      backgroundColor: '#f8f9fa',
      padding: '12px',
      borderRadius: '12px',
    },
    buttonGroup: {
      display: 'flex',
      gap: '12px',
      marginTop: '20px',
      marginBottom: '20px',
    },
    riskButton: {
      flex: '1',
      padding: '10px 20px',
      backgroundColor: '#363636',
      color: '#ffffff',
      border: 'none',
      borderRadius: '10px',
      fontSize: '13px',
      fontWeight: '600',
      cursor: 'pointer',
      transition: 'all 0.2s ease',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '8px',
      fontFamily: 'inherit',
    },
    riskButtonDisabled: {
      backgroundColor: '#dbdbdb',
      cursor: 'not-allowed',
    },
    groundButton: {
      flex: '1',
      padding: '10px 20px',
      backgroundColor: '#0095f6',
      color: '#ffffff',
      border: 'none',
      borderRadius: '10px',
      fontSize: '13px',
      fontWeight: '600',
      cursor: 'pointer',
      transition: 'all 0.2s ease',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '8px',
      fontFamily: 'inherit',
    },
    groundButtonDisabled: {
      backgroundColor: '#b3e0ff',
      cursor: 'not-allowed',
    },
    errorBox: {
      marginTop: '16px',
      padding: '16px',
      backgroundColor: '#fee',
      border: '1px solid #fcc',
      borderRadius: '12px',
      color: '#c33',
    },
    errorTitle: {
      fontSize: '14px',
      fontWeight: '600',
      marginBottom: '8px',
    },
    errorContent: {
      fontSize: '13px',
      fontFamily: 'monospace',
      margin: '0',
      whiteSpace: 'pre-wrap',
      wordWrap: 'break-word',
    },
    resultBox: {
      marginTop: '20px',
      padding: '20px',
      backgroundColor: '#f8f9fa',
      borderRadius: '12px',
      border: '1px solid #efefef',
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
    metricsGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
      gap: '16px',
      marginBottom: '16px',
    },
    metricCard: {
      backgroundColor: '#ffffff',
      padding: '12px',
      borderRadius: '10px',
      border: '1px solid #efefef',
    },
    metricLabel: {
      fontSize: '11px',
      fontWeight: '600',
      color: '#8e8e8e',
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
      marginBottom: '6px',
    },
    metricValue: {
      fontSize: '20px',
      fontWeight: '600',
      color: '#262626',
    },
    details: {
      marginTop: '12px',
    },
    summary: {
      fontSize: '13px',
      fontWeight: '500',
      color: '#0095f6',
      cursor: 'pointer',
    },
    pre: {
      marginTop: '12px',
      fontSize: '12px',
      fontFamily: 'monospace',
      whiteSpace: 'pre-wrap',
      wordWrap: 'break-word',
      backgroundColor: '#ffffff',
      padding: '12px',
      borderRadius: '8px',
      overflow: 'auto',
    },
    loadingSpinner: {
      display: 'inline-block',
      width: '14px',
      height: '14px',
      border: '2px solid rgba(255,255,255,0.3)',
      borderTopColor: '#ffffff',
      borderRadius: '50%',
      animation: 'spin 0.6s linear infinite',
    },
  };

  // Button hover handlers
  const handleRiskButtonHover = (e, isHovering, disabled) => {
    if (disabled) return;
    if (isHovering) {
      e.currentTarget.style.backgroundColor = '#262626';
      e.currentTarget.style.transform = 'translateY(-1px)';
    } else {
      e.currentTarget.style.backgroundColor = '#363636';
      e.currentTarget.style.transform = 'translateY(0)';
    }
  };

  const handleGroundButtonHover = (e, isHovering, disabled) => {
    if (disabled) return;
    if (isHovering) {
      e.currentTarget.style.backgroundColor = '#005c8a';
      e.currentTarget.style.transform = 'translateY(-1px)';
    } else {
      e.currentTarget.style.backgroundColor = '#0095f6';
      e.currentTarget.style.transform = 'translateY(0)';
    }
  };

  const handleCardHover = (e, isHovering) => {
    if (isHovering) {
      e.currentTarget.style.transform = 'translateY(-2px)';
      e.currentTarget.style.boxShadow = '0 8px 24px rgba(0,0,0,0.1)';
    } else {
      e.currentTarget.style.transform = 'translateY(0)';
      e.currentTarget.style.boxShadow = '0 1px 3px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04)';
    }
  };

  // Add spinner animation
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

  const getRatingColor = (rating) => {
    if (!rating) return '#8e8e8e';
    if (rating === 'High Risk') return '#dc3545';
    if (rating === 'Medium Risk') return '#ffc107';
    if (rating === 'Low Risk') return '#28a745';
    return '#0095f6';
  };

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>Agent Turns</h3>
        <div style={styles.headerUnderline}></div>
      </div>

      <div style={styles.turnsContainer}>
        {turns.map((t, index) => (
          <div
            key={t.id}
            style={styles.turnCard}
            onMouseEnter={(e) => handleCardHover(e, true)}
            onMouseLeave={(e) => handleCardHover(e, false)}
          >
            <div style={styles.turnHeader}>
              <span style={styles.turnNumber}>Turn #{index + 1}</span>
              <span style={styles.turnId}>ID: {t.id}</span>
            </div>

            <div style={styles.contentSection}>
              <div style={styles.messageBlock}>
                <div style={styles.messageLabel}>
                  <span style={styles.labelIcon}>❓</span>
                  <span style={styles.labelText}>Question</span>
                </div>
                <p style={styles.questionText}>{t.user_message}</p>
              </div>

              <div style={styles.messageBlock}>
                <div style={styles.messageLabel}>
                  <span style={styles.labelIcon}>💬</span>
                  <span style={styles.labelText}>Answer</span>
                </div>
                <p style={styles.answerText}>{t.assistant_message}</p>
              </div>

              <div style={styles.buttonGroup}>
                <button
                  onClick={() => evaluate(t.id, "high")}
                  disabled={loading[t.id]}
                  style={{
                    ...styles.riskButton,
                    ...(loading[t.id] && styles.riskButtonDisabled),
                  }}
                  onMouseEnter={(e) => handleRiskButtonHover(e, true, loading[t.id])}
                  onMouseLeave={(e) => handleRiskButtonHover(e, false, loading[t.id])}
                >
                  {loading[t.id] ? (
                    <>
                      <div style={styles.loadingSpinner}></div>
                      Evaluating...
                    </>
                  ) : (
                    <>⚠️ High Risk</>
                  )}
                </button>

                <button
                  onClick={() => evaluate(t.id, "gt")}
                  disabled={loading[t.id]}
                  style={{
                    ...styles.groundButton,
                    ...(loading[t.id] && styles.groundButtonDisabled),
                  }}
                  onMouseEnter={(e) => handleGroundButtonHover(e, true, loading[t.id])}
                  onMouseLeave={(e) => handleGroundButtonHover(e, false, loading[t.id])}
                >
                  {loading[t.id] ? (
                    <>
                      <div style={styles.loadingSpinner}></div>
                      Evaluating...
                    </>
                  ) : (
                    <>✓ Ground Truth</>
                  )}
                </button>
              </div>

              {/* Error Display */}
              {errors[t.id] && (
                <div style={styles.errorBox}>
                  <div style={styles.errorTitle}>❌ Evaluation Error</div>
                  <pre style={styles.errorContent}>
                    {typeof errors[t.id] === 'object' 
                      ? JSON.stringify(errors[t.id], null, 2)
                      : errors[t.id]}
                  </pre>
                </div>
              )}

              {/* Result Display */}
              {results[t.id] && (
                <div style={styles.resultBox}>
                  <div style={styles.resultTitle}>
                    <span>📊</span>
                    <span>Evaluation Results</span>
                  </div>

                  <div style={styles.metricsGrid}>
                    {results[t.id].final_score !== undefined && (
                      <div style={styles.metricCard}>
                        <div style={styles.metricLabel}>Final Score</div>
                        <div style={styles.metricValue}>
                          {(results[t.id].final_score * 100).toFixed(1)}%
                        </div>
                      </div>
                    )}
                    
                    {results[t.id].rating && (
                      <div style={styles.metricCard}>
                        <div style={styles.metricLabel}>Rating</div>
                        <div style={{
                          ...styles.metricValue,
                          color: getRatingColor(results[t.id].rating),
                          fontSize: '16px',
                        }}>
                          {results[t.id].rating}
                        </div>
                      </div>
                    )}

                    {results[t.id].confidence !== undefined && (
                      <div style={styles.metricCard}>
                        <div style={styles.metricLabel}>Confidence</div>
                        <div style={styles.metricValue}>
                          {(results[t.id].confidence * 100).toFixed(1)}%
                        </div>
                      </div>
                    )}
                  </div>

                  <details style={styles.details}>
                    <summary style={styles.summary}>View Full JSON Response</summary>
                    <pre style={styles.pre}>
                      {JSON.stringify(results[t.id], null, 2)}
                    </pre>
                  </details>
                </div>
              )}
            </div>
          </div>
        ))}

        {turns.length === 0 && (
          <div style={{ textAlign: 'center', padding: '60px 20px', color: '#8e8e8e' }}>
            <div style={{ fontSize: '48px', marginBottom: '16px' }}>📭</div>
            <p>No turns available. Please check your connection or refresh.</p>
          </div>
        )}
      </div>
    </div>
  );
}