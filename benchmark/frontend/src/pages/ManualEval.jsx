import { useState } from "react";
import { manualHighRisk, manualGroundTruth } from "../api";

export default function ManualEval() {
  const [form, setForm] = useState({
    question: "",
    answer: "",
    start_timestamp: "",
    end_timestamp: "",
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    setForm({
      ...form,
      [e.target.name]: e.target.value,
    });
  };

  const run = async (mode) => {
    try {
      setLoading(true);
      setError(null);
      setResult(null);

      const payload = {
        ...form,
        start_timestamp: parseFloat(form.start_timestamp),
        end_timestamp: parseFloat(form.end_timestamp),
      };

      let res;
      if (mode === "high") {
        res = await manualHighRisk(payload);
      } else {
        res = await manualGroundTruth(payload);
      }

      setResult(res.data);
    } catch (err) {
      setError(err.response?.data || err.message);
    } finally {
      setLoading(false);
    }
  };

  const fillNow = () => {
    const now = Date.now() / 1000;
    setForm((prev) => ({
      ...prev,
      start_timestamp: now.toString(),
      end_timestamp: (now + 1).toString(),
    }));
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
    card: {
      backgroundColor: '#ffffff',
      borderRadius: '16px',
      boxShadow: '0 1px 3px rgba(0,0,0,0.08), 0 2px 4px rgba(0,0,0,0.04)',
      overflow: 'hidden',
    },
    formSection: {
      padding: '24px',
    },
    formGroup: {
      marginBottom: '20px',
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
    labelHint: {
      fontSize: '12px',
      fontWeight: '400',
      color: '#8e8e8e',
      marginLeft: '8px',
    },
    textarea: {
      width: '100%',
      padding: '12px 16px',
      fontSize: '14px',
      border: '1px solid #dbdbdb',
      borderRadius: '12px',
      backgroundColor: '#ffffff',
      fontFamily: 'inherit',
      transition: 'all 0.2s ease',
      resize: 'vertical',
      boxSizing: 'border-box',
    },
    input: {
      width: '100%',
      padding: '12px 16px',
      fontSize: '14px',
      border: '1px solid #dbdbdb',
      borderRadius: '12px',
      backgroundColor: '#ffffff',
      transition: 'all 0.2s ease',
      fontFamily: 'inherit',
      boxSizing: 'border-box',
    },
    timestampGroup: {
      display: 'grid',
      gridTemplateColumns: '1fr 1fr auto',
      gap: '12px',
      alignItems: 'center',
      marginBottom: '20px',
    },
    autoFillButton: {
      padding: '12px 20px',
      backgroundColor: '#f8f9fa',
      color: '#0095f6',
      border: '1px solid #dbdbdb',
      borderRadius: '10px',
      fontSize: '13px',
      fontWeight: '600',
      cursor: 'pointer',
      transition: 'all 0.2s ease',
      fontFamily: 'inherit',
      whiteSpace: 'nowrap',
    },
    buttonGroup: {
      display: 'flex',
      gap: '12px',
      marginTop: '20px',
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
    riskButtonDisabled: {
      backgroundColor: '#ababab',
      cursor: 'not-allowed',
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
    groundButtonDisabled: {
      backgroundColor: '#b3e0ff',
      cursor: 'not-allowed',
    },
    errorBox: {
      marginTop: '20px',
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
      padding: '24px',
      backgroundColor: '#f8f9fa',
      borderRadius: '12px',
      border: '1px solid #efefef',
    },
    resultTitle: {
      fontSize: '18px',
      fontWeight: '600',
      color: '#262626',
      marginBottom: '20px',
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
    },
    metricsGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
      gap: '16px',
      marginBottom: '20px',
    },
    metricCard: {
      backgroundColor: '#ffffff',
      padding: '16px',
      borderRadius: '12px',
      border: '1px solid #efefef',
    },
    metricLabel: {
      fontSize: '11px',
      fontWeight: '600',
      color: '#8e8e8e',
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
      marginBottom: '8px',
    },
    metricValue: {
      fontSize: '24px',
      fontWeight: '600',
      color: '#262626',
    },
    metricSubValue: {
      fontSize: '14px',
      color: '#8e8e8e',
      marginTop: '4px',
    },
    metricsSection: {
      marginBottom: '20px',
    },
    metricsTitle: {
      fontSize: '14px',
      fontWeight: '600',
      color: '#262626',
      marginBottom: '12px',
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
    },
    metricsList: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))',
      gap: '12px',
    },
    metricItem: {
      backgroundColor: '#ffffff',
      padding: '10px 12px',
      borderRadius: '10px',
      border: '1px solid #efefef',
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
    },
    metricItemLabel: {
      fontSize: '12px',
      color: '#8e8e8e',
      textTransform: 'capitalize',
    },
    metricItemValue: {
      fontSize: '14px',
      fontWeight: '600',
      color: '#262626',
    },
    details: {
      marginTop: '16px',
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
      maxHeight: '400px',
    },
    loadingSpinner: {
      display: 'inline-block',
      width: '16px',
      height: '16px',
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

  const handleAutoFillHover = (e, isHovering) => {
    if (isHovering) {
      e.currentTarget.style.backgroundColor = '#e9ecef';
      e.currentTarget.style.transform = 'translateY(-1px)';
    } else {
      e.currentTarget.style.backgroundColor = '#f8f9fa';
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

  const isFormValid = form.question.trim() && form.answer.trim();

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>Manual Evaluation</h3>
        <div style={styles.headerUnderline}></div>
      </div>

      <div style={styles.card}>
        <div style={styles.formSection}>
          <div style={styles.formGroup}>
            <label style={styles.label}>
              <span style={styles.labelIcon}>❓</span>
              Question
            </label>
            <textarea
              name="question"
              placeholder="Enter the question to evaluate..."
              value={form.question}
              onChange={handleChange}
              rows={3}
              style={styles.textarea}
              onMouseEnter={(e) => handleInputHover(e, true)}
              onMouseLeave={(e) => handleInputHover(e, false)}
              onFocus={handleInputFocus}
              onBlur={handleInputBlur}
            />
          </div>

          <div style={styles.formGroup}>
            <label style={styles.label}>
              <span style={styles.labelIcon}>💬</span>
              Answer
            </label>
            <textarea
              name="answer"
              placeholder="Enter the answer to evaluate..."
              value={form.answer}
              onChange={handleChange}
              rows={5}
              style={styles.textarea}
              onMouseEnter={(e) => handleInputHover(e, true)}
              onMouseLeave={(e) => handleInputHover(e, false)}
              onFocus={handleInputFocus}
              onBlur={handleInputBlur}
            />
          </div>

          <div style={styles.timestampGroup}>
            <input
              name="start_timestamp"
              placeholder="Start timestamp (unix)"
              value={form.start_timestamp}
              onChange={handleChange}
              style={styles.input}
              onMouseEnter={(e) => handleInputHover(e, true)}
              onMouseLeave={(e) => handleInputHover(e, false)}
              onFocus={handleInputFocus}
              onBlur={handleInputBlur}
            />
            <input
              name="end_timestamp"
              placeholder="End timestamp (unix)"
              value={form.end_timestamp}
              onChange={handleChange}
              style={styles.input}
              onMouseEnter={(e) => handleInputHover(e, true)}
              onMouseLeave={(e) => handleInputHover(e, false)}
              onFocus={handleInputFocus}
              onBlur={handleInputBlur}
            />
            <button
              onClick={fillNow}
              style={styles.autoFillButton}
              onMouseEnter={(e) => handleAutoFillHover(e, true)}
              onMouseLeave={(e) => handleAutoFillHover(e, false)}
            >
              ⚡ Auto-fill
            </button>
          </div>

          <div style={styles.buttonGroup}>
            <button
              onClick={() => run("high")}
              disabled={loading || !isFormValid}
              style={{
                ...styles.riskButton,
                ...((loading || !isFormValid) && styles.riskButtonDisabled),
              }}
              onMouseEnter={(e) => handleRiskButtonHover(e, true, loading || !isFormValid)}
              onMouseLeave={(e) => handleRiskButtonHover(e, false, loading || !isFormValid)}
            >
              {loading ? (
                <>
                  <div style={styles.loadingSpinner}></div>
                  Evaluating...
                </>
              ) : (
                <>⚠️ Evaluate High Risk</>
              )}
            </button>

            <button
              onClick={() => run("gt")}
              disabled={loading || !isFormValid}
              style={{
                ...styles.groundButton,
                ...((loading || !isFormValid) && styles.groundButtonDisabled),
              }}
              onMouseEnter={(e) => handleGroundButtonHover(e, true, loading || !isFormValid)}
              onMouseLeave={(e) => handleGroundButtonHover(e, false, loading || !isFormValid)}
            >
              {loading ? (
                <>
                  <div style={styles.loadingSpinner}></div>
                  Evaluating...
                </>
              ) : (
                <>✓ Evaluate Ground Truth</>
              )}
            </button>
          </div>
        </div>

        {/* Error Display */}
        {error && (
          <div style={styles.errorBox}>
            <div style={styles.errorTitle}>❌ Evaluation Error</div>
            <pre style={styles.errorContent}>
              {typeof error === 'object' ? JSON.stringify(error, null, 2) : error}
            </pre>
          </div>
        )}

        {/* Result Display */}
        {result && (
          <div style={styles.resultBox}>
            <div style={styles.resultTitle}>
              <span>📊</span>
              <span>Evaluation Results</span>
            </div>

            <div style={styles.metricsGrid}>
              {result.final_score !== undefined && (
                <div style={styles.metricCard}>
                  <div style={styles.metricLabel}>Final Score</div>
                  <div style={styles.metricValue}>
                    {(result.final_score * 100).toFixed(1)}%
                  </div>
                  <div style={styles.metricSubValue}>out of 100%</div>
                </div>
              )}

              {result.rating && (
                <div style={styles.metricCard}>
                  <div style={styles.metricLabel}>Rating</div>
                  <div style={{
                    ...styles.metricValue,
                    color: getRatingColor(result.rating),
                    fontSize: '20px',
                  }}>
                    {result.rating}
                  </div>
                </div>
              )}

              {result.confidence !== undefined && (
                <div style={styles.metricCard}>
                  <div style={styles.metricLabel}>Confidence</div>
                  <div style={styles.metricValue}>
                    {(result.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              )}
            </div>

            {/* Detailed Metrics */}
            {result.metrics && Object.keys(result.metrics).length > 0 && (
              <div style={styles.metricsSection}>
                <div style={styles.metricsTitle}>
                  <span>📈</span>
                  <span>Detailed Metrics</span>
                </div>
                <div style={styles.metricsList}>
                  {Object.entries(result.metrics).map(([key, value]) => (
                    <div key={key} style={styles.metricItem}>
                      <span style={styles.metricItemLabel}>
                        {key.replace(/_/g, ' ')}
                      </span>
                      <span style={styles.metricItemValue}>
                        {typeof value === 'number' ? value.toFixed(3) : String(value)}
                      </span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <details style={styles.details}>
              <summary style={styles.summary}>View Full JSON Response</summary>
              <pre style={styles.pre}>
                {JSON.stringify(result, null, 2)}
              </pre>
            </details>
          </div>
        )}
      </div>
    </div>
  );
}