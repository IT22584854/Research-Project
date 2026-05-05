import { useState } from "react";
import { batchHighRisk, batchGroundTruth } from "../api";

export default function BatchEval() {
  const [limit, setLimit] = useState(20);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [activeType, setActiveType] = useState(null);

  const run = async (type) => {
    setLoading(true);
    setActiveType(type);
    setResult(null);
    
    try {
      const res =
        type === "high"
          ? await batchHighRisk(limit)
          : await batchGroundTruth(limit);
      setResult(res.data);
    } catch (error) {
      console.error("Batch evaluation error:", error);
      setResult({ 
        error: "Failed to run batch evaluation. Please try again.",
        details: error.message 
      });
    } finally {
      setLoading(false);
      setActiveType(null);
    }
  };

  const styles = {
    container: {
      maxWidth: '900px',
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
    },
    
    inputSection: {
      padding: '24px',
      borderBottom: '1px solid #efefef',
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
    
    inputWrapper: {
      marginBottom: '24px',
    },
    
    inputContainer: {
      display: 'flex',
      alignItems: 'center',
      gap: '12px',
      flexWrap: 'wrap',
    },
    
    input: {
      flex: '1',
      padding: '12px 16px',
      fontSize: '15px',
      border: '1px solid #dbdbdb',
      borderRadius: '12px',
      backgroundColor: '#ffffff',
      transition: 'all 0.2s ease',
      fontFamily: 'inherit',
      minWidth: '150px',
    },
    
    limitDisplay: {
      padding: '8px 16px',
      backgroundColor: '#f8f9fa',
      borderRadius: '10px',
      fontSize: '14px',
      color: '#0095f6',
      fontWeight: '500',
    },
    
    buttonGroup: {
      display: 'flex',
      gap: '12px',
      flexWrap: 'wrap',
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
      minWidth: '200px',
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
      minWidth: '200px',
    },
    
    resultSection: {
      padding: '24px',
    },
    
    resultHeader: {
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      marginBottom: '16px',
      flexWrap: 'wrap',
      gap: '12px',
    },
    
    resultTitle: {
      fontSize: '16px',
      fontWeight: '600',
      color: '#262626',
      display: 'flex',
      alignItems: 'center',
      gap: '8px',
    },
    
    resultBadge: {
      padding: '4px 12px',
      borderRadius: '20px',
      fontSize: '12px',
      fontWeight: '500',
      backgroundColor: '#efefef',
      color: '#262626',
    },
    
    resultContent: {
      backgroundColor: '#f8f9fa',
      borderRadius: '12px',
      padding: '16px',
      overflow: 'auto',
      maxHeight: '500px',
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
      padding: '60px 24px',
      color: '#8e8e8e',
    },
    
    loadingOverlay: {
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      gap: '16px',
      padding: '40px',
      backgroundColor: '#f8f9fa',
      borderRadius: '12px',
      color: '#0095f6',
    },
    
    spinner: {
      width: '40px',
      height: '40px',
      border: '3px solid #efefef',
      borderTop: '3px solid #0095f6',
      borderRadius: '50%',
      animation: 'spin 0.8s linear infinite',
    },
    
    loadingText: {
      fontSize: '14px',
      fontWeight: '500',
    },
    
    loadingSubtext: {
      fontSize: '12px',
      color: '#8e8e8e',
      marginTop: '4px',
    },
    
    statsGrid: {
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
      gap: '16px',
      marginBottom: '20px',
    },
    
    statCard: {
      backgroundColor: '#ffffff',
      borderRadius: '12px',
      padding: '16px',
      border: '1px solid #efefef',
    },
    
    statLabel: {
      fontSize: '12px',
      color: '#8e8e8e',
      marginBottom: '8px',
      textTransform: 'uppercase',
      letterSpacing: '0.5px',
    },
    
    statValue: {
      fontSize: '24px',
      fontWeight: '600',
      color: '#262626',
    },
    
    errorState: {
      backgroundColor: '#fee',
      border: '1px solid #fcc',
      color: '#c33',
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

  const handleLimitChange = (e) => {
    let value = parseInt(e.target.value);
    if (isNaN(value)) value = 1;
    if (value < 1) value = 1;
    if (value > 1000) value = 1000;
    setLimit(value);
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

  // Extract stats from result if available
  const getStats = () => {
    if (!result || result.error) return null;
    
    // Try to extract common stats from the result
    const stats = {};
    if (result.total_turns) stats.total = result.total_turns;
    if (result.processed) stats.processed = result.processed;
    if (result.success) stats.success = result.success;
    if (result.failed) stats.failed = result.failed;
    if (result.duration) stats.duration = result.duration;
    
    return Object.keys(stats).length > 0 ? stats : null;
  };

  const stats = getStats();

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <h3 style={styles.title}>Batch Evaluation</h3>
        <div style={styles.headerUnderline}></div>
      </div>

      <div style={styles.card}>
        <div style={styles.inputSection}>
          <div style={styles.inputWrapper}>
            <label style={styles.label}>
              <span style={styles.labelIcon}>📊</span>
              Batch Size
              <span style={styles.labelHint}>(1-1000 turns)</span>
            </label>
            <div style={styles.inputContainer}>
              <input
                type="number"
                style={styles.input}
                value={limit}
                onChange={handleLimitChange}
                onMouseEnter={(e) => handleInputHover(e, true)}
                onMouseLeave={(e) => handleInputHover(e, false)}
                onFocus={handleInputFocus}
                onBlur={handleInputBlur}
                min="1"
                max="1000"
                step="1"
              />
              <div style={styles.limitDisplay}>
                Will process up to {limit} turn{limit !== 1 ? 's' : ''}
              </div>
            </div>
          </div>

          <div style={styles.buttonGroup}>
            <button
              onClick={() => run("high")}
              style={styles.riskButton}
              onMouseEnter={(e) => handleRiskButtonHover(e, true)}
              onMouseLeave={(e) => handleRiskButtonHover(e, false)}
              disabled={loading}
            >
              {loading && activeType === "high" ? (
                <>⏳ Processing...</>
              ) : (
                <>⚠️ Run High Risk Batch</>
              )}
            </button>

            <button
              onClick={() => run("gt")}
              style={styles.groundButton}
              onMouseEnter={(e) => handleGroundButtonHover(e, true)}
              onMouseLeave={(e) => handleGroundButtonHover(e, false)}
              disabled={loading}
            >
              {loading && activeType === "gt" ? (
                <>⏳ Processing...</>
              ) : (
                <>✓ Run Ground Truth Batch</>
              )}
            </button>
          </div>
        </div>

        <div style={styles.resultSection}>
          <div style={styles.resultHeader}>
            <div style={styles.resultTitle}>
              <span>📈</span>
              <span>Batch Results</span>
              {result && !loading && !result.error && (
                <span style={styles.resultBadge}>
                  {activeType === "high" ? "High Risk" : "Ground Truth"}
                </span>
              )}
            </div>
            {result && !loading && result.timestamp && (
              <div style={{ fontSize: '12px', color: '#8e8e8e' }}>
                {new Date(result.timestamp).toLocaleString()}
              </div>
            )}
          </div>
          
          {loading ? (
            <div style={styles.loadingOverlay}>
              <div style={styles.spinner}></div>
              <div style={styles.loadingText}>
                Running {activeType === "high" ? "High Risk" : "Ground Truth"} batch evaluation...
              </div>
              <div style={styles.loadingSubtext}>
                Processing up to {limit} turns. This may take a moment.
              </div>
            </div>
          ) : result ? (
            <>
              {stats && !result.error && (
                <div style={styles.statsGrid}>
                  {stats.total && (
                    <div style={styles.statCard}>
                      <div style={styles.statLabel}>Total Turns</div>
                      <div style={styles.statValue}>{stats.total}</div>
                    </div>
                  )}
                  {stats.processed && (
                    <div style={styles.statCard}>
                      <div style={styles.statLabel}>Processed</div>
                      <div style={styles.statValue}>{stats.processed}</div>
                    </div>
                  )}
                  {stats.success && (
                    <div style={styles.statCard}>
                      <div style={styles.statLabel}>Successful</div>
                      <div style={styles.statValue}>{stats.success}</div>
                    </div>
                  )}
                  {stats.failed && (
                    <div style={styles.statCard}>
                      <div style={styles.statLabel}>Failed</div>
                      <div style={styles.statValue}>{stats.failed}</div>
                    </div>
                  )}
                  {stats.duration && (
                    <div style={styles.statCard}>
                      <div style={styles.statLabel}>Duration</div>
                      <div style={styles.statValue}>{stats.duration}s</div>
                    </div>
                  )}
                </div>
              )}
              
              <div style={{
                ...styles.resultContent,
                ...(result.error ? styles.errorState : {})
              }}>
                <pre style={styles.pre}>
                  {JSON.stringify(result, null, 2)}
                </pre>
              </div>
            </>
          ) : (
            <div style={styles.emptyState}>
              <div style={{ fontSize: '48px', marginBottom: '16px' }}>🚀</div>
              <p style={{ margin: '8px 0 0 0', fontSize: '14px' }}>
                Configure batch size and click a button to start batch evaluation
              </p>
              <p style={{ margin: '8px 0 0 0', fontSize: '12px', color: '#c0c0c0' }}>
                Batch evaluation processes multiple turns at once for efficiency
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}