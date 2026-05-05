import { useEffect, useState } from "react";
import "./AgentEvaluationPage.css";
import {
  ClipboardCheck,
  BarChart2,
  CheckCircle,
  Clock,
  AlertTriangle,
  Send,
  Database,
  FileText,
  Play,
  BookOpen,
  Zap,
  Loader,
} from "lucide-react";

import {
  getTurns,
  evalHighRiskTurn,
  evalGroundTruthTurn,
  manualHighRisk,
  manualGroundTruth,
  evalHighRiskSession,
  evalGroundTruthSession,
  batchHighRisk,
  batchGroundTruth,
} from "../../utils/evaluationApi";

/**
 * Fully connected Agent Evaluation Dashboard
 */
export default function AgentEvaluationPage() {
  // =========================
  // STATE
  // =========================
  const [turns, setTurns] = useState([]);
  const [loadingTurn, setLoadingTurn] = useState({});
  const [results, setResults] = useState({});
  const [errors, setErrors] = useState({});

  const [manualForm, setManualForm] = useState({
    question: "",
    answer: "",
    start_timestamp: "",
    end_timestamp: "",
  });

  const [manualResult, setManualResult] = useState(null);
  const [manualLoading, setManualLoading] = useState(false);
  const [manualError, setManualError] = useState(null);
  
  const [sessionId, setSessionId] = useState("");
  const [sessionResult, setSessionResult] = useState(null);
  const [sessionLoading, setSessionLoading] = useState(false);
  const [sessionError, setSessionError] = useState(null);

  const [batchStatus, setBatchStatus] = useState(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchError, setBatchError] = useState(null);

  // =========================
  // LOAD TURNS
  // =========================
  useEffect(() => {
    loadTurns();
  }, []);

  const loadTurns = async () => {
    try {
      const res = await getTurns();
      setTurns(res.data.turns || []);
    } catch (err) {
      console.error("Failed to load turns", err);
    }
  };

  // =========================
  // SINGLE TURN EVAL
  // =========================
  const runTurnEval = async (id, mode) => {
    setLoadingTurn((p) => ({ ...p, [id]: true }));
    setErrors((p) => ({ ...p, [id]: null }));

    try {
      const res =
        mode === "high"
          ? await evalHighRiskTurn(id)
          : await evalGroundTruthTurn(id);

      setResults((p) => ({
        ...p,
        [id]: res.data,
      }));
    } catch (err) {
      setErrors((p) => ({
        ...p,
        [id]: err.response?.data || err.message,
      }));
    } finally {
      setTimeout(() => {
        setLoadingTurn((p) => ({ ...p, [id]: false }));
      }, 500);
    }
  };

  // =========================
  // MANUAL EVAL
  // =========================
  const runManual = async (mode) => {
    setManualLoading(true);
    setManualResult(null);
    setManualError(null);
    
    try {
      const payload = {
        ...manualForm,
        start_timestamp: parseFloat(manualForm.start_timestamp),
        end_timestamp: parseFloat(manualForm.end_timestamp),
      };

      const res =
        mode === "high"
          ? await manualHighRisk(payload)
          : await manualGroundTruth(payload);

      setTimeout(() => {
        setManualResult(res.data);
        setManualLoading(false);
      }, 500);
    } catch (err) {
      setTimeout(() => {
        setManualError(err.response?.data || err.message);
        setManualLoading(false);
      }, 500);
    }
  };

  // =========================
  // SESSION EVAL
  // =========================
  const runSessionEval = async (mode) => {
    setSessionLoading(true);
    setSessionResult(null);
    setSessionError(null);
    
    try {
      const res =
        mode === "high"
          ? await evalHighRiskSession(sessionId)
          : await evalGroundTruthSession(sessionId);

      setTimeout(() => {
        setSessionResult(res.data);
        setSessionLoading(false);
      }, 500);
    } catch (err) {
      setTimeout(() => {
        setSessionError(err.response?.data || err.message);
        setSessionLoading(false);
      }, 500);
    }
  };

  // =========================
  // BATCH
  // =========================
  const runBatch = async (mode) => {
    setBatchLoading(true);
    setBatchStatus(null);
    setBatchError(null);
    
    try {
      const res =
        mode === "high"
          ? await batchHighRisk(20)
          : await batchGroundTruth(20);

      setTimeout(() => {
        setBatchStatus(res.data);
        setBatchLoading(false);
      }, 500);
    } catch (err) {
      setTimeout(() => {
        setBatchError(err.response?.data || err.message);
        setBatchLoading(false);
      }, 500);
    }
  };

  // Helper for auto-fill timestamps
  const fillNow = () => {
    const now = Date.now() / 1000;
    setManualForm((prev) => ({
      ...prev,
      start_timestamp: now.toString(),
      end_timestamp: (now + 1).toString(),
    }));
  };

  // Helper to get rating color
  const getRatingColor = (rating) => {
    if (!rating) return '#8e8e8e';
    if (rating === 'High Risk') return '#dc3545';
    if (rating === 'Medium Risk') return '#ffc107';
    if (rating === 'Low Risk') return '#28a745';
    return '#0095f6';
  };

  // Helper to format metrics display
  const formatMetricValue = (value) => {
    if (typeof value === 'number') {
      return value.toFixed(3);
    }
    return String(value);
  };

  // =========================
  // UI
  // =========================
  return (
    <div className="dashboardPage">

      {/* HEADER */}
      <div className="dashboardHeader">
        <div className="headerContent">
          <div>
            <h2>Agent Response Evaluation</h2>
            <p>Quality assessment and scoring of agent responses</p>
          </div>
          <div className="headerBadge">
            <Zap size={16} />
            <span>Real-time Evaluation</span>
          </div>
        </div>
      </div>

      {/* KPI GRID */}
      <div className="kpiGrid">
        <div className="kpiCard">
          <div className="kpiIcon" style={{ background: 'rgba(0,149,246,0.1)', color: '#0095f6' }}>
            <ClipboardCheck size={20} />
          </div>
          <div className="kpiBody">
            <span className="kpiValue">{turns.length}</span>
            <span className="kpiLabel">Loaded Turns</span>
          </div>
        </div>
        <div className="kpiCard">
          <div className="kpiIcon" style={{ background: 'rgba(40,167,69,0.1)', color: '#28a745' }}>
            <CheckCircle size={20} />
          </div>
          <div className="kpiBody">
            <span className="kpiValue">
              {Object.keys(results).filter(id => !errors[id] && results[id] && !results[id]?.error).length}
            </span>
            <span className="kpiLabel">Successful Evals</span>
          </div>
        </div>
        <div className="kpiCard">
          <div className="kpiIcon" style={{ background: 'rgba(220,53,69,0.1)', color: '#dc3545' }}>
            <AlertTriangle size={20} />
          </div>
          <div className="kpiBody">
            <span className="kpiValue">
              {Object.keys(errors).filter(id => errors[id]).length}
            </span>
            <span className="kpiLabel">Failed Evals</span>
          </div>
        </div>
        <div className="kpiCard">
          <div className="kpiIcon" style={{ background: 'rgba(255,193,7,0.1)', color: '#ffc107' }}>
            <BarChart2 size={20} />
          </div>
          <div className="kpiBody">
            <span className="kpiValue">
              {Object.values(results).filter(r => r?.rating === 'High Risk').length}
            </span>
            <span className="kpiLabel">High Risk Turns</span>
          </div>
        </div>
      </div>

      {/* =========================
          MANUAL EVALUATION
      ========================= */}
      <div className="adminCard">
        <div className="cardHeader">
          <div className="cardTitle">
            <FileText size={20} />
            <h3>Manual Evaluation</h3>
          </div>
          <div className="cardBadge">Test custom inputs</div>
        </div>

        <div className="formGrid">
          <div className="formGroup fullWidth">
            <label className="formLabel">Question</label>
            <input
              className="formInput"
              placeholder="Enter the question..."
              value={manualForm.question}
              onChange={(e) =>
                setManualForm({ ...manualForm, question: e.target.value })
              }
            />
          </div>

          <div className="formGroup fullWidth">
            <label className="formLabel">Answer</label>
            <textarea
              className="formTextarea"
              placeholder="Enter the answer to evaluate..."
              rows={4}
              value={manualForm.answer}
              onChange={(e) =>
                setManualForm({ ...manualForm, answer: e.target.value })
              }
            />
          </div>

          <div className="formGroup">
            <label className="formLabel">Start Timestamp (Unix)</label>
            <input
              className="formInput"
              placeholder="e.g., 1704067200"
              value={manualForm.start_timestamp}
              onChange={(e) =>
                setManualForm({ ...manualForm, start_timestamp: e.target.value })
              }
            />
          </div>

          <div className="formGroup">
            <label className="formLabel">End Timestamp (Unix)</label>
            <div style={{ display: 'flex', gap: '12px' }}>
              <input
                className="formInput"
                placeholder="e.g., 1704067260"
                value={manualForm.end_timestamp}
                onChange={(e) =>
                  setManualForm({ ...manualForm, end_timestamp: e.target.value })
                }
              />
              <button className="autoFillButton" onClick={fillNow}>
                <Clock size={14} />
                Auto-fill
              </button>
            </div>
          </div>
        </div>

        <div className="buttonGroup">
          <button 
            className="riskButton" 
            onClick={() => runManual("high")}
            disabled={manualLoading}
          >
            {manualLoading ? (
              <>
                <Loader size={16} className="spinning" />
                Evaluating...
              </>
            ) : (
              <>
                <AlertTriangle size={16} />
                High Risk Evaluation
              </>
            )}
          </button>
          <button 
            className="groundButton" 
            onClick={() => runManual("gt")}
            disabled={manualLoading}
          >
            {manualLoading ? (
              <>
                <Loader size={16} className="spinning" />
                Evaluating...
              </>
            ) : (
              <>
                <CheckCircle size={16} />
                Ground Truth Evaluation
              </>
            )}
          </button>
        </div>

        {/* Manual Evaluation Result Display */}
        {manualLoading && (
          <div className="loadingOverlay">
            <div className="loadingContent">
              <Loader size={32} className="spinning" />
              <p>Evaluating your custom input...</p>
              <span className="loadingSubtext">This may take a few seconds</span>
            </div>
          </div>
        )}

        {manualError && !manualLoading && (
          <div className="errorBox">
            <div className="errorTitle">❌ Evaluation Error</div>
            <pre className="errorContent">
              {typeof manualError === 'object' ? JSON.stringify(manualError, null, 2) : manualError}
            </pre>
          </div>
        )}

        {manualResult && !manualLoading && (
          <div className="resultBox">
            <div className="resultHeader">📊 Evaluation Result</div>
            
            <div className="metricsGrid">
              {manualResult.final_score !== undefined && (
                <div className="metricCard">
                  <div className="metricLabel">Final Score</div>
                  <div className="metricValue">
                    {(manualResult.final_score * 100).toFixed(1)}%
                  </div>
                </div>
              )}
              {manualResult.rating && (
                <div className="metricCard">
                  <div className="metricLabel">Rating</div>
                  <div className="metricValue" style={{ color: getRatingColor(manualResult.rating), fontSize: '20px' }}>
                    {manualResult.rating}
                  </div>
                </div>
              )}
              {manualResult.confidence !== undefined && (
                <div className="metricCard">
                  <div className="metricLabel">Confidence</div>
                  <div className="metricValue">
                    {(manualResult.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              )}
            </div>

            {manualResult.metrics && Object.keys(manualResult.metrics).length > 0 && (
              <div className="metricsSection">
                <div className="metricsTitle">Detailed Metrics</div>
                <div className="metricsList">
                  {Object.entries(manualResult.metrics).map(([key, value]) => (
                    <div key={key} className="metricItem">
                      <span className="metricItemLabel">{key.replace(/_/g, ' ')}</span>
                      <span className="metricItemValue">{formatMetricValue(value)}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <details className="details">
              <summary>View Full JSON Response</summary>
              <pre className="jsonPre">{JSON.stringify(manualResult, null, 2)}</pre>
            </details>
          </div>
        )}
      </div>

      {/* =========================
          SESSION EVALUATION
      ========================= */}
      <div className="adminCard">
        <div className="cardHeader">
          <div className="cardTitle">
            <Database size={20} />
            <h3>Session Evaluation</h3>
          </div>
          <div className="cardBadge">Evaluate entire session</div>
        </div>

        <div className="formGroup" style={{ padding: '0 24px' }}>
          <label className="formLabel">Session ID</label>
          <input
            className="formInput"
            placeholder="Enter session ID (e.g., session_123)"
            value={sessionId}
            onChange={(e) => setSessionId(e.target.value)}
          />
        </div>

        <div className="buttonGroup">
          <button 
            className="riskButton" 
            onClick={() => runSessionEval("high")}
            disabled={sessionLoading || !sessionId.trim()}
          >
            {sessionLoading ? (
              <>
                <Loader size={16} className="spinning" />
                Evaluating Session...
              </>
            ) : (
              <>
                <AlertTriangle size={16} />
                High Risk Session
              </>
            )}
          </button>
          <button 
            className="groundButton" 
            onClick={() => runSessionEval("gt")}
            disabled={sessionLoading || !sessionId.trim()}
          >
            {sessionLoading ? (
              <>
                <Loader size={16} className="spinning" />
                Evaluating Session...
              </>
            ) : (
              <>
                <CheckCircle size={16} />
                Ground Truth Session
              </>
            )}
          </button>
        </div>

        {sessionLoading && (
          <div className="loadingInline">
            <Loader size={20} className="spinning" />
            <span>Evaluating session {sessionId}...</span>
          </div>
        )}

        {sessionError && !sessionLoading && (
          <div className="errorBox">
            <div className="errorTitle">❌ Session Evaluation Error</div>
            <pre className="errorContent">
              {typeof sessionError === 'object' ? JSON.stringify(sessionError, null, 2) : sessionError}
            </pre>
          </div>
        )}

        {sessionResult && !sessionLoading && (
          <div className="resultBox">
            <div className="resultHeader">📊 Session Result</div>
            
            <div className="metricsGrid">
              {sessionResult.final_score !== undefined && (
                <div className="metricCard">
                  <div className="metricLabel">Average Score</div>
                  <div className="metricValue">
                    {(sessionResult.final_score * 100).toFixed(1)}%
                  </div>
                </div>
              )}
              {sessionResult.total_turns !== undefined && (
                <div className="metricCard">
                  <div className="metricLabel">Total Turns</div>
                  <div className="metricValue">{sessionResult.total_turns}</div>
                </div>
              )}
            </div>

            <details className="details">
              <summary>View Full JSON Response</summary>
              <pre className="jsonPre">{JSON.stringify(sessionResult, null, 2)}</pre>
            </details>
          </div>
        )}
      </div>

      {/* =========================
          BATCH EVALUATION
      ========================= */}
      <div className="adminCard">
        <div className="cardHeader">
          <div className="cardTitle">
            <Send size={20} />
            <h3>Batch Evaluation</h3>
          </div>
          <div className="cardBadge">Process 20 turns</div>
        </div>

        <div className="buttonGroup">
          <button 
            className="riskButton" 
            onClick={() => runBatch("high")}
            disabled={batchLoading}
          >
            {batchLoading ? (
              <>
                <Loader size={16} className="spinning" />
                Processing Batch...
              </>
            ) : (
              <>
                <Play size={16} />
                Run High Risk Batch
              </>
            )}
          </button>
          <button 
            className="groundButton" 
            onClick={() => runBatch("gt")}
            disabled={batchLoading}
          >
            {batchLoading ? (
              <>
                <Loader size={16} className="spinning" />
                Processing Batch...
              </>
            ) : (
              <>
                <Play size={16} />
                Run Ground Truth Batch
              </>
            )}
          </button>
        </div>

        {batchLoading && (
          <div className="loadingInline">
            <Loader size={20} className="spinning" />
            <span>Processing batch evaluation...</span>
          </div>
        )}

        {batchError && !batchLoading && (
          <div className="errorBox">
            <div className="errorTitle">❌ Batch Evaluation Error</div>
            <pre className="errorContent">
              {typeof batchError === 'object' ? JSON.stringify(batchError, null, 2) : batchError}
            </pre>
          </div>
        )}

        {batchStatus && !batchLoading && (
          <div className="resultBox">
            <div className="resultHeader">🚀 Batch Result</div>
            
            <div className="metricsGrid">
              {batchStatus.total_turns !== undefined && (
                <div className="metricCard">
                  <div className="metricLabel">Total Turns</div>
                  <div className="metricValue">{batchStatus.total_turns}</div>
                </div>
              )}
              {batchStatus.processed !== undefined && (
                <div className="metricCard">
                  <div className="metricLabel">Processed</div>
                  <div className="metricValue">{batchStatus.processed}</div>
                </div>
              )}
              {batchStatus.successful !== undefined && (
                <div className="metricCard">
                  <div className="metricLabel">Successful</div>
                  <div className="metricValue">{batchStatus.successful}</div>
                </div>
              )}
              {batchStatus.failed !== undefined && (
                <div className="metricCard">
                  <div className="metricLabel">Failed</div>
                  <div className="metricValue">{batchStatus.failed}</div>
                </div>
              )}
            </div>

            <details className="details">
              <summary>View Full JSON Response</summary>
              <pre className="jsonPre">{JSON.stringify(batchStatus, null, 2)}</pre>
            </details>
          </div>
        )}
      </div>

      {/* =========================
          TURN LIST + INLINE EVAL
      ========================= */}
      <div className="adminCard">
        <div className="cardHeader">
          <div className="cardTitle">
            <BookOpen size={20} />
            <h3>Agent Turns</h3>
          </div>
          <div className="cardBadge">{turns.length} total turns</div>
        </div>

        <div className="turnsList">
          {turns.map((t, index) => (
            <div key={t.id} className="turnItem">
              <div className="turnHeader">
                <span className="turnNumber">Turn #{index + 1}</span>
                <span className="turnId">ID: {t.id}</span>
              </div>

              <div className="turnContent">
                <div className="messageBlock">
                  <div className="messageLabel">❓ Question</div>
                  <p className="questionText">{t.user_message}</p>
                </div>

                <div className="messageBlock">
                  <div className="messageLabel">💬 Answer</div>
                  <p className="answerText">{t.assistant_message}</p>
                </div>

                <div className="turnButtons">
                  <button
                    className="smallRiskButton"
                    onClick={() => runTurnEval(t.id, "high")}
                    disabled={loadingTurn[t.id]}
                  >
                    {loadingTurn[t.id] ? (
                      <>
                        <Loader size={12} className="spinning" />
                        Evaluating...
                      </>
                    ) : (
                      <>⚠️ High Risk</>
                    )}
                  </button>

                  <button
                    className="smallGroundButton"
                    onClick={() => runTurnEval(t.id, "gt")}
                    disabled={loadingTurn[t.id]}
                  >
                    {loadingTurn[t.id] ? (
                      <>
                        <Loader size={12} className="spinning" />
                        Evaluating...
                      </>
                    ) : (
                      <>✓ Ground Truth</>
                    )}
                  </button>
                </div>

                {/* Loading Indicator */}
                {loadingTurn[t.id] && (
                  <div className="loadingInline">
                    <Loader size={16} className="spinning" />
                    <span>Evaluating turn {t.id}...</span>
                  </div>
                )}

                {/* Error Display */}
                {errors[t.id] && !loadingTurn[t.id] && (
                  <div className="errorBox">
                    <div className="errorTitle">❌ Evaluation Error</div>
                    <pre className="errorContent">
                      {typeof errors[t.id] === 'object' 
                        ? JSON.stringify(errors[t.id], null, 2)
                        : errors[t.id]}
                    </pre>
                  </div>
                )}

                {/* Result Display - Similar to Turns component */}
                {results[t.id] && !loadingTurn[t.id] && (
                  <div className="resultBox">
                    <div className="resultTitle">
                      <span>📊</span>
                      <span>Evaluation Results</span>
                    </div>

                    <div className="metricsGrid">
                      {results[t.id].final_score !== undefined && (
                        <div className="metricCard">
                          <div className="metricLabel">Final Score</div>
                          <div className="metricValue">
                            {(results[t.id].final_score * 100).toFixed(1)}%
                          </div>
                        </div>
                      )}
                      
                      {results[t.id].rating && (
                        <div className="metricCard">
                          <div className="metricLabel">Rating</div>
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
                        <div className="metricCard">
                          <div className="metricLabel">Confidence</div>
                          <div className="metricValue">
                            {(results[t.id].confidence * 100).toFixed(1)}%
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Detailed Metrics */}
                    {results[t.id].metrics && Object.keys(results[t.id].metrics).length > 0 && (
                      <div className="metricsSection">
                        <div className="metricsTitle">Detailed Metrics</div>
                        <div className="metricsList">
                          {Object.entries(results[t.id].metrics).map(([key, value]) => (
                            <div key={key} className="metricItem">
                              <span className="metricItemLabel">{key.replace(/_/g, ' ')}</span>
                              <span className="metricItemValue">{formatMetricValue(value)}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    <details className="details">
                      <summary>View Full JSON Response</summary>
                      <pre className="jsonPre">
                        {JSON.stringify(results[t.id], null, 2)}
                      </pre>
                    </details>
                  </div>
                )}
              </div>
            </div>
          ))}

          {turns.length === 0 && (
            <div className="emptyState">
              <div className="emptyStateIcon">📭</div>
              <p className="emptyStateText">No turns available. Please check your connection or refresh.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Styles object for dynamic styling
const styles = {
  metricValue: {}
};