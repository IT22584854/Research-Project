import { ClipboardCheck, BarChart2, CheckCircle, Clock } from 'lucide-react';
import { mockEvaluationData } from '../../utils/mockAdminData';

/**
 * Placeholder page for agent response evaluation.
 * Will be connected to evaluation endpoints in Phase 2.
 */
export default function AgentEvaluationPage() {
  return (
    <div className="dashboardPage">
      <div className="dashboardHeader">
        <div>
          <h2>Agent Response Evaluation</h2>
          <p>Quality assessment and scoring of agent responses</p>
        </div>
      </div>

      <div className="kpiGrid">
        <div className="kpiCard">
          <div className="kpiIcon" style={{ backgroundColor: 'rgba(15, 118, 110, 0.1)', color: '#0f766e' }}>
            <ClipboardCheck size={20} />
          </div>
          <div className="kpiBody">
            <span className="kpiValue">{mockEvaluationData.totalEvaluations.toLocaleString()}</span>
            <span className="kpiLabel">Total Evaluations</span>
          </div>
        </div>
        <div className="kpiCard">
          <div className="kpiIcon" style={{ backgroundColor: 'rgba(16, 185, 129, 0.1)', color: '#10b981' }}>
            <CheckCircle size={20} />
          </div>
          <div className="kpiBody">
            <span className="kpiValue">{mockEvaluationData.passRate}%</span>
            <span className="kpiLabel">Pass Rate</span>
          </div>
        </div>
        <div className="kpiCard">
          <div className="kpiIcon" style={{ backgroundColor: 'rgba(245, 158, 11, 0.1)', color: '#f59e0b' }}>
            <BarChart2 size={20} />
          </div>
          <div className="kpiBody">
            <span className="kpiValue">{mockEvaluationData.avgScore}/5.0</span>
            <span className="kpiLabel">Avg Score</span>
          </div>
        </div>
        <div className="kpiCard">
          <div className="kpiIcon" style={{ backgroundColor: 'rgba(239, 68, 68, 0.1)', color: '#ef4444' }}>
            <Clock size={20} />
          </div>
          <div className="kpiBody">
            <span className="kpiValue">{mockEvaluationData.pendingReview}</span>
            <span className="kpiLabel">Pending Review</span>
          </div>
        </div>
      </div>

      <div className="adminCard placeholderCard">
        <ClipboardCheck size={48} />
        <h3>Evaluation Dashboard Coming Soon</h3>
        <p>This page will display detailed evaluation metrics, per-turn scoring breakdowns, and annotation tools for reviewing agent responses.</p>
      </div>
    </div>
  );
}
