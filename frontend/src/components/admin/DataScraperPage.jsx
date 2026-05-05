import { Database, FileText, Sparkles, Upload } from 'lucide-react';
import { mockScraperData } from '../../utils/mockAdminData';

/**
 * Placeholder page for corpus data management and synthetic data generation.
 * Will be connected to scraper/corpus endpoints in Phase 2.
 */
export default function DataScraperPage() {
  return (
    <div className="dashboardPage">
      <div className="dashboardHeader">
        <div>
          <h2>Data & Corpus Management</h2>
          <p>Manage source documents and synthetic data generation</p>
        </div>
      </div>

      <div className="kpiGrid">
        <div className="kpiCard">
          <div className="kpiIcon" style={{ backgroundColor: 'rgba(59, 130, 246, 0.1)', color: '#3b82f6' }}>
            <FileText size={20} />
          </div>
          <div className="kpiBody">
            <span className="kpiValue">{mockScraperData.totalDocuments.toLocaleString()}</span>
            <span className="kpiLabel">Total Documents</span>
          </div>
        </div>
        <div className="kpiCard">
          <div className="kpiIcon" style={{ backgroundColor: 'rgba(15, 118, 110, 0.1)', color: '#0f766e' }}>
            <Database size={20} />
          </div>
          <div className="kpiBody">
            <span className="kpiValue">{mockScraperData.activeSources}</span>
            <span className="kpiLabel">Active Sources</span>
          </div>
        </div>
        <div className="kpiCard">
          <div className="kpiIcon" style={{ backgroundColor: 'rgba(139, 92, 246, 0.1)', color: '#8b5cf6' }}>
            <Sparkles size={20} />
          </div>
          <div className="kpiBody">
            <span className="kpiValue">{mockScraperData.syntheticGenerated.toLocaleString()}</span>
            <span className="kpiLabel">Synthetic Generated</span>
          </div>
        </div>
        <div className="kpiCard">
          <div className="kpiIcon" style={{ backgroundColor: 'rgba(245, 158, 11, 0.1)', color: '#f59e0b' }}>
            <Upload size={20} />
          </div>
          <div className="kpiBody">
            <span className="kpiValue">{new Date(mockScraperData.lastScrapeDate).toLocaleDateString()}</span>
            <span className="kpiLabel">Last Scrape</span>
          </div>
        </div>
      </div>

      <div className="adminCard placeholderCard">
        <Database size={48} />
        <h3>Corpus Management Coming Soon</h3>
        <p>This page will provide tools for managing the medical corpus, triggering scraper jobs, monitoring ingestion status, and controlling synthetic data generation pipelines.</p>
      </div>
    </div>
  );
}
