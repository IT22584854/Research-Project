import { useState } from 'react';
import { BarChart2, ClipboardCheck, Database, LogOut, Stethoscope } from 'lucide-react';
import IntentAnalyticsDashboard from './IntentAnalyticsDashboard';
import AgentEvaluationPage from './AgentEvaluationPage';
import DataScraperPage from './DataScraperPage';

/**
 * Admin panel shell with a permanently visible 20%-width sidebar
 * and a scrollable content area that renders the selected page.
 *
 * ROUTING NOTE: We use simple React state (`activePage`) instead of
 * react-router-dom to keep the dependency footprint small. This is
 * intentional — the admin panel is a secondary view embedded inside
 * the existing chat SPA. If multi-tab deep-linking is ever needed,
 * swap this for a proper router.
 */

const NAV_ITEMS = [
  { id: 'analytics', label: 'Intent Analytics', icon: BarChart2 },
  { id: 'evaluation', label: 'Agent Evaluation', icon: ClipboardCheck },
  { id: 'scraper', label: 'Data & Corpus', icon: Database },
];

export default function AdminLayout({ onLogout }) {
  const [activePage, setActivePage] = useState('analytics');

  function renderPage() {
    switch (activePage) {
      case 'analytics':
        return <IntentAnalyticsDashboard />;
      case 'evaluation':
        return <AgentEvaluationPage />;
      case 'scraper':
        return <DataScraperPage />;
      default:
        return <IntentAnalyticsDashboard />;
    }
  }

  return (
    <div className="adminShell">
      {/* ── Sidebar (fixed 20% width) ─────────────────────────── */}
      <aside className="adminSidebar">
        <div className="adminBrand">
          <div className="adminBrandIcon">
            <Stethoscope size={22} />
          </div>
          <div>
            <h1>MedTriage</h1>
            <p>Admin Dashboard</p>
          </div>
        </div>

        <nav className="adminNav">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            return (
              <button
                key={item.id}
                className={`adminNavItem ${activePage === item.id ? 'active' : ''}`}
                type="button"
                onClick={() => setActivePage(item.id)}
              >
                <Icon size={18} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>

        <div className="adminSidebarFooter">
          <button className="adminNavItem adminLogout" type="button" onClick={onLogout}>
            <LogOut size={18} />
            <span>Back to Chat</span>
          </button>
        </div>
      </aside>

      {/* ── Content area ──────────────────────────────────────── */}
      <main className="adminContent">
        {renderPage()}
      </main>
    </div>
  );
}
