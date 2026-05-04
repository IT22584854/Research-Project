import React, { useState } from 'react';
import { BarChart2, MessageSquare, Database, LogOut } from 'lucide-react';
import IntentAnalyticsDashboard from './IntentAnalyticsDashboard';
import AgentEvaluationPage from './AgentEvaluationPage';
import DataScraperPage from './DataScraperPage';

const NAV_ITEMS = [
  { id: 'intent', label: 'Intent Classifier Analytics', icon: BarChart2 },
  { id: 'evaluation', label: 'Agent Response Evaluation', icon: MessageSquare },
  { id: 'scraper', label: 'Data Scraper & Synthetic Data', icon: Database },
];

export default function AdminPanelLayout({ onLogout }) {
  const [activePage, setActivePage] = useState('intent');

  const renderPage = () => {
    switch (activePage) {
      case 'intent': return <IntentAnalyticsDashboard />;
      case 'evaluation': return <AgentEvaluationPage />;
      case 'scraper': return <DataScraperPage />;
      default: return <IntentAnalyticsDashboard />;
    }
  };

  return (
    <div className="flex h-screen w-full bg-[#f8f9fa] overflow-hidden">
      
      {/* Sidebar — permanently open, 20% width */}
      <aside className="w-1/5 min-w-[220px] max-w-[280px] bg-white border-r border-gray-200 flex flex-col h-full shrink-0">
        {/* Logo Area */}
        <div className="p-5 border-b border-gray-100 flex items-center gap-3">
          <div className="w-10 h-10 bg-emerald-500 rounded-lg flex items-center justify-center rotate-45 shadow-md">
            <span className="text-white font-bold text-lg -rotate-45">G</span>
          </div>
          <div>
            <h1 className="font-semibold text-gray-900 text-sm leading-tight">GovHealth</h1>
            <p className="text-xs text-gray-400 mt-0.5">Admin Portal</p>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 overflow-y-auto p-3 space-y-1">
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const isActive = activePage === item.id;
            return (
              <button
                key={item.id}
                onClick={() => setActivePage(item.id)}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all
                  ${isActive 
                    ? 'bg-emerald-50 text-emerald-700' 
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                  }
                `}
              >
                <Icon size={18} className={isActive ? 'text-emerald-600' : 'text-gray-400'} />
                <span className="truncate">{item.label}</span>
              </button>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="p-3 border-t border-gray-100">
          <button 
            onClick={onLogout}
            className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium text-gray-500 hover:bg-gray-50 hover:text-gray-700 transition-colors"
          >
            <LogOut size={18} />
            Return to Chat
          </button>
        </div>
      </aside>

      {/* Main Content Area — 80% width */}
      <main className="flex-1 overflow-y-auto">
        {renderPage()}
      </main>

    </div>
  );
}
