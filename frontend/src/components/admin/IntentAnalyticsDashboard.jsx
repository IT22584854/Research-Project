import React, { useState, useRef, useEffect } from 'react';
import { Download } from 'lucide-react';
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ResponsiveContainer } from 'recharts';
import ForceGraph2D from 'react-force-graph-2d';
import { mockKPIs, mockRadarData, mockGraphData, mockTableData } from '../../utils/mockAdminData';

export default function IntentAnalyticsDashboard() {
  const [filterLang, setFilterLang] = useState('all');
  const [filterIntent, setFilterIntent] = useState('all');
  const graphContainerRef = useRef(null);
  const [graphDimensions, setGraphDimensions] = useState({ width: 400, height: 300 });

  useEffect(() => {
    const container = graphContainerRef.current;
    if (!container) return;

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setGraphDimensions({
          width: entry.contentRect.width,
          height: entry.contentRect.height
        });
      }
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const filteredData = mockTableData.filter(row => {
    if (filterLang !== 'all' && row.lang !== filterLang) return false;
    if (filterIntent !== 'all' && row.intent !== filterIntent) return false;
    return true;
  });

  return (
    <div className="p-6 lg:p-8 space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-lg font-semibold text-gray-900">Intent Classifier Analytics</h2>
        <p className="text-sm text-gray-500 mt-1">Real-time classification metrics and keyword analysis</p>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <KPICard title="Total Queries" value={mockKPIs.totalQueries} />
        <KPICard title="Unique Users" value={mockKPIs.uniqueUsers} />
        <KPICard title="Most Common Intent" value={mockKPIs.commonIntent} isText />
        <KPICard title="Model Being Used" value={mockKPIs.modelUsed} isText />
      </div>

      {/* Graphs Row */}
      <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
        {/* Network Graph — takes 3/5 */}
        <div className="lg:col-span-3 admin-card p-5 flex flex-col">
          <h3 className="text-sm font-semibold text-gray-700 mb-4">Keyword Spread Linking Graph</h3>
          <div ref={graphContainerRef} className="flex-1 rounded-lg overflow-hidden bg-gray-50 border border-gray-100 min-h-[320px] relative">
            <ForceGraph2D
              graphData={mockGraphData}
              width={graphDimensions.width}
              height={graphDimensions.height}
              nodeLabel="id"
              nodeColor={node => {
                if (node.group === 1) return '#f87171';
                if (node.group === 2) return '#fbbf24';
                return '#34d399';
              }}
              nodeRelSize={6}
              linkColor={() => '#d1d5db'}
              backgroundColor="#f9fafb"
            />
          </div>
        </div>

        {/* Radar Graph — takes 2/5 */}
        <div className="lg:col-span-2 admin-card p-5 flex flex-col">
          <h3 className="text-sm font-semibold text-gray-700 mb-4">Radar Graph of Intents</h3>
          <div className="flex-1 w-full min-h-[320px]">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart cx="50%" cy="50%" outerRadius="75%" data={mockRadarData}>
                <PolarGrid stroke="#e2e8f0" />
                <PolarAngleAxis dataKey="subject" tick={{ fill: '#64748b', fontSize: 12, fontFamily: 'Inter' }} />
                <PolarRadiusAxis angle={30} domain={[0, 150]} tick={false} axisLine={false} />
                <Radar name="Intents" dataKey="A" stroke="#10b981" strokeWidth={2} fill="#10b981" fillOpacity={0.15} />
              </RadarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Table Section */}
      <div className="flex flex-col xl:flex-row gap-6">
        {/* Table */}
        <div className="flex-1 admin-card overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-100 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-gray-700">User Queries & Intent Classification</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left min-w-[800px]">
              <thead className="bg-gray-50 text-xs uppercase text-gray-500 border-b border-gray-100">
                <tr>
                  <th className="px-6 py-3 font-medium">User Query</th>
                  <th className="px-6 py-3 font-medium">Lang</th>
                  <th className="px-6 py-3 font-medium">Keywords</th>
                  <th className="px-6 py-3 font-medium">Intent</th>
                  <th className="px-6 py-3 font-medium">RAG Query</th>
                  <th className="px-6 py-3 font-medium">Reasoning</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {filteredData.map((row) => (
                  <tr key={row.id} className="hover:bg-gray-50/50 transition-colors">
                    <td className="px-6 py-4 text-gray-900 font-medium max-w-[200px] truncate" title={row.query}>{row.query}</td>
                    <td className="px-6 py-4">
                      <span className="inline-flex px-2 py-0.5 rounded-md text-xs font-medium bg-gray-100 text-gray-600">
                        {row.lang}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-gray-500">{row.keywords}</td>
                    <td className="px-6 py-4">
                      <span className="inline-flex px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-50 text-emerald-700">
                        {row.intent}
                      </span>
                    </td>
                    <td className="px-6 py-4 text-gray-500">{row.rag}</td>
                    <td className="px-6 py-4 text-gray-400 text-xs">{row.reasoning}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Filters Panel */}
        <div className="w-full xl:w-64 flex flex-col gap-4 shrink-0">
          <button className="admin-btn admin-btn-primary w-full">
            <Download size={14} />
            Download Data
          </button>

          <div className="admin-card p-5 flex-1">
            <h3 className="text-sm font-semibold text-gray-700 mb-4 pb-3 border-b border-gray-100">Filters</h3>
            
            <div className="space-y-5">
              <div>
                <label className="block text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">Language Tag</label>
                <select 
                  value={filterLang}
                  onChange={(e) => setFilterLang(e.target.value)}
                  className="admin-input text-sm"
                >
                  <option value="all">All Languages</option>
                  <option value="en">English (en)</option>
                  <option value="si">Sinhala (si)</option>
                  <option value="ta">Tamil (ta)</option>
                </select>
              </div>

              <div>
                <label className="block text-xs font-medium text-gray-500 uppercase tracking-wider mb-2">Intent</label>
                <div className="space-y-2">
                  {['all', 'Symptom Check', 'Appointment', 'Medication'].map((intent) => (
                    <label key={intent} className="flex items-center gap-2.5 text-sm text-gray-700 cursor-pointer hover:text-gray-900 transition-colors">
                      <input 
                        type="radio" 
                        name="intentFilter" 
                        value={intent}
                        checked={filterIntent === intent}
                        onChange={(e) => setFilterIntent(e.target.value)}
                        className="w-4 h-4 accent-emerald-600"
                      />
                      {intent === 'all' ? 'All Intents' : intent}
                    </label>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function KPICard({ title, value, isText }) {
  return (
    <div className="admin-card p-5">
      <p className="text-xs font-medium text-gray-500 uppercase tracking-wider">{title}</p>
      <p className={`${isText ? 'text-lg' : 'text-2xl'} font-semibold text-gray-900 mt-1`}>
        {isText ? value : value.toLocaleString()}
      </p>
    </div>
  );
}
