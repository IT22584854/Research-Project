import { useEffect, useRef, useState, useCallback } from 'react';
import * as d3 from 'd3';
import {
  mockKPIs,
  mockIntentRadarData,
  mockLanguageData,
  mockKeywordNodes,
  mockKeywordLinks,
  mockTableData,
  LANGUAGE_COLORS,
  KEYWORD_GROUP_COLORS,
  LANGUAGE_CATEGORIES,
} from '../../utils/mockAdminData';
import { Activity, Users, Zap, TrendingUp, Download, RefreshCw, AlertCircle, ChevronLeft, ChevronRight } from 'lucide-react';

const INTENT_CATEGORIES = [
  "Emergency_Triage",
  "Symptom_Information",
  "Disease_Information",
  "Facility_Locator",
  "Provider_Locator",
  "Appointment_Booking",
  "Medication_Information",
  "Vaccine_Information",
  "Test_Diagnostics",
  "Treatment_Procedure",
  "Cost_Insurance",
  "General_Health_Education",
  "Non_Medical",
  "Unclear"
];

/* ════════════════════════════════════════════════════════════════════
   D3 Radar Chart — 14 intent types
   ════════════════════════════════════════════════════════════════════ */
function IntentRadarChart({ data }) {
  const svgRef = useRef(null);
  const containerRef = useRef(null);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || !data || data.length === 0) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = Math.min(width, 420);
    const margin = 60;
    const radius = Math.min(width, height) / 2 - margin;
    const levels = 5;
    const maxValue = Math.max(...data.map((d) => d.count), 1); // fallback to 1

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    svg.attr('width', width).attr('height', height);

    const g = svg.append('g').attr('transform', `translate(${width / 2}, ${height / 2})`);

    const angleSlice = (2 * Math.PI) / data.length;
    const rScale = d3.scaleLinear().domain([0, maxValue]).range([0, radius]);

    // Grid circles
    for (let i = 1; i <= levels; i++) {
      g.append('circle')
        .attr('r', (radius * i) / levels)
        .attr('fill', 'none')
        .attr('stroke', '#e2e8f0')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3,3');
    }

    // Axis lines + labels
    data.forEach((d, i) => {
      const angle = angleSlice * i - Math.PI / 2;
      const x = radius * Math.cos(angle);
      const y = radius * Math.sin(angle);

      g.append('line')
        .attr('x1', 0).attr('y1', 0)
        .attr('x2', x).attr('y2', y)
        .attr('stroke', '#e2e8f0')
        .attr('stroke-width', 1);

      const labelX = (radius + 18) * Math.cos(angle);
      const labelY = (radius + 18) * Math.sin(angle);
      const label = d.intent.replace(/_/g, ' ');

      g.append('text')
        .attr('x', labelX)
        .attr('y', labelY)
        .attr('text-anchor', Math.abs(labelX) < 5 ? 'middle' : labelX > 0 ? 'start' : 'end')
        .attr('dominant-baseline', Math.abs(labelY) < 5 ? 'middle' : labelY > 0 ? 'hanging' : 'auto')
        .attr('font-size', '10px')
        .attr('fill', '#64748b')
        .text(label.length > 16 ? label.slice(0, 14) + '…' : label);
    });

    // Data polygon
    const lineGen = d3.lineRadial()
      .radius((d) => rScale(d.count))
      .angle((_, i) => i * angleSlice)
      .curve(d3.curveLinearClosed);

    g.append('path')
      .datum(data)
      .attr('d', lineGen)
      .attr('fill', 'rgba(15, 118, 110, 0.15)')
      .attr('stroke', '#0f766e')
      .attr('stroke-width', 2);

    // Data points
    data.forEach((d, i) => {
      const angle = angleSlice * i - Math.PI / 2;
      g.append('circle')
        .attr('cx', rScale(d.count) * Math.cos(angle))
        .attr('cy', rScale(d.count) * Math.sin(angle))
        .attr('r', 4)
        .attr('fill', '#0f766e')
        .attr('stroke', '#fff')
        .attr('stroke-width', 2);
    });
  }, [data]);

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={svgRef} />
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════
   D3 Donut Chart — 7 languages
   ════════════════════════════════════════════════════════════════════ */
function LanguageDonutChart({ data }) {
  const svgRef = useRef(null);
  const containerRef = useRef(null);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || !data || data.length === 0) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = Math.min(width, 420);
    const radius = Math.min(width, height) / 2 - 20;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    svg.attr('width', width).attr('height', height);

    const g = svg.append('g').attr('transform', `translate(${width / 2}, ${height / 2})`);

    const total = d3.sum(data, (d) => d.count);
    const pie = d3.pie().value((d) => d.count).sort(null).padAngle(0.02);
    const arc = d3.arc().innerRadius(radius * 0.55).outerRadius(radius);
    const arcHover = d3.arc().innerRadius(radius * 0.55).outerRadius(radius + 6);

    const arcs = g.selectAll('.arc')
      .data(pie(data))
      .join('g')
      .attr('class', 'arc');

    arcs.append('path')
      .attr('d', arc)
      .attr('fill', (d) => LANGUAGE_COLORS[d.data.language] || '#94a3b8')
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)
      .style('cursor', 'pointer')
      .style('transition', 'all 0.2s ease')
      .on('mouseenter', function (event, d) {
        d3.select(this).transition().duration(200).attr('d', arcHover);
      })
      .on('mouseleave', function (event, d) {
        d3.select(this).transition().duration(200).attr('d', arc);
      });

    // Center label
    g.append('text')
      .attr('text-anchor', 'middle')
      .attr('dominant-baseline', 'middle')
      .attr('font-size', '28px')
      .attr('font-weight', '700')
      .attr('fill', '#1e293b')
      .text(total.toLocaleString());

    g.append('text')
      .attr('text-anchor', 'middle')
      .attr('y', 24)
      .attr('font-size', '12px')
      .attr('fill', '#64748b')
      .text('total queries');

  }, [data]);

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={svgRef} />
      <div className="donutLegend">
        {data && data.map((d) => (
          <div key={d.language} className="donutLegendItem">
            <span
              className="donutLegendDot"
              style={{ backgroundColor: LANGUAGE_COLORS[d.language] || '#94a3b8' }}
            />
            <span className="donutLegendLabel">{d.language}</span>
            <span className="donutLegendValue">{d.count.toLocaleString()}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════
   D3 Force-Directed Keyword Network Graph
   ════════════════════════════════════════════════════════════════════ */
function KeywordNetworkGraph({ data }) {
  const svgRef = useRef(null);
  const containerRef = useRef(null);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current || !data || !data.nodes || !data.links) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = 400;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    svg.attr('width', width).attr('height', height).attr('viewBox', `0 0 ${width} ${height}`);

    // Deep clone data so D3 mutation doesn't break React re-renders
    const nodes = data.nodes.map((d) => ({ ...d }));
    const links = data.links.map((d) => ({ ...d }));

    if (nodes.length === 0) return;

    const sizeScale = d3.scaleSqrt()
      .domain(d3.extent(nodes, (d) => d.size))
      .range([6, 24]);

    const simulation = d3.forceSimulation(nodes)
      .force('link', d3.forceLink(links).id((d) => d.id).distance(80))
      .force('charge', d3.forceManyBody().strength(-200))
      .force('center', d3.forceCenter(width / 2, height / 2))
      .force('collision', d3.forceCollide().radius((d) => sizeScale(d.size) + 4));

    // Links
    const link = svg.append('g')
      .selectAll('line')
      .data(links)
      .join('line')
      .attr('stroke', '#cbd5e1')
      .attr('stroke-width', (d) => Math.max(1, d.value / 3))
      .attr('stroke-opacity', 0.6);

    // Nodes
    const node = svg.append('g')
      .selectAll('g')
      .data(nodes)
      .join('g')
      .style('cursor', 'grab')
      .call(d3.drag()
        .on('start', (event, d) => {
          if (!event.active) simulation.alphaTarget(0.3).restart();
          d.fx = d.x;
          d.fy = d.y;
        })
        .on('drag', (event, d) => {
          d.fx = event.x;
          d.fy = event.y;
        })
        .on('end', (event, d) => {
          if (!event.active) simulation.alphaTarget(0);
          d.fx = null;
          d.fy = null;
        })
      );

    node.append('circle')
      .attr('r', (d) => sizeScale(d.size))
      .attr('fill', (d) => KEYWORD_GROUP_COLORS[d.group] || '#94a3b8')
      .attr('fill-opacity', 0.85)
      .attr('stroke', '#fff')
      .attr('stroke-width', 2);

    node.append('text')
      .text((d) => d.id)
      .attr('text-anchor', 'middle')
      .attr('dy', (d) => sizeScale(d.size) + 14)
      .attr('font-size', '10px')
      .attr('fill', '#475569')
      .attr('pointer-events', 'none');

    simulation.on('tick', () => {
      link
        .attr('x1', (d) => d.source.x)
        .attr('y1', (d) => d.source.y)
        .attr('x2', (d) => d.target.x)
        .attr('y2', (d) => d.target.y);
      node.attr('transform', (d) => `translate(${d.x}, ${d.y})`);
    });

    return () => simulation.stop();
  }, [data]);

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={svgRef} />
      <div className="networkLegend">
        {Object.entries(KEYWORD_GROUP_COLORS).map(([group, color]) => (
          <div key={group} className="donutLegendItem">
            <span className="donutLegendDot" style={{ backgroundColor: color }} />
            <span className="donutLegendLabel">{group}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════
   Main Dashboard Page
   ════════════════════════════════════════════════════════════════════ */
export default function IntentAnalyticsDashboard() {
  const [logs, setLogs] = useState([]);
  const [stats, setStats] = useState(null);
  const [keywordsData, setKeywordsData] = useState(null);
  
  const [loading, setLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [fallbackMode, setFallbackMode] = useState(false);
  
  // Pagination & Filters
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);
  const [totalCount, setTotalCount] = useState(0);
  const [totalPages, setTotalPages] = useState(0);
  
  const [searchTerm, setSearchTerm] = useState('');
  const [debouncedSearchTerm, setDebouncedSearchTerm] = useState('');
  const [intentFilter, setIntentFilter] = useState('');
  const [languageFilter, setLanguageFilter] = useState('');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');

  const [lastUpdated, setLastUpdated] = useState(new Date());

  // Debounce search
  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedSearchTerm(searchTerm);
      setPage(1); // Reset to page 1 on search change
    }, 300);
    return () => clearTimeout(handler);
  }, [searchTerm]);

  const fetchStatsAndKeywords = useCallback(async () => {
    try {
      const statsRes = await fetch('/api/admin/router-logs/stats');
      if (!statsRes.ok) throw new Error('Stats fetch failed');
      const statsData = await statsRes.json();
      setStats(statsData);

      const kwRes = await fetch('/api/admin/router-logs/keywords');
      if (kwRes.ok) {
        const kwData = await kwRes.json();
        setKeywordsData(kwData);
      }
      setFallbackMode(false);
    } catch (err) {
      console.error(err);
      setFallbackMode(true);
      setStats({
        totalQueries: mockKPIs.totalQueries,
        uniqueSessions: mockKPIs.uniqueSessions,
        topIntent: mockKPIs.topIntent,
        avgLatencyMs: mockKPIs.avgLatencyMs,
        intentDistribution: mockIntentRadarData,
        languageDistribution: mockLanguageData
      });
      setKeywordsData({
        nodes: mockKeywordNodes,
        links: mockKeywordLinks
      });
    }
  }, []);

  const fetchLogs = useCallback(async () => {
    try {
      const params = new URLSearchParams({
        page: page,
        page_size: pageSize
      });
      if (debouncedSearchTerm) params.append('search', debouncedSearchTerm);
      if (intentFilter) params.append('intent', intentFilter);
      if (languageFilter) params.append('language', languageFilter);
      if (dateFrom) params.append('date_from', new Date(dateFrom).toISOString());
      if (dateTo) params.append('date_to', new Date(dateTo).toISOString());

      const res = await fetch(`/api/admin/router-logs?${params.toString()}`);
      if (!res.ok) throw new Error('Logs fetch failed');
      
      const data = await res.json();
      setLogs(data.data);
      setTotalCount(data.total_count);
      setTotalPages(data.total_pages);
    } catch (err) {
      console.error(err);
      setFallbackMode(true);
      // Client-side filtering for fallback
      let filtered = mockTableData.filter(row => {
        if (debouncedSearchTerm) {
          const term = debouncedSearchTerm.toLowerCase();
          if (!row.user_message.toLowerCase().includes(term) && 
              !row.session_id.toLowerCase().includes(term)) return false;
        }
        if (intentFilter && row.intent !== intentFilter) return false;
        if (languageFilter && row.language !== languageFilter) return false;
        return true;
      });
      
      setTotalCount(filtered.length);
      setTotalPages(Math.ceil(filtered.length / pageSize));
      const start = (page - 1) * pageSize;
      setLogs(filtered.slice(start, start + pageSize));
    }
  }, [page, pageSize, debouncedSearchTerm, intentFilter, languageFilter, dateFrom, dateTo]);

  const loadAllData = useCallback(async () => {
    setLoading(true);
    await Promise.all([fetchStatsAndKeywords(), fetchLogs()]);
    setLastUpdated(new Date());
    setLoading(false);
  }, [fetchStatsAndKeywords, fetchLogs]);

  useEffect(() => {
    loadAllData();
  }, [loadAllData]);

  // Refetch logs when dependencies change (but stats don't need to re-fetch on every filter)
  useEffect(() => {
    if (!loading) {
      fetchLogs();
    }
  }, [page, pageSize, debouncedSearchTerm, intentFilter, languageFilter, dateFrom, dateTo, fetchLogs, loading]);

  const handleRefresh = async () => {
    setIsRefreshing(true);
    await loadAllData();
    setIsRefreshing(false);
  };

  const handleExport = () => {
    const params = new URLSearchParams();
    if (debouncedSearchTerm) params.append('search', debouncedSearchTerm);
    if (intentFilter) params.append('intent', intentFilter);
    if (languageFilter) params.append('language', languageFilter);
    if (dateFrom) params.append('date_from', new Date(dateFrom).toISOString());
    if (dateTo) params.append('date_to', new Date(dateTo).toISOString());

    window.open(`/api/admin/router-logs/export?${params.toString()}`, '_blank');
  };

  const clearFilters = () => {
    setSearchTerm('');
    setIntentFilter('');
    setLanguageFilter('');
    setDateFrom('');
    setDateTo('');
    setPage(1);
  };

  return (
    <div className="dashboardPage">
      {fallbackMode && (
        <div className="fallbackBanner">
          <AlertCircle size={18} />
          <span>API connection failed. Showing sample data for demonstration.</span>
        </div>
      )}

      <div className="dashboardHeader">
        <div>
          <h2>Intent Routing Analytics</h2>
          <p>Real-time analysis of intent classifier routing decisions</p>
        </div>
        <div className="headerActions">
          <span className="lastUpdatedText">
            Last updated: {lastUpdated.toLocaleTimeString()}
          </span>
          <button className={`iconAdminBtn ${isRefreshing ? 'refreshing' : ''}`} onClick={handleRefresh} title="Refresh Data">
            <RefreshCw size={18} />
          </button>
          <button className="adminBtn" type="button" onClick={handleExport}>
            <Download size={16} />
            Export CSV
          </button>
        </div>
      </div>

      {/* ── KPI Cards ────────────────────────────────────────── */}
      <div className="kpiGrid">
        <div className="kpiCard">
          <div className="kpiIcon" style={{ backgroundColor: 'rgba(15, 118, 110, 0.1)', color: '#0f766e' }}>
            <Activity size={20} />
          </div>
          <div className="kpiBody">
            <span className="kpiValue">{stats?.totalQueries?.toLocaleString() || 0}</span>
            <span className="kpiLabel">Total Queries</span>
          </div>
        </div>
        <div className="kpiCard">
          <div className="kpiIcon" style={{ backgroundColor: 'rgba(59, 130, 246, 0.1)', color: '#3b82f6' }}>
            <Users size={20} />
          </div>
          <div className="kpiBody">
            <span className="kpiValue">{stats?.uniqueSessions?.toLocaleString() || 0}</span>
            <span className="kpiLabel">Unique Sessions</span>
          </div>
        </div>
        <div className="kpiCard">
          <div className="kpiIcon" style={{ backgroundColor: 'rgba(245, 158, 11, 0.1)', color: '#f59e0b' }}>
            <TrendingUp size={20} />
          </div>
          <div className="kpiBody">
            <span className="kpiValue">{stats?.topIntent ? stats.topIntent.replace(/_/g, ' ') : 'N/A'}</span>
            <span className="kpiLabel">Top Intent</span>
          </div>
        </div>
        <div className="kpiCard">
          <div className="kpiIcon" style={{ backgroundColor: 'rgba(139, 92, 246, 0.1)', color: '#8b5cf6' }}>
            <Zap size={20} />
          </div>
          <div className="kpiBody">
            <span className="kpiValue">{stats?.avgLatencyMs?.toLocaleString() || 0}ms</span>
            <span className="kpiLabel">Avg Latency</span>
          </div>
        </div>
      </div>

      {/* ── Charts Row ───────────────────────────────────────── */}
      <div className="chartsRow">
        <div className="adminCard">
          <h3>Intent Distribution</h3>
          <p className="chartSubtitle">Radar view of 14 intent categories</p>
          {stats?.intentDistribution && <IntentRadarChart data={stats.intentDistribution} />}
        </div>
        <div className="adminCard">
          <h3>Language Distribution</h3>
          <p className="chartSubtitle">Breakdown of 7 supported languages</p>
          {stats?.languageDistribution && <LanguageDonutChart data={stats.languageDistribution} />}
        </div>
      </div>

      {/* ── Keyword Network ──────────────────────────────────── */}
      <div className="adminCard">
        <h3>Keyword Network</h3>
        <p className="chartSubtitle">Force-directed graph of co-occurring keywords (drag nodes to explore)</p>
        {keywordsData && <KeywordNetworkGraph data={keywordsData} />}
      </div>

      {/* ── Data Table ───────────────────────────────────────── */}
      <div className="adminCard">
        <div className="tableHeader">
          <h3>Recent Router Logs</h3>
          
          <div className="tableFiltersRow">
            <input
              className="tableFilterInput"
              type="text"
              placeholder="Search message or session..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
            
            <select 
              className="tableFilterSelect"
              value={intentFilter}
              onChange={(e) => { setIntentFilter(e.target.value); setPage(1); }}
            >
              <option value="">All Intents</option>
              {INTENT_CATEGORIES.map(intent => (
                <option key={intent} value={intent}>{intent.replace(/_/g, ' ')}</option>
              ))}
            </select>

            <select 
              className="tableFilterSelect"
              value={languageFilter}
              onChange={(e) => { setLanguageFilter(e.target.value); setPage(1); }}
            >
              <option value="">All Languages</option>
              {LANGUAGE_CATEGORIES.map(lang => (
                <option key={lang} value={lang}>{lang}</option>
              ))}
            </select>

            <div className="dateFilterGroup">
              <input 
                type="date" 
                className="tableFilterDate" 
                value={dateFrom}
                onChange={(e) => { setDateFrom(e.target.value); setPage(1); }}
                title="From Date"
              />
              <span className="dateSeparator">-</span>
              <input 
                type="date" 
                className="tableFilterDate" 
                value={dateTo}
                onChange={(e) => { setDateTo(e.target.value); setPage(1); }}
                title="To Date"
              />
            </div>
            
            {(searchTerm || intentFilter || languageFilter || dateFrom || dateTo) && (
              <button className="clearFiltersBtn" onClick={clearFilters}>Clear</button>
            )}
          </div>
        </div>
        
        <div className="tableWrap">
          <table className="adminTable">
            <thead>
              <tr>
                <th>Session</th>
                <th>User Message</th>
                <th>Intent</th>
                <th>Language</th>
                <th>Latency</th>
                <th>Time</th>
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={6} style={{ textAlign: 'center', padding: '40px', color: '#64748b' }}>
                    <RefreshCw className="spin" size={24} style={{ margin: '0 auto 12px' }} />
                    <div>Loading data...</div>
                  </td>
                </tr>
              ) : logs.length === 0 ? (
                <tr>
                  <td colSpan={6} style={{ textAlign: 'center', padding: '40px', color: '#94a3b8' }}>
                    No matching entries found
                  </td>
                </tr>
              ) : (
                logs.map((row) => (
                  <tr key={row.id}>
                    <td><code>{row.session_id ? row.session_id.substring(0, 8) + '...' : 'N/A'}</code></td>
                    <td className="messageCell">{row.user_message}</td>
                    <td><span className="intentBadge">{row.intent ? row.intent.replace(/_/g, ' ') : 'Unclear'}</span></td>
                    <td>{row.language}</td>
                    <td>{row.latency_ms}ms</td>
                    <td>{new Date(row.created_at).toLocaleString()}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Controls */}
        {!loading && logs.length > 0 && (
          <div className="paginationControls">
            <div className="paginationInfo">
              Showing {((page - 1) * pageSize) + 1} to {Math.min(page * pageSize, totalCount)} of {totalCount} entries
            </div>
            
            <div className="paginationActions">
              <select 
                className="pageSizeSelect"
                value={pageSize}
                onChange={(e) => {
                  setPageSize(Number(e.target.value));
                  setPage(1);
                }}
              >
                <option value={10}>10 per page</option>
                <option value={25}>25 per page</option>
                <option value={50}>50 per page</option>
                <option value={100}>100 per page</option>
              </select>

              <div className="pageButtons">
                <button 
                  className="pageBtn" 
                  disabled={page <= 1}
                  onClick={() => setPage(p => Math.max(1, p - 1))}
                >
                  <ChevronLeft size={16} />
                </button>
                <span className="pageText">Page {page} of {Math.max(1, totalPages)}</span>
                <button 
                  className="pageBtn" 
                  disabled={page >= totalPages}
                  onClick={() => setPage(p => Math.min(totalPages, p + 1))}
                >
                  <ChevronRight size={16} />
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
