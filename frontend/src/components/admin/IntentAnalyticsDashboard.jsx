import { useEffect, useRef, useState } from 'react';
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
} from '../../utils/mockAdminData';
import { Activity, Users, Zap, TrendingUp, Download } from 'lucide-react';

/* ════════════════════════════════════════════════════════════════════
   D3 Radar Chart — 14 intent types
   ════════════════════════════════════════════════════════════════════ */
function IntentRadarChart() {
  const svgRef = useRef(null);
  const containerRef = useRef(null);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = Math.min(width, 420);
    const margin = 60;
    const radius = Math.min(width, height) / 2 - margin;
    const levels = 5;
    const maxValue = Math.max(...mockIntentRadarData.map((d) => d.count));

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    svg.attr('width', width).attr('height', height);

    const g = svg.append('g').attr('transform', `translate(${width / 2}, ${height / 2})`);

    const angleSlice = (2 * Math.PI) / mockIntentRadarData.length;
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
    mockIntentRadarData.forEach((d, i) => {
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
      .datum(mockIntentRadarData)
      .attr('d', lineGen)
      .attr('fill', 'rgba(15, 118, 110, 0.15)')
      .attr('stroke', '#0f766e')
      .attr('stroke-width', 2);

    // Data points
    mockIntentRadarData.forEach((d, i) => {
      const angle = angleSlice * i - Math.PI / 2;
      g.append('circle')
        .attr('cx', rScale(d.count) * Math.cos(angle))
        .attr('cy', rScale(d.count) * Math.sin(angle))
        .attr('r', 4)
        .attr('fill', '#0f766e')
        .attr('stroke', '#fff')
        .attr('stroke-width', 2);
    });
  }, []);

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={svgRef} />
    </div>
  );
}

/* ════════════════════════════════════════════════════════════════════
   D3 Donut Chart — 7 languages
   ════════════════════════════════════════════════════════════════════ */
function LanguageDonutChart() {
  const svgRef = useRef(null);
  const containerRef = useRef(null);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = Math.min(width, 420);
    const radius = Math.min(width, height) / 2 - 20;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    svg.attr('width', width).attr('height', height);

    const g = svg.append('g').attr('transform', `translate(${width / 2}, ${height / 2})`);

    const total = d3.sum(mockLanguageData, (d) => d.count);
    const pie = d3.pie().value((d) => d.count).sort(null).padAngle(0.02);
    const arc = d3.arc().innerRadius(radius * 0.55).outerRadius(radius);
    const arcHover = d3.arc().innerRadius(radius * 0.55).outerRadius(radius + 6);

    const arcs = g.selectAll('.arc')
      .data(pie(mockLanguageData))
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

    // Legend below chart
    const legend = svg.append('g')
      .attr('transform', `translate(${width / 2 - 120}, ${height - 10})`);

    // We'll render the legend outside SVG in HTML for better wrapping
  }, []);

  return (
    <div ref={containerRef} style={{ width: '100%' }}>
      <svg ref={svgRef} />
      <div className="donutLegend">
        {mockLanguageData.map((d) => (
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
function KeywordNetworkGraph() {
  const svgRef = useRef(null);
  const containerRef = useRef(null);

  useEffect(() => {
    if (!svgRef.current || !containerRef.current) return;

    const container = containerRef.current;
    const width = container.clientWidth;
    const height = 400;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    svg.attr('width', width).attr('height', height).attr('viewBox', `0 0 ${width} ${height}`);

    // Deep clone data so D3 mutation doesn't break React re-renders
    const nodes = mockKeywordNodes.map((d) => ({ ...d }));
    const links = mockKeywordLinks.map((d) => ({ ...d }));

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
  }, []);

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
  const [filter, setFilter] = useState('');

  const filteredRows = mockTableData.filter((row) => {
    if (!filter) return true;
    const term = filter.toLowerCase();
    return (
      row.user_message.toLowerCase().includes(term) ||
      row.intent.toLowerCase().includes(term) ||
      row.language.toLowerCase().includes(term) ||
      row.session_id.toLowerCase().includes(term)
    );
  });

  return (
    <div className="dashboardPage">
      <div className="dashboardHeader">
        <div>
          <h2>Intent Routing Analytics</h2>
          <p>Real-time analysis of intent classifier routing decisions</p>
        </div>
        <button className="adminBtn" type="button">
          <Download size={16} />
          Export CSV
        </button>
      </div>

      {/* ── KPI Cards ────────────────────────────────────────── */}
      <div className="kpiGrid">
        <div className="kpiCard">
          <div className="kpiIcon" style={{ backgroundColor: 'rgba(15, 118, 110, 0.1)', color: '#0f766e' }}>
            <Activity size={20} />
          </div>
          <div className="kpiBody">
            <span className="kpiValue">{mockKPIs.totalQueries.toLocaleString()}</span>
            <span className="kpiLabel">Total Queries</span>
          </div>
        </div>
        <div className="kpiCard">
          <div className="kpiIcon" style={{ backgroundColor: 'rgba(59, 130, 246, 0.1)', color: '#3b82f6' }}>
            <Users size={20} />
          </div>
          <div className="kpiBody">
            <span className="kpiValue">{mockKPIs.uniqueSessions.toLocaleString()}</span>
            <span className="kpiLabel">Unique Sessions</span>
          </div>
        </div>
        <div className="kpiCard">
          <div className="kpiIcon" style={{ backgroundColor: 'rgba(245, 158, 11, 0.1)', color: '#f59e0b' }}>
            <TrendingUp size={20} />
          </div>
          <div className="kpiBody">
            <span className="kpiValue">{mockKPIs.topIntent.replace(/_/g, ' ')}</span>
            <span className="kpiLabel">Top Intent</span>
          </div>
        </div>
        <div className="kpiCard">
          <div className="kpiIcon" style={{ backgroundColor: 'rgba(139, 92, 246, 0.1)', color: '#8b5cf6' }}>
            <Zap size={20} />
          </div>
          <div className="kpiBody">
            <span className="kpiValue">{mockKPIs.avgLatencyMs.toLocaleString()}ms</span>
            <span className="kpiLabel">Avg Latency</span>
          </div>
        </div>
      </div>

      {/* ── Charts Row ───────────────────────────────────────── */}
      <div className="chartsRow">
        <div className="adminCard">
          <h3>Intent Distribution</h3>
          <p className="chartSubtitle">Radar view of 14 intent categories</p>
          <IntentRadarChart />
        </div>
        <div className="adminCard">
          <h3>Language Distribution</h3>
          <p className="chartSubtitle">Breakdown of 7 supported languages</p>
          <LanguageDonutChart />
        </div>
      </div>

      {/* ── Keyword Network ──────────────────────────────────── */}
      <div className="adminCard">
        <h3>Keyword Network</h3>
        <p className="chartSubtitle">Force-directed graph of co-occurring keywords (drag nodes to explore)</p>
        <KeywordNetworkGraph />
      </div>

      {/* ── Data Table ───────────────────────────────────────── */}
      <div className="adminCard">
        <div className="tableHeader">
          <h3>Recent Router Logs</h3>
          <input
            className="tableFilter"
            type="text"
            placeholder="Filter by message, intent, or language…"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
          />
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
              {filteredRows.map((row) => (
                <tr key={row.id}>
                  <td><code>{row.session_id}</code></td>
                  <td className="messageCell">{row.user_message}</td>
                  <td><span className="intentBadge">{row.intent.replace(/_/g, ' ')}</span></td>
                  <td>{row.language}</td>
                  <td>{row.latency_ms}ms</td>
                  <td>{new Date(row.created_at).toLocaleTimeString()}</td>
                </tr>
              ))}
              {filteredRows.length === 0 && (
                <tr>
                  <td colSpan={6} style={{ textAlign: 'center', padding: '24px', color: '#94a3b8' }}>
                    No matching entries found
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
