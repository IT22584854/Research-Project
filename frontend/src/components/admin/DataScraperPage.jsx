import React from 'react';
import { Database, FileText, Sparkles, Upload } from 'lucide-react';
import { mockScraperData } from '../../utils/mockAdminData';

export default function DataScraperPage() {
  return (
    <div className="p-6 lg:p-8 space-y-6">
      {/* Header */}
      <div>
        <h2 className="text-lg font-semibold text-gray-900">Data Scraper & Synthetic Data Generator</h2>
        <p className="text-sm text-gray-500 mt-1">Demonstratable with each script on stages</p>
      </div>

      {/* KPI Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <KPI title="Raw Assets" value={mockScraperData.rawAssets} icon={<Database size={18} className="text-blue-500" />} />
        <KPI title="New Docs Added" value={mockScraperData.newDocs} icon={<Upload size={18} className="text-emerald-500" />} />
        <KPI title="Cleaned Docs" value={mockScraperData.cleanedDocs} icon={<FileText size={18} className="text-amber-500" />} />
        <KPI title="Agent Exports" value={mockScraperData.agentExports} icon={<Sparkles size={18} className="text-purple-500" />} />
      </div>

      {/* Pipeline Stages */}
      <div className="admin-card p-6">
        <h3 className="text-sm font-semibold text-gray-700 mb-4">Pipeline Stages</h3>
        <div className="space-y-3">
          {mockScraperData.stages.map((stage, i) => (
            <div key={i} className="flex items-center gap-4 p-4 rounded-lg bg-gray-50 border border-gray-100">
              <div className={`w-2.5 h-2.5 rounded-full shrink-0 ${
                stage.status === 'complete' ? 'bg-emerald-500' : 
                stage.status === 'running' ? 'bg-amber-500 animate-pulse' : 'bg-gray-300'
              }`} />
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-gray-900">{stage.name}</p>
                <p className="text-xs text-gray-400 mt-0.5">{stage.description}</p>
              </div>
              <span className={`text-xs font-medium px-2.5 py-1 rounded-full ${
                stage.status === 'complete' ? 'bg-emerald-50 text-emerald-700' : 
                stage.status === 'running' ? 'bg-amber-50 text-amber-700' : 'bg-gray-100 text-gray-500'
              }`}>
                {stage.status}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Data Preview Table */}
      <div className="admin-card overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-100">
          <h3 className="text-sm font-semibold text-gray-700">Data Preview</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left min-w-[600px]">
            <thead className="bg-gray-50 text-xs uppercase text-gray-500 border-b border-gray-100">
              <tr>
                <th className="px-6 py-3 font-medium">Document</th>
                <th className="px-6 py-3 font-medium">Source</th>
                <th className="px-6 py-3 font-medium">Status</th>
                <th className="px-6 py-3 font-medium">Size</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-50">
              {mockScraperData.documents.map((doc, i) => (
                <tr key={i} className="hover:bg-gray-50/50 transition-colors">
                  <td className="px-6 py-4 text-gray-900 font-medium">{doc.name}</td>
                  <td className="px-6 py-4 text-gray-500">{doc.source}</td>
                  <td className="px-6 py-4">
                    <span className={`inline-flex px-2.5 py-1 rounded-full text-xs font-medium ${
                      doc.status === 'cleaned' ? 'bg-emerald-50 text-emerald-700' :
                      doc.status === 'processing' ? 'bg-amber-50 text-amber-700' : 'bg-gray-100 text-gray-500'
                    }`}>
                      {doc.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-gray-400">{doc.size}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

function KPI({ title, value, icon }) {
  return (
    <div className="admin-card p-5 flex items-start gap-4">
      <div className="w-10 h-10 rounded-lg bg-gray-50 flex items-center justify-center shrink-0">
        {icon}
      </div>
      <div>
        <p className="text-xs font-medium text-gray-500 uppercase tracking-wider">{title}</p>
        <p className="text-2xl font-semibold text-gray-900 mt-0.5">{value.toLocaleString()}</p>
      </div>
    </div>
  );
}
