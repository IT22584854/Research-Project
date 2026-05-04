import React from 'react';
import { Play, CheckCircle } from 'lucide-react';
import { mockEvaluationData } from '../../utils/mockAdminData';

export default function AgentEvaluationPage() {
  return (
    <div className="p-6 lg:p-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-gray-900">Agent Response Evaluation</h2>
          <p className="text-sm text-gray-500 mt-1">Review and evaluate agent turn logs from Supabase</p>
        </div>
        <button className="admin-btn admin-btn-primary">
          <Play size={14} />
          Evaluate All
        </button>
      </div>

      {/* KPI Row */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <KPI title="Total Turns" value="2,847" />
        <KPI title="Evaluated" value="1,203" />
        <KPI title="Pending" value="1,644" />
        <KPI title="Avg Score" value="4.2 / 5" />
      </div>

      {/* Table + Evaluation Result */}
      <div className="flex flex-col xl:flex-row gap-6">
        {/* Table */}
        <div className="flex-1 admin-card overflow-hidden">
          <div className="px-6 py-4 border-b border-gray-100">
            <h3 className="text-sm font-semibold text-gray-700">agent_turn_logs — Final Responses</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left min-w-[700px]">
              <thead className="bg-gray-50 text-xs uppercase text-gray-500 border-b border-gray-100">
                <tr>
                  <th className="px-6 py-3 font-medium">Agent Question</th>
                  <th className="px-6 py-3 font-medium">Response</th>
                  <th className="px-6 py-3 font-medium">Timestamp</th>
                  <th className="px-6 py-3 font-medium">Status</th>
                  <th className="px-6 py-3 font-medium">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-50">
                {mockEvaluationData.map((row) => (
                  <tr key={row.id} className="hover:bg-gray-50/50 transition-colors">
                    <td className="px-6 py-4 text-gray-900 font-medium max-w-[200px] truncate" title={row.question}>{row.question}</td>
                    <td className="px-6 py-4 text-gray-600 max-w-[250px] truncate" title={row.response}>{row.response}</td>
                    <td className="px-6 py-4 text-gray-400 text-xs whitespace-nowrap">{row.timestamp}</td>
                    <td className="px-6 py-4">
                      <span className={`inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium ${
                        row.evaluated 
                          ? 'bg-emerald-50 text-emerald-700' 
                          : 'bg-amber-50 text-amber-700'
                      }`}>
                        {row.evaluated ? 'Evaluated' : 'Pending'}
                      </span>
                    </td>
                    <td className="px-6 py-4">
                      <button className="admin-btn text-xs py-1.5 px-3">
                        <CheckCircle size={12} />
                        Evaluate
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Evaluation Result Panel */}
        <div className="w-full xl:w-72 admin-card p-5 shrink-0 self-start">
          <h3 className="text-sm font-semibold text-gray-700 mb-4 pb-3 border-b border-gray-100">Evaluation Result</h3>
          <div className="space-y-4 text-sm text-gray-500">
            <p className="text-center py-8 text-gray-400">Select a row to view evaluation details</p>
          </div>
        </div>
      </div>
    </div>
  );
}

function KPI({ title, value }) {
  return (
    <div className="admin-card p-5">
      <p className="text-xs font-medium text-gray-500 uppercase tracking-wider">{title}</p>
      <p className="text-2xl font-semibold text-gray-900 mt-1">{value}</p>
    </div>
  );
}
