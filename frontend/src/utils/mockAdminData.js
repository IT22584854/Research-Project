/**
 * Mock data for the Admin Dashboard.
 * 
 * This file provides placeholder data that mirrors the schema of the
 * `agent_router_logs` Supabase table. In Phase 2, these exports will be
 * replaced by real API calls to GET /api/admin/router-logs.
 * 
 * Intent categories (14) and language categories (7) are deterministic
 * values produced by the intent classifier agent. Keywords are
 * non-deterministic (LLM-generated).
 */

// ── Intent categories (14 deterministic) ────────────────────────────
export const INTENT_CATEGORIES = [
  'General_Health_Education',
  'Facility_Locator',
  'Symptom_Information',
  'Unclear',
  'Insurance_Information',
  'Vaccine_Information',
  'Provider_Locator',
  'Test_Diagnostics',
  'Treatment_Procedure',
  'Disease_Information',
  'Medication_Information',
  'Appointment_Booking',
  'Emergency',
  'Non_Medical',
];

// ── Language categories (7 deterministic) ───────────────────────────
export const LANGUAGE_CATEGORIES = [
  'Tamil',
  'Singlish',
  'English',
  'Sinhala',
  'Sinhala English Code Mixed',
  'Tamil English Code Mixed',
  'Tamilish',
];

// ── KPI summary ─────────────────────────────────────────────────────
export const mockKPIs = {
  totalQueries: 12458,
  uniqueSessions: 3421,
  topIntent: 'General_Health_Education',
  avgLatencyMs: 1842,
};

// ── Radar chart data (intent distribution) ──────────────────────────
export const mockIntentRadarData = [
  { intent: 'General_Health_Education', count: 2840 },
  { intent: 'Symptom_Information', count: 2215 },
  { intent: 'Disease_Information', count: 1580 },
  { intent: 'Vaccine_Information', count: 1120 },
  { intent: 'Medication_Information', count: 980 },
  { intent: 'Treatment_Procedure', count: 870 },
  { intent: 'Facility_Locator', count: 720 },
  { intent: 'Provider_Locator', count: 540 },
  { intent: 'Appointment_Booking', count: 410 },
  { intent: 'Test_Diagnostics', count: 380 },
  { intent: 'Insurance_Information', count: 290 },
  { intent: 'Emergency', count: 210 },
  { intent: 'Non_Medical', count: 180 },
  { intent: 'Unclear', count: 123 },
];

// ── Donut chart data (language distribution) ────────────────────────
export const mockLanguageData = [
  { language: 'English', count: 4820 },
  { language: 'Sinhala', count: 2950 },
  { language: 'Tamil', count: 1680 },
  { language: 'Singlish', count: 1240 },
  { language: 'Sinhala English Code Mixed', count: 890 },
  { language: 'Tamil English Code Mixed', count: 540 },
  { language: 'Tamilish', count: 338 },
];

// ── Language color map ──────────────────────────────────────────────
export const LANGUAGE_COLORS = {
  'English': '#0f766e',
  'Sinhala': '#3b82f6',
  'Tamil': '#f59e0b',
  'Singlish': '#8b5cf6',
  'Sinhala English Code Mixed': '#ec4899',
  'Tamil English Code Mixed': '#ef4444',
  'Tamilish': '#10b981',
};

// ── Keyword network graph data (non-deterministic) ──────────────────
export const mockKeywordNodes = [
  { id: 'dengue', group: 'disease', size: 42 },
  { id: 'fever', group: 'symptom', size: 38 },
  { id: 'headache', group: 'symptom', size: 28 },
  { id: 'hospital', group: 'facility', size: 25 },
  { id: 'vaccination', group: 'vaccine', size: 24 },
  { id: 'diabetes', group: 'disease', size: 22 },
  { id: 'blood test', group: 'diagnostic', size: 20 },
  { id: 'pharmacy', group: 'facility', size: 18 },
  { id: 'pregnancy', group: 'condition', size: 17 },
  { id: 'malaria', group: 'disease', size: 16 },
  { id: 'covid-19', group: 'disease', size: 15 },
  { id: 'chest pain', group: 'symptom', size: 14 },
  { id: 'appointment', group: 'action', size: 13 },
  { id: 'insurance', group: 'admin', size: 12 },
  { id: 'clinic', group: 'facility', size: 11 },
  { id: 'cough', group: 'symptom', size: 10 },
  { id: 'X-ray', group: 'diagnostic', size: 9 },
  { id: 'rash', group: 'symptom', size: 8 },
  { id: 'asthma', group: 'disease', size: 7 },
  { id: 'paracetamol', group: 'medication', size: 6 },
];

export const mockKeywordLinks = [
  { source: 'dengue', target: 'fever', value: 12 },
  { source: 'dengue', target: 'headache', value: 8 },
  { source: 'dengue', target: 'rash', value: 6 },
  { source: 'dengue', target: 'blood test', value: 5 },
  { source: 'fever', target: 'headache', value: 9 },
  { source: 'fever', target: 'hospital', value: 7 },
  { source: 'fever', target: 'paracetamol', value: 4 },
  { source: 'diabetes', target: 'blood test', value: 6 },
  { source: 'diabetes', target: 'clinic', value: 4 },
  { source: 'vaccination', target: 'clinic', value: 5 },
  { source: 'vaccination', target: 'appointment', value: 4 },
  { source: 'vaccination', target: 'covid-19', value: 7 },
  { source: 'covid-19', target: 'cough', value: 5 },
  { source: 'covid-19', target: 'hospital', value: 6 },
  { source: 'pregnancy', target: 'clinic', value: 3 },
  { source: 'pregnancy', target: 'hospital', value: 4 },
  { source: 'malaria', target: 'fever', value: 6 },
  { source: 'malaria', target: 'blood test', value: 4 },
  { source: 'chest pain', target: 'hospital', value: 8 },
  { source: 'chest pain', target: 'X-ray', value: 3 },
  { source: 'asthma', target: 'cough', value: 4 },
  { source: 'asthma', target: 'pharmacy', value: 3 },
  { source: 'insurance', target: 'hospital', value: 3 },
  { source: 'insurance', target: 'appointment', value: 2 },
];

// ── Keyword group colors ────────────────────────────────────────────
export const KEYWORD_GROUP_COLORS = {
  disease: '#ef4444',
  symptom: '#f59e0b',
  facility: '#3b82f6',
  vaccine: '#10b981',
  diagnostic: '#8b5cf6',
  condition: '#ec4899',
  action: '#6366f1',
  admin: '#64748b',
  medication: '#14b8a6',
};

// ── Table data (recent log entries) ─────────────────────────────────
export const mockTableData = [
  { id: 1, session_id: 'sess_a1b2', user_message: 'What are the symptoms of dengue?', intent: 'Symptom_Information', language: 'English', latency_ms: 1240, created_at: '2026-05-04T10:12:00Z' },
  { id: 2, session_id: 'sess_c3d4', user_message: 'डेंगू का इलाज कैसे करें?', intent: 'Treatment_Procedure', language: 'Sinhala', latency_ms: 2100, created_at: '2026-05-04T10:15:30Z' },
  { id: 3, session_id: 'sess_e5f6', user_message: 'Where is the nearest hospital?', intent: 'Facility_Locator', language: 'English', latency_ms: 980, created_at: '2026-05-04T10:18:45Z' },
  { id: 4, session_id: 'sess_g7h8', user_message: 'covid vaccine eka koheda ganne?', intent: 'Vaccine_Information', language: 'Singlish', latency_ms: 1560, created_at: '2026-05-04T10:22:10Z' },
  { id: 5, session_id: 'sess_i9j0', user_message: 'தடுப்பூசி எங்கே போடலாம்?', intent: 'Vaccine_Information', language: 'Tamil', latency_ms: 1890, created_at: '2026-05-04T10:25:00Z' },
  { id: 6, session_id: 'sess_k1l2', user_message: 'I have chest pain and difficulty breathing', intent: 'Emergency', language: 'English', latency_ms: 780, created_at: '2026-05-04T10:28:30Z' },
  { id: 7, session_id: 'sess_m3n4', user_message: 'mama doctor kenekta appointment ekak ගන්න ඕනේ', intent: 'Appointment_Booking', language: 'Sinhala English Code Mixed', latency_ms: 2340, created_at: '2026-05-04T10:31:15Z' },
  { id: 8, session_id: 'sess_o5p6', user_message: 'What does my blood test report mean?', intent: 'Test_Diagnostics', language: 'English', latency_ms: 1670, created_at: '2026-05-04T10:35:00Z' },
  { id: 9, session_id: 'sess_q7r8', user_message: 'hello how are you', intent: 'Non_Medical', language: 'English', latency_ms: 420, created_at: '2026-05-04T10:38:20Z' },
  { id: 10, session_id: 'sess_s9t0', user_message: 'asdfgh jkl', intent: 'Unclear', language: 'English', latency_ms: 310, created_at: '2026-05-04T10:40:00Z' },
];

// ── Evaluation page placeholder data ────────────────────────────────
export const mockEvaluationData = {
  totalEvaluations: 856,
  passRate: 92.4,
  avgScore: 4.2,
  pendingReview: 34,
};

// ── Scraper page placeholder data ───────────────────────────────────
export const mockScraperData = {
  totalDocuments: 2847,
  lastScrapeDate: '2026-05-03T14:30:00Z',
  activeSources: 12,
  syntheticGenerated: 1240,
};
