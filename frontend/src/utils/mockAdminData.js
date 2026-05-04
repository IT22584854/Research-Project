export const mockKPIs = {
  totalQueries: 12458,
  uniqueUsers: 3421,
  commonIntent: 'Symptom Check',
  modelUsed: 'Llama-3-70b-Instruct'
};

export const mockRadarData = [
  { subject: 'Symptoms', A: 120, fullMark: 150 },
  { subject: 'Appointments', A: 98, fullMark: 150 },
  { subject: 'Medication', A: 86, fullMark: 150 },
  { subject: 'General', A: 99, fullMark: 150 },
  { subject: 'Emergency', A: 45, fullMark: 150 },
  { subject: 'Insurance', A: 65, fullMark: 150 },
];

export const mockGraphData = {
  nodes: [
    { id: 'fever', group: 1, val: 20 },
    { id: 'cough', group: 1, val: 15 },
    { id: 'headache', group: 1, val: 10 },
    { id: 'paracetamol', group: 2, val: 18 },
    { id: 'doctor', group: 3, val: 25 },
    { id: 'appointment', group: 3, val: 22 },
    { id: 'clinic', group: 3, val: 12 },
  ],
  links: [
    { source: 'fever', target: 'paracetamol' },
    { source: 'fever', target: 'cough' },
    { source: 'fever', target: 'doctor' },
    { source: 'headache', target: 'paracetamol' },
    { source: 'appointment', target: 'doctor' },
    { source: 'clinic', target: 'appointment' },
    { source: 'cough', target: 'clinic' },
  ]
};

export const mockTableData = [
  { id: 1, query: "I have a bad headache and fever", lang: "en", keywords: "headache, fever", intent: "Symptom Check", rag: "fever treatments", reasoning: "Identified two primary symptom keywords" },
  { id: 2, query: "මට උණ හැදිලා තියෙන්නේ", lang: "si", keywords: "උණ", intent: "Symptom Check", rag: "fever remedies", reasoning: "Sinhala keyword for fever identified" },
  { id: 3, query: "How to book an appointment with Dr. Silva?", lang: "en", keywords: "book, appointment, doctor", intent: "Appointment", rag: "booking process", reasoning: "Explicit booking request" },
  { id: 4, query: "எனிக்கு பனி உண்டு", lang: "ta", keywords: "பனி", intent: "Symptom Check", rag: "fever info", reasoning: "Tamil keyword for fever identified" },
  { id: 5, query: "Is paracetamol safe during pregnancy?", lang: "en", keywords: "paracetamol, safe, pregnancy", intent: "Medication", rag: "paracetamol pregnancy safety", reasoning: "Drug and condition mentioned" },
];

// Agent Evaluation Page — mock data
export const mockEvaluationData = [
  { id: 1, question: "What are the symptoms of dengue?", response: "Dengue symptoms include high fever, severe headache, pain behind the eyes...", timestamp: "2026-05-04 14:32:10", evaluated: true },
  { id: 2, question: "How do I get a referral letter?", response: "You can obtain a referral letter by visiting your nearest government clinic...", timestamp: "2026-05-04 14:28:45", evaluated: false },
  { id: 3, question: "Is there a hospital near Kandy?", response: "Yes, the Kandy General Hospital is located at...", timestamp: "2026-05-04 13:55:22", evaluated: false },
  { id: 4, question: "What vaccinations do infants need?", response: "According to the Sri Lankan immunization schedule, infants should receive...", timestamp: "2026-05-04 13:40:11", evaluated: true },
  { id: 5, question: "Can I get paracetamol from a pharmacy?", response: "Yes, paracetamol is available over the counter at most pharmacies...", timestamp: "2026-05-04 12:15:33", evaluated: false },
];

// Data Scraper Page — mock data
export const mockScraperData = {
  rawAssets: 1412,
  newDocs: 20,
  cleanedDocs: 1263,
  agentExports: 1263,
  stages: [
    { name: 'Web Scraper', description: 'Scrapes government health portals for raw content', status: 'complete' },
    { name: 'Document Parser', description: 'Extracts structured text from PDF and HTML documents', status: 'complete' },
    { name: 'Data Cleaner', description: 'Removes duplicates, normalizes text, fixes encoding', status: 'running' },
    { name: 'Synthetic QA Generator', description: 'Generates question-answer pairs from cleaned documents', status: 'pending' },
    { name: 'Export to Agent', description: 'Pushes processed data to the RAG pipeline', status: 'pending' },
  ],
  documents: [
    { name: 'MOH_Guidelines_2025.pdf', source: 'health.gov.lk', status: 'cleaned', size: '2.4 MB' },
    { name: 'Vaccination_Schedule.html', source: 'epid.gov.lk', status: 'cleaned', size: '156 KB' },
    { name: 'Hospital_Directory.csv', source: 'health.gov.lk', status: 'processing', size: '890 KB' },
    { name: 'Drug_Registry_Update.pdf', source: 'nmra.gov.lk', status: 'cleaned', size: '1.1 MB' },
    { name: 'Mental_Health_FAQ.html', source: 'health.gov.lk', status: 'raw', size: '340 KB' },
  ],
};
